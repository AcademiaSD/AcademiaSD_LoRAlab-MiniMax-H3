r"""
AcademiaSD LoRAlab MiniMax-H3 - Auto-captioning del dataset.
AcademiaSD LoRAlab MiniMax-H3 - Dataset auto-captioning.

QUE HACE / WHAT IT DOES
-----------------------
Genera un .txt por cada imagen del dataset con la palabra trigger al principio,
usando Qwen3-VL-4B-Instruct. El modelo se descarga solo la primera vez.

Writes one .txt per dataset image with the trigger word first, using
Qwen3-VL-4B-Instruct. The model downloads itself on first use.

POR QUE UN MODELO APARTE / WHY A SEPARATE MODEL
------------------------------------------------
Lo primero que se intento fue reutilizar el Qwen3-VL-32B que ya trae el repo NF4,
para no descargar nada. No sale a cuenta.

El pre-cache lo carga MUTILADO a proposito: 50 de 64 capas y la lm_head
sustituida por Identity, porque H3 solo consume hidden_states[50] y la cabeza son
~3,1 GB de peso muerto. Ese modelo no puede emitir ni un token. Cargarlo entero
son 17,1 GiB, que en una tarjeta de 16 GB desbordan a memoria compartida.
Medido: 242 segundos por imagen. Inviable.

El 4B en NF4 son ~3 GB, entra entero en cualquier tarjeta desde 6 GB y no toca
memoria compartida. La descarga son ~8 GB una sola vez.

Reusing the Qwen3-VL-32B already inside the NF4 repo was tried first, to avoid
any download. It does not pay off. The pre-cache loads it MUTILATED on purpose:
50 of 64 layers and lm_head replaced by Identity, since H3 only consumes
hidden_states[50] and the head is ~3.1 GB of dead weight -- that model cannot
emit a single token. Loading it whole is 17.1 GiB, which on a 16 GB card spills
into shared memory. Measured: 242 seconds per image. The 4B in NF4 is ~3 GB,
fits entirely on any card from 6 GB up, and never touches shared memory.

USO / USAGE
-----------
    venv\Scripts\python.exe 0_caption_MiniMaxH3.py

Lee caption_settings.json. Sin ese fichero usa los valores de DEFAULTS y toma la
ruta del dataset y el trigger de pre_cache_settings.json.
Reads caption_settings.json; falls back to DEFAULTS and to pre_cache_settings.json
for the dataset path and the trigger word.
"""

import json
import os
import sys
import time

import torch


# =============================================================================
# CONFIG
# =============================================================================

DEFAULTS = {
    "dataset_path": "./dataset",
    "trigger_word": "",

    # Modelo dedicado al captioning. NO es el text encoder de H3.
    #
    # Se probo primero reutilizar el Qwen3-VL-32B que ya trae el repo NF4, para
    # no descargar nada. No funciona: para GENERAR texto hacen falta sus 64 capas
    # y su lm_head, que son 17,1 GiB, y en una tarjeta de 16 GB eso desborda a
    # memoria compartida. Medido: 242 segundos por imagen. Inviable.
    #
    # El 4B en NF4 son ~3 GB, entra entero en cualquier tarjeta a partir de 6 GB
    # y no toca memoria compartida.
    #
    # Dedicated captioning model. NOT the H3 text encoder. Reusing the
    # Qwen3-VL-32B already inside the NF4 repo was tried first, to avoid any
    # download. It does not work: GENERATING text needs its 64 layers and its
    # lm_head, 17.1 GiB, which on a 16 GB card spills into shared memory.
    # Measured: 242 seconds per image. The 4B in NF4 is ~3 GB, fits entirely on
    # any card from 6 GB up, and never touches shared memory.
    "captioner_repo": "Qwen/Qwen3-VL-4B-Instruct",
    "captioner_dir": "./Qwen3-VL-4B-Instruct",

    # 4 bits deja el modelo en ~3 GB. Ponlo a False para bf16 (~8 GB) si te
    # sobra VRAM y quieres el maximo de calidad en la descripcion.
    # 4-bit keeps the model at ~3 GB. Set to False for bf16 (~8 GB) if you have
    # VRAM to spare and want the best description quality.
    "captioner_4bit": True,

    # El prompt que se le da al modelo por cada imagen. Editable desde la GUI.
    # The prompt handed to the model for each image. Editable from the GUI.
    "caption_prompt": (
        "Describe this image for training a text-to-video model. One single "
        "paragraph, plain prose, no lists and no preamble. Cover: the subject "
        "and what they are doing, hair and clothing, the framing and camera "
        "angle, the setting, and the lighting. Do not invent a name and do not "
        "mention that this is an image or a photo."
    ),

    # 80 tokens son unas 60 palabras: dos o tres frases, que es justo lo que
    # necesita un caption. Y hay un techo mas alto que manda sobre este: lo que
    # pase de max_seq_len (pre_cache_settings.json) se trunca al hacer la
    # pre-cache, asi que generar mas largo es tiempo de GPU tirado.
    # 80 tokens is about 60 words: two or three sentences, which is what a
    # caption needs. A higher ceiling overrides this one: anything past
    # max_seq_len (pre_cache_settings.json) is truncated during the pre-cache,
    # so generating longer is wasted GPU time.
    "max_new_tokens": 80,
    "temperature": 0.3,

    # Lado largo al que se reduce la imagen ANTES de dársela al modelo. Qwen3-VL
    # usa resolucion dinamica: cuanto mas grande la imagen, mas tokens de vision
    # genera, y ese coste se paga en el prefill antes de escribir la primera
    # palabra. Para describir sujeto, ropa, encuadre e iluminacion, 512 px
    # sobran; lo que se pierde a esa escala es detalle fino que ademas no
    # queremos en un caption. 0 desactiva el reescalado.
    #
    # LA IMAGEN DEL DATASET NO SE TOCA: el reescalado es solo sobre la copia en
    # memoria. Este script nunca abre una imagen en modo escritura.
    #
    # Longest side the image is shrunk to BEFORE handing it to the model.
    # Qwen3-VL uses dynamic resolution: a bigger image means more vision tokens,
    # paid during prefill before the first word is written. 512 px is plenty to
    # describe subject, clothing, framing and lighting. 0 disables the resize.
    # THE DATASET IMAGE IS NOT TOUCHED: the resize applies to the in-memory copy
    # only; this script never opens an image for writing.
    "max_image_side": 512,

    # ---- clips de video ----
    # Prompt aparte: para un clip lo que importa es el MOVIMIENTO. Con el prompt
    # de imagen el modelo describe el primer fotograma y el LoRA aprende estilo
    # pero no efecto, que es justo lo que se quiere entrenar.
    # Separate prompt: for a clip what matters is the MOTION. With the image
    # prompt the model describes the first frame, and the LoRA learns style but
    # not the effect, which is the whole point.
    # El limite de palabras va DENTRO del prompt, no solo en max_new_tokens.
    # Sin el, el modelo escribe una descripcion larga del CONTENIDO y se queda
    # sin presupuesto justo al llegar al efecto: los captions salian cortados a
    # media frase ("...as the", "...The hands rotate the miniature tank") y esa
    # frase rota es lo que aprenderia el LoRA.
    #
    # The word limit goes INSIDE the prompt, not just in max_new_tokens. Without
    # it the model writes a long description of the CONTENT and runs out of
    # budget right as it reaches the effect: captions came out cut mid-sentence,
    # and that broken sentence is what the LoRA would learn.
    "caption_video_prompt": (
        "Describe this video clip in at most 40 words, as one short paragraph, "
        "for training a text-to-video model. Focus on what CHANGES over time: "
        "the transition, the transformation, and how the subject or the camera "
        "moves. Name the subject in a few words, then spend the rest on the "
        "effect. Do not describe details that stay the same. No preamble, no "
        "lists, and do not mention that this is a video or a clip."
    ),

    # Fotogramas que se le pasan al captioner, repartidos por todo el clip. No
    # hacen falta los 73: con 12 se ve perfectamente que ocurre, y el processor
    # tiene un presupuesto de pixeles x fotogramas (25.165.824) que 73 fotogramas
    # a 512x512 casi agotan antes de empezar.
    # Frames handed to the captioner, spread across the clip. All 73 are not
    # needed: 12 show what happens, and the processor has a pixels x frames
    # budget (25,165,824) that 73 frames at 512x512 nearly exhaust.
    "caption_video_frames": 12,

    # False = no toca las imagenes que ya tienen .txt con contenido.
    # False = leaves alone any image that already has a non-empty .txt.
    "overwrite": True,
}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv", ".avi")
MEDIA_EXTS = IMAGE_EXTS + VIDEO_EXTS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "caption_settings.json")
PRECACHE_CONFIG = os.path.join(BASE_DIR, "pre_cache_settings.json")


def L(en, es):
    """Ingles primero, siempre. / English first, always."""
    return "{} / {}".format(en, es)


def LF(en, es, *args):
    """L() con formato, formateando cada idioma por separado.

    L(a, b).format(x) es una trampa: al unir las dos mitades los marcadores de
    ambas comparten la misma lista de argumentos, asi que hay que pasar cada
    valor DOS veces y basta olvidarse una para que reviente en tiempo de
    ejecucion. LF formatea cada mitad por su cuenta y el problema desaparece.

    L() with formatting, formatting each language separately. L(a, b).format(x)
    is a trap: joining the halves makes both share one argument list, so every
    value must be passed TWICE and forgetting once blows up at run time.
    """
    return "{} / {}".format(en.format(*args), es.format(*args))


def log(msg="", flush=True):
    print(msg, flush=flush)


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default if default is not None else {}


def load_config():
    cfg = dict(DEFAULTS)
    precache = read_json(PRECACHE_CONFIG, {})
    # num_frames viene de la pre-cache a proposito: decide QUE TROZO del clip se
    # va a entrenar, y el caption tiene que describir ese trozo y no otro.
    # num_frames comes from the pre-cache on purpose: it decides WHICH SLICE of
    # the clip gets trained, and the caption must describe that slice.
    for key in ("dataset_path", "trigger_word", "num_frames"):
        if precache.get(key):
            cfg[key] = precache[key]
    cfg.update({k: v for k, v in read_json(CONFIG_PATH, {}).items() if v != ""})
    return cfg


# =============================================================================
# DESCARGA Y CARGA DEL CAPTIONER / CAPTIONER DOWNLOAD AND LOADING
# =============================================================================

# Ficheros minimos que tiene que haber para dar la descarga por buena. Una
# descarga cortada deja la carpeta creada y a medias, y eso no puede pasar por
# valida: from_pretrained fallaria mucho mas tarde y con un error opaco.
# Minimum files required to consider the download complete. An interrupted
# download leaves the folder created and half-full, and that must not pass as
# valid: from_pretrained would fail much later with an opaque error.
REQUIRED_FILES = ("config.json", "preprocessor_config.json", "tokenizer_config.json")


def captioner_is_present(directory):
    if not os.path.isdir(directory):
        return False
    for name in REQUIRED_FILES:
        if not os.path.isfile(os.path.join(directory, name)):
            return False
    weights = [f for f in os.listdir(directory) if f.endswith(".safetensors")]
    return bool(weights)


def ensure_captioner(cfg):
    """Descarga el modelo de captioning si no esta ya en disco.

    Downloads the captioning model if it is not on disk already.
    """
    directory = cfg["captioner_dir"]

    if captioner_is_present(directory):
        log(LF("[CAPTION] Model found: {}",
               "[CAPTION] Modelo encontrado: {}", os.path.abspath(directory)))
        return directory

    log("=" * 78)
    log(LF("[CAPTION] The captioning model is not on disk: downloading {}.",
           "[CAPTION] El modelo de captioning no esta en disco: descargando {}.",
           cfg["captioner_repo"]))
    log(LF("[CAPTION] Destination: {}",
           "[CAPTION] Destino: {}", os.path.abspath(directory)))
    log(L("[CAPTION] About 8 GB. It happens once; later runs reuse it.",
          "[CAPTION] Unos 8 GB. Ocurre una sola vez; las siguientes ejecuciones "
          "lo reutilizan."))
    log("=" * 78)

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=cfg["captioner_repo"], local_dir=directory)
    except Exception as exc:
        raise RuntimeError(
            LF("Could not download {}: {}",
               "No se pudo descargar {}: {}", cfg["captioner_repo"], exc))

    if not captioner_is_present(directory):
        raise RuntimeError(
            LF("The download finished but {} is incomplete. Delete it and retry.",
               "La descarga termino pero {} esta incompleta. Borrala y reintenta.",
               os.path.abspath(directory)))

    log(L("[CAPTION] Download complete.", "[CAPTION] Descarga completada."))
    return directory


def load_captioner(cfg):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    directory = ensure_captioner(cfg)     # ya descargado en main(); aqui solo resuelve la ruta
    four_bit = bool(cfg["captioner_4bit"])

    log("=" * 78)
    log(LF("[CAPTION] Loading {} in {}.",
           "[CAPTION] Cargando {} en {}.",
           os.path.basename(os.path.abspath(directory)),
           "NF4 (~3 GB)" if four_bit else "bf16 (~8 GB)"))
    log("=" * 78)

    kwargs = {"dtype": torch.bfloat16, "device_map": "cuda:0"}
    if four_bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = Qwen3VLForConditionalGeneration.from_pretrained(directory, **kwargs)
    model.eval()

    processor = AutoProcessor.from_pretrained(directory)

    try:
        used = torch.cuda.memory_allocated() / (1024 ** 3)
        log(LF("[CAPTION] Ready. VRAM: {:.2f} GiB.",
               "[CAPTION] Listo. VRAM: {:.2f} GiB.", used))
    except Exception:
        pass

    return model, processor


# =============================================================================
# GENERACION / GENERATION
# =============================================================================

def shrink_for_captioner(image, max_side):
    """Reduce la copia EN MEMORIA de la imagen. El fichero del dataset no se toca.

    Qwen3-VL usa resolucion dinamica, asi que una imagen de 1024 px genera
    muchisimos mas tokens de vision que una de 512, y todos se pagan en el
    prefill antes de escribir la primera palabra. Con el modelo desbordando a
    memoria compartida ese coste se multiplica.

    Shrinks the IN-MEMORY copy. The dataset file is never touched. Qwen3-VL uses
    dynamic resolution, so a 1024 px image produces far more vision tokens than a
    512 px one, all paid during prefill before the first word is written. With
    the model spilling into shared memory that cost multiplies.
    """
    if max_side <= 0:
        return image

    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image

    scale = max_side / float(longest)
    from PIL import Image as _Image
    return image.resize((max(1, int(round(width * scale))),
                         max(1, int(round(height * scale)))),
                        _Image.LANCZOS)


def find_ffmpeg():
    """ffmpeg del PATH. Es el unico decodificador disponible: el venv no trae
    decord, av, cv2 ni imageio.
    ffmpeg from PATH: the only decoder available, since the venv ships no
    decord, av, cv2 or imageio."""
    import shutil
    return shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")


# Misma geometria que 1_pre_cache_MiniMaxH3.py: H3 solo admite 17n+5
# fotogramas (5, 22, 39, 56, 73, 90, 107, 124, 141, 158...). Se repite aqui en
# vez de importarla porque los dos scripts se lanzan por separado y no comparten
# modulo; si alguna vez cambia, cambia en los dos sitios.
# Same geometry as the pre-cache: H3 only accepts 17n+5 frames. Duplicated here
# rather than imported because the two scripts run independently.
H3_CLIP_LENGTH = 17
H3_BASE_FRAMES = 5


def h3_valid_frames(count, target=0):
    """Fotogramas que la pre-cache usara de un clip de `count`.

    Devuelve el mayor 17n+5 que cabe, o el que pida `target` si es menor.
    Returns the largest 17n+5 that fits, or `target`'s if that is smaller.
    """
    if count < H3_BASE_FRAMES:
        return 0
    mayor = H3_CLIP_LENGTH * ((count - H3_BASE_FRAMES) // H3_CLIP_LENGTH) + H3_BASE_FRAMES
    if target and target >= H3_BASE_FRAMES:
        objetivo = H3_CLIP_LENGTH * ((int(target) - H3_BASE_FRAMES) // H3_CLIP_LENGTH) + H3_BASE_FRAMES
        return min(objetivo, mayor)
    return mayor


def caption_window(path, cfg):
    """Cuantos fotogramas del clip vera de verdad el entrenamiento.

    La pre-cache lee los N PRIMEROS fotogramas, no el clip entero, asi que este
    es el unico tramo sobre el que el caption puede decir la verdad.

    How many of the clip's frames training will actually see. The pre-cache reads
    the FIRST N frames, not the whole clip, so this is the only stretch the
    caption can tell the truth about.
    """
    total = video_frame_count(path) or 0
    return h3_valid_frames(total, cfg.get("num_frames", 0)) or total


def extract_frames(path, count, max_side, window=0):
    """Saca `count` fotogramas repartidos por el tramo entrenable del clip.

    `window` limita el reparto a los `window` primeros fotogramas, que son los
    que la pre-cache va a entrenar. Antes se repartian por el clip ENTERO, y esa
    era una trampa silenciosa: con un clip de 161 fotogramas entrenado a 107, el
    modelo de captioning veia el final de la transformacion y lo describia,
    mientras que el entrenamiento se quedaba en el 66%. El resultado es que se le
    ensena al modelo que el texto del estado final corresponde a los pixeles de
    un estado intermedio, y al generar se planta ahi.

    `window` limits the spread to the first `window` frames, the ones the
    pre-cache will train on. They used to be spread across the WHOLE clip, which
    was a silent trap: on a 161 frame clip trained at 107, the captioner saw the
    end of the transformation and described it while training stopped at 66% --
    teaching the model that the final state's text goes with an intermediate
    state's pixels, so generation stalls there.

    Se piden por posicion y no con `fps=`, porque un ritmo fijo depende de la
    duracion y da un numero distinto de fotogramas en cada clip. Aqui hace falta
    exactamente el mismo numero siempre.

    Pulls `count` frames spread across the clip as PIL images. They are selected
    by position rather than with `fps=`, because a fixed rate depends on the
    duration and yields a different count per clip; here the count must be the
    same every time.
    """
    import subprocess
    import tempfile

    from PIL import Image

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(L("ffmpeg not found in PATH; it is needed to read video.",
                             "No se encuentra ffmpeg en el PATH; hace falta para leer video."))

    total = video_frame_count(path)
    if not total:
        raise RuntimeError(L("could not count the clip's frames",
                             "no se pudo contar los fotogramas del clip"))

    ultimo = int(window) if window and window > 0 else total
    ultimo = max(1, min(ultimo, total))

    count = max(1, min(int(count), ultimo))
    wanted = [int(round(i * (ultimo - 1) / float(max(1, count - 1)))) for i in range(count)] \
        if count > 1 else [0]
    expr = "+".join("eq(n\\,{})".format(k) for k in sorted(set(wanted)))

    escala = ""
    if max_side and max_side > 0:
        # -1 mantiene la proporcion; el 2 evita dimensiones impares.
        # -1 keeps the aspect ratio; 2 avoids odd dimensions.
        escala = ",scale='min({m},iw)':-2".format(m=int(max_side))

    with tempfile.TemporaryDirectory() as tmp:
        patron = os.path.join(tmp, "f%04d.png")
        subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-i", str(path),
             "-vf", "select='{}'{}".format(expr, escala),
             "-vsync", "0", "-frames:v", str(len(set(wanted))), patron],
            capture_output=True, text=True, timeout=300, check=True,
        )
        salida = sorted(os.listdir(tmp))
        if not salida:
            raise RuntimeError(L("ffmpeg extracted no frames",
                                 "ffmpeg no extrajo ningun fotograma"))
        return [Image.open(os.path.join(tmp, f)).convert("RGB").copy() for f in salida]


def video_duration(path):
    """Duracion del clip en segundos, o None. / Clip duration in seconds."""
    fps = video_fps(path)
    frames = video_frame_count(path)
    if fps and frames:
        return frames / float(fps)
    return None


def video_fps(path):
    """Fotogramas por segundo del clip, o None. / Clip frame rate, or None."""
    import shutil
    import subprocess

    probe = shutil.which("ffprobe") or shutil.which("ffprobe.exe")
    if not probe:
        return None
    try:
        out = subprocess.run(
            [probe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        num, _, den = out.partition("/")
        return float(num) / float(den or 1)
    except Exception:
        return None


def video_frame_count(path):
    """Fotogramas reales del clip, contados. / Real frame count, counted."""
    import shutil
    import subprocess

    probe = shutil.which("ffprobe") or shutil.which("ffprobe.exe")
    if not probe:
        return 0
    try:
        out = subprocess.run(
            [probe, "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=120,
        ).stdout.strip()
        return int(out)
    except Exception:
        return 0


def build_video_inputs(processor, frames, prompt, duration=None):
    """Entrada para un clip: la lista de fotogramas va como UN video, no como
    imagenes sueltas. Qwen3-VL trae un Qwen3VLVideoProcessor propio y su
    plantilla emite <|vision_start|><|video_pad|><|vision_end|>; pasarlos como
    imagenes produciria tokens distintos y el modelo los leeria sin relacion
    temporal entre ellos.

    Input for a clip: the frame list goes in as ONE video, not as separate
    images. Qwen3-VL ships its own Qwen3VLVideoProcessor and its template emits
    the video tokens; passing them as images would produce different tokens and
    the model would read them with no temporal relation.
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    # Los metadatos describen LO QUE SE ENTREGA, no el clip original. Como aqui
    # ya se han submuestreado los fotogramas, el "fps efectivo" es
    # len(frames)/duracion: declarar los 73 del clip hace que el processor
    # intente reindexar sobre una linea temporal que no tiene y reviente con un
    # IndexError.
    #
    # Sin metadatos, transformers asume 24 fps y avisa. Con 12 fotogramas
    # repartidos por 3 segundos, 24 fps es falso: creeria que el clip dura medio
    # segundo y describiria mal el ritmo, que es justo lo que se le pregunta.
    #
    # The metadata describes WHAT IS HANDED IN, not the source clip. The frames
    # are already subsampled here, so the effective fps is len(frames)/duration:
    # declaring the clip's 73 makes the processor re-index over a timeline it
    # does not have and blow up with an IndexError.
    kwargs = {}
    if duration and duration > 0:
        kwargs["video_metadata"] = [{
            "fps": len(frames) / float(duration),
            "total_num_frames": len(frames),
            "duration": float(duration),
        }]

    return processor(text=[text], videos=[frames], return_tensors="pt", **kwargs)


def build_inputs(processor, image, prompt):
    """Construye la entrada con el chat template del modelo.

    Se usa apply_chat_template y no una cadena a mano: Qwen3-VL espera unos
    tokens de imagen concretos en unas posiciones concretas, y el template del
    repo es la unica fuente fiable de cuales son.

    Builds the input with the model's chat template. apply_chat_template is used
    rather than a hand-rolled string: Qwen3-VL expects specific image tokens at
    specific positions, and the repo's template is the only reliable source.
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    return processor(text=[text], images=[image], return_tensors="pt")


def clean_caption(text):
    """Quita los tics de conversacion que el modelo suele anteponer."""
    text = " ".join(str(text).split())
    for prefix in ("The image shows ", "This image shows ", "The image depicts ",
                   "This image depicts ", "The photo shows ", "In this image, ",
                   "In the image, ", "Sure, ", "Certainly, ", "Here is a "):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):]
            text = text[:1].upper() + text[1:] if text else text
            break
    return text.strip().strip('"')


def caption_image(model, processor, image, cfg):
    inputs = build_inputs(processor, image, cfg["caption_prompt"])
    inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(cfg["max_new_tokens"]),
            do_sample=float(cfg["temperature"]) > 0,
            temperature=max(float(cfg["temperature"]), 1e-4),
        )

    # Recortar el prompt: generate() devuelve la secuencia entera.
    # Trim the prompt: generate() returns the whole sequence.
    input_len = inputs["input_ids"].shape[1]
    tokenizer = getattr(processor, "tokenizer", processor)
    text = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    return clean_caption(text)


def caption_clip(model, processor, frames, cfg, duration=None):
    """Igual que caption_image pero con la lista de fotogramas y el prompt de
    movimiento. / Same as caption_image but with the frame list and the motion
    prompt."""
    inputs = build_video_inputs(processor, frames, cfg["caption_video_prompt"],
                                duration=duration)
    inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(cfg["max_new_tokens"]),
            do_sample=float(cfg["temperature"]) > 0,
            temperature=max(float(cfg["temperature"]), 1e-4),
        )

    input_len = inputs["input_ids"].shape[1]
    tokenizer = getattr(processor, "tokenizer", processor)
    return clean_caption(tokenizer.decode(out[0][input_len:], skip_special_tokens=True))


def compose(trigger, caption):
    """Trigger primero, siempre. / Trigger first, always."""
    trigger = (trigger or "").strip().rstrip(",")
    if not trigger:
        return caption
    if caption.lower().startswith(trigger.lower()):
        return caption
    return "{}, {}".format(trigger, caption)


# =============================================================================
# MAIN
# =============================================================================

def main():
    cfg = load_config()

    # El modelo PRIMERO, antes de mirar el dataset. Quien pulsa el boton espera
    # que si falta se descargue; que el dataset este vacio es harina de otro
    # costal y no debe impedir dejar la herramienta lista.
    # The model FIRST, before looking at the dataset. Whoever presses the button
    # expects a missing model to be downloaded; an empty dataset is a separate
    # matter and must not stop the tool from being made ready.
    try:
        ensure_captioner(cfg)
    except Exception as exc:
        log(LF("[ERROR] {}", "[ERROR] {}", exc))
        return 1

    dataset = cfg["dataset_path"]
    if not os.path.isdir(dataset):
        log(LF("[ERROR] Dataset folder not found: {}",
               "[ERROR] No existe la carpeta del dataset: {}", dataset))
        return 1

    images = sorted(f for f in os.listdir(dataset)
                    if f.lower().endswith(MEDIA_EXTS))
    if not images:
        log(LF("[ERROR] No images or clips in {}",
               "[ERROR] No hay imagenes ni clips en {}", dataset))
        return 1

    overwrite = bool(cfg["overwrite"])
    pending = []
    for name in images:
        txt = os.path.join(dataset, os.path.splitext(name)[0] + ".txt")
        if not overwrite and os.path.isfile(txt):
            try:
                if open(txt, encoding="utf-8").read().strip():
                    continue
            except Exception:
                pass
        pending.append(name)

    log("=" * 78)
    log("[CAPTION] Dataset : {}".format(os.path.abspath(dataset)))
    _clips = sum(1 for f in images if f.lower().endswith(VIDEO_EXTS))
    log(LF("[CAPTION] Files   : {} total ({} clips), {} to caption",
           "[CAPTION] Ficheros: {} en total ({} clips), {} por describir",
           len(images), _clips, len(pending)))
    log("[CAPTION] Trigger : {}".format(
        cfg["trigger_word"] or L("(none)", "(ninguno)")))
    log("[CAPTION] Mode    : {}".format(
        L("overwrite everything", "rehacer todos") if overwrite
        else L("only the missing ones", "solo los que faltan")))
    log("[CAPTION] Tokens  : {}".format(cfg["max_new_tokens"]))
    log("[CAPTION] Resize  : {}".format(
        LF("{} px longest side, in memory only",
           "{} px de lado largo, solo en memoria", cfg["max_image_side"])
        if int(cfg["max_image_side"]) > 0
        else L("off (full resolution)", "desactivado (resolucion completa)")))
    log("=" * 78)

    if not pending:
        log(L("[CAPTION] Nothing to do: every image already has a caption.",
              "[CAPTION] Nada que hacer: todas las imagenes ya tienen caption."))
        return 0

    from PIL import Image

    model, processor = load_captioner(cfg)

    started = time.time()
    done = 0
    failed = []

    for i, name in enumerate(pending, 1):
        path = os.path.join(dataset, name)
        step_started = time.time()
        try:
            if name.lower().endswith(VIDEO_EXTS):
                ventana = caption_window(path, cfg)
                total_clip = video_frame_count(path) or ventana
                if ventana < total_clip:
                    # Merece un aviso: significa que parte del clip no se
                    # entrena, y es exactamente el final, donde suele estar el
                    # remate del efecto. / Worth a warning: part of the clip is
                    # not trained, and it is the end -- usually where the payoff
                    # of the effect lives.
                    log(LF("[!] {}: captioning the first {} of {} frames "
                           "({:.0f}% of the clip), which is what pre-cache will "
                           "train. The rest is NOT described.",
                           "[!] {}: se hace caption de los {} primeros "
                           "fotogramas de {} ({:.0f}% del clip), que es lo que "
                           "entrenara la pre-cache. El resto NO se describe.",
                           name, ventana, total_clip, 100.0 * ventana / total_clip))
                fps = video_fps(path)
                frames = extract_frames(path, cfg["caption_video_frames"],
                                        int(cfg["max_image_side"]), ventana)
                caption = caption_clip(model, processor, frames, cfg,
                                       duration=(ventana / float(fps)) if fps else None)
            else:
                with Image.open(path) as img:
                    image = shrink_for_captioner(img.convert("RGB"),
                                                 int(cfg["max_image_side"]))
                caption = caption_image(model, processor, image, cfg)
            if not caption:
                raise RuntimeError(L("empty caption", "caption vacio"))

            final = compose(cfg["trigger_word"], caption)
            txt = os.path.join(dataset, os.path.splitext(name)[0] + ".txt")
            with open(txt, "w", encoding="utf-8") as fh:
                fh.write(final)
            done += 1

            elapsed = time.time() - step_started
            eta = (time.time() - started) / i * (len(pending) - i)
            log("[{}/{}] {:<38} {:5.1f}s  ETA {:.0f}m{:02.0f}s".format(
                i, len(pending), name[:38], elapsed, eta // 60, eta % 60))
            log("        {}".format(final[:150] + ("..." if len(final) > 150 else "")))

        except Exception as exc:
            failed.append(name)
            log(LF("[{}/{}] {} FAILED: {}",
                   "[{}/{}] {} FALLO: {}", i, len(pending), name, exc))

    total = time.time() - started
    log("=" * 78)
    log(LF("[CAPTION] Done: {} captions in {:.0f}m{:02.0f}s ({:.1f}s per image).",
           "[CAPTION] Hecho: {} captions en {:.0f}m{:02.0f}s ({:.1f}s por imagen).",
           done, total // 60, total % 60, total / max(done, 1)))
    if failed:
        log(LF("[CAPTION] {} failed: {}",
               "[CAPTION] {} fallaron: {}", len(failed), ", ".join(failed[:10])))
    log(L("[CAPTION] Review them in the Dataset Manager before pre-caching.",
          "[CAPTION] Revisalos en el Dataset Manager antes de hacer la pre-cache."))
    log("=" * 78)
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
