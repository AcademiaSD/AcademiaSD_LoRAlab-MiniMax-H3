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

    # False = no toca las imagenes que ya tienen .txt con contenido.
    # False = leaves alone any image that already has a non-empty .txt.
    "overwrite": True,
}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

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
    for key in ("dataset_path", "trigger_word"):
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

    directory = ensure_captioner(cfg)
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

    dataset = cfg["dataset_path"]
    if not os.path.isdir(dataset):
        log(LF("[ERROR] Dataset folder not found: {}",
               "[ERROR] No existe la carpeta del dataset: {}", dataset))
        return 1

    images = sorted(f for f in os.listdir(dataset)
                    if f.lower().endswith(IMAGE_EXTS))
    if not images:
        log(LF("[ERROR] No images in {}", "[ERROR] No hay imagenes en {}", dataset))
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
    log(LF("[CAPTION] Images  : {} total, {} to caption",
           "[CAPTION] Imagenes: {} en total, {} por describir", len(images), len(pending)))
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
