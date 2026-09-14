# -*- coding: utf-8 -*-
"""Extraccion de RefMods de MiniMax H3. / MiniMax H3 RefMod extraction.

QUE ES UN REFMOD Y POR QUE NO ES UN LORA
----------------------------------------
Un RefMod no modifica ni un peso. Es un latente de referencia ya codificado que
se anade a los `refs` del conditioning, de modo que el DiT lo atiende a traves
de todos los bloques igual que atenderia una imagen o un video de referencia. Es
el camino nativo de referencia de H3 con material pre-cocinado.

La consecuencia practica es la que importa en este proyecto: entrenar audio
mueve pesos compartidos y la rama de video se va a la deriva -- el compromiso
que nos costo semanas medir, con su umbral en lr x pasos ~= 0,26. Un RefMod
mueve CERO pesos, asi que ese compromiso no puede ocurrir. A cambio no puede
ensenarle nada nuevo al modelo: solo puede apuntarlo a algo que ya sabe
representar.

Y no sale gratis. Los tokens del RefMod entran en la secuencia que el DiT
atiende en CADA paso de CADA generacion, para siempre. Donde un LoRA no cuesta
nada al generar, esto se paga en todos los renders.

COMPATIBILIDAD CON EL NODO DE COMFYUI
-------------------------------------
El fichero que escribe esto ES el que carga ComfyUI-MiniMaxH3Mod: un
.safetensors con el tensor "latent" y los metadatos en la cabecera bajo la clave
"refmod_meta". No hay conversion ni paso de export; por eso la salida apunta
directamente a models/refmods.

Dos cosas se verificaron leyendo el codigo del nodo antes de escribir esto, y
ambas podrian haber producido un fichero silenciosamente inservible:

  * NORMALIZACION. El VAE de audio de ComfyUI aplica (z - mean) / std DENTRO de
    encode(), y codifica cada canal como mono por separado. Es exactamente lo
    que hace encode_audio_latent() de la pre-cache. Compatible.
  * FORMA. El nodo exige [1, 32, 2, T] con el eje de canal separado; la
    pre-cache devuelve [1, 32, 2T] con los canales uno tras otro (todo el
    izquierdo y luego todo el derecho). Es un reshape, y el orden coincide.

GEOMETRIA TEMPORAL
------------------
El paquete de ComfyUI afirma que el VAE de video de H3 es causal y solo admite
4k+1 fotogramas. La documentacion de este proyecto dice 17n+5 -> 5n+2 y anade
explicitamente que NO es el patron 4k+1 de Wan o Hunyuan. Las dos no pueden ser
ciertas. Aqui se sigue la de este proyecto, que lleva meses validada por
entrenamientos reales, y se alimenta al encoder la rejilla que espera. El
latente resultante es un latente H3 legitimo sea cual sea T: el DiT atiende
[1, 24, T, H, W] sin importarle como se llego a T.

A RefMod is not a LoRA: it changes no weights, it appends a pre-encoded
reference to the conditioning's refs so the DiT attends to it through every
block. That is why it cannot cause the audio/video trade-off this project spent
weeks measuring -- and also why it cannot teach anything new. Its tokens are
paid on every step of every generation, forever.
"""

import json
import os
import tempfile

import torch
from safetensors.torch import save_file

# Clave de los metadatos en la cabecera del safetensors, tal como la lee
# core.py del nodo. Si esto cambia, los mods dejan de cargar.
# Metadata key in the safetensors header, as the node's core.py reads it.
META_KEY = "refmod_meta"
FORMAT_VERSION = 4

# Un "bundle" es un CONTENEDOR, no una fusion. Guarda varias referencias en un
# fichero, cada una con su tensor propio (ref_0, ref_1...) y sus metadatos, y al
# cargarlo se despliegan como bloques independientes: exactamente lo mismo que
# dos ficheros sueltos, con un fichero menos que manejar.
#
# Conviene tenerlo claro porque el nombre invita a pensar lo contrario: NO ata la
# voz a la cara. El propio formato lo dice en su primera linea -- "it does not
# concatenate audio with visual latents or change H3 conditioning semantics" -- y
# el autor lo repite en las notas de la v0.2.6: "bundling does not add
# voice-to-character binding". Quien busque esa union necesita el bloque
# video_audio del modelo, que ningun extractor emite.
#
# A bundle is a CONTAINER, not a fusion: several references in one file, each
# with its own tensor and metadata, expanded into independent blocks on load.
# It does NOT bind voice to identity, whatever the name suggests.
BUNDLE_VERSION = 5
BUNDLE_MAX_MIEMBROS = 256

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")

# Un token cubre 32x32 pixeles reales (VAE /16 y patch 2x2). En audio, un token
# son dos latentes y hay 40 latentes por segundo y canal -> 80 tokens/s.
# One token covers 32x32 real pixels; in audio a token is two latents at 40 per
# second and channel -> 80 tokens/s.
TOKENS_POR_SEGUNDO_AUDIO = 80

_PRECACHE = None


def precache():
    """El modulo de pre-cache, cargado una vez. / The pre-cache module, once.

    Se importa por ruta porque el nombre empieza por un digito y no es un
    identificador valido de Python. Su codigo de nivel superior solo lee la
    configuracion y define funciones -- el trabajo real esta detras de
    `if __name__ == "__main__"` -- asi que importarlo es barato y seguro.

    Imported by path because the name starts with a digit. Its top level only
    reads config and defines functions; the real work sits behind the __main__
    guard, so importing is cheap and safe.
    """
    global _PRECACHE
    if _PRECACHE is None:
        import importlib.util
        ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "1_pre_cache_MiniMaxH3.py")
        spec = importlib.util.spec_from_file_location("h3_precache", ruta)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _PRECACHE = mod
    return _PRECACHE


REPO_NF4 = "AcademiaSD/MiniMax-H3-NF4"


def vaes_presentes(nf4_model_id):
    """(hay_video, hay_audio). Delega en la pre-cache, que define donde vive cada uno."""
    return precache().vaes_presentes(nf4_model_id)


def asegurar_vaes(nf4_model_id, video=True, audio=True, repo_id=REPO_NF4, log=print):
    """Descarga los VAEs que falten, y SOLO esos.

    La implementacion vive en 1_pre_cache_MiniMaxH3.py, que es quien sabe donde
    busca cada VAE, y asi la pre-cache tambien se repara sola si alguien borra
    uno a mano. Aqui importa por el motivo contrario: un RefMod no carga el DiT,
    asi que quien solo extraiga referencias no tiene por que bajarse los 41 GB
    del repo -- y como el VAE de video pesa 5,2 GB y el de audio 0,6, cada pasada
    pide unicamente el suyo.

    The implementation lives in the pre-cache module, which owns where each VAE
    is looked up, so the pre-cache repairs itself too when one is deleted by
    hand. It matters here for the opposite reason: a RefMod never loads the DiT.
    """
    return precache().asegurar_vaes(nf4_model_id, video=video, audio=audio,
                                    repo_id=repo_id, log=log)


# ══════════════════════════════════════════════════════════════════════════
# Guardado
# ══════════════════════════════════════════════════════════════════════════

def token_count(latent, kind):
    """Tokens que el mod inyecta, con la misma cuenta que core.py del nodo."""
    if kind == "audio":
        return 2 * int(latent.shape[-1])
    return int(latent.shape[2]) * (int(latent.shape[3]) // 2) * (int(latent.shape[4]) // 2)


def _metadatos(latent, kind, name, mode="encode", source="", source_shape="",
               pool="", description="", concept_type="generic", tags=None,
               sample_rate=32000):
    """Los metadatos de UNA referencia, validando su forma.

    Es la misma estructura tanto si acaba sola en un fichero como si va dentro de
    un bundle: alli cada miembro conserva sus propios metadatos de version 4.
    The same structure whether it ends up alone in a file or inside a bundle,
    where each member keeps its own version-4 metadata.
    """
    if kind == "audio":
        if latent.ndim != 4 or tuple(latent.shape[:3]) != (1, 32, 2):
            raise ValueError("El latente de audio debe ser [1,32,2,T], no {}"
                             .format(tuple(latent.shape)))
        latent_t = int(latent.shape[-1])
        latent_h = latent_w = 0
        if latent_t <= 0:
            raise ValueError("El latente de audio necesita T > 0.")
    else:
        if latent.ndim != 5:
            raise ValueError("El latente visual debe ser [1,24,T,H,W], no {}"
                             .format(tuple(latent.shape)))
        latent_t = int(latent.shape[2])
        latent_h, latent_w = int(latent.shape[3]), int(latent.shape[4])
        # El formato exige H y W PARES y positivas: los tokens son (H/2)x(W/2),
        # asi que una dimension impar daria un recuento que no cuadra y el
        # cargador rechaza el fichero.
        # The format requires EVEN positive H and W: tokens are (H/2)x(W/2), so an
        # odd dimension yields a count that does not add up and the loader
        # rejects the file.
        if latent_t <= 0 or latent_h <= 0 or latent_w <= 0:
            raise ValueError("El latente visual necesita T, H y W positivos.")
        if latent_h % 2 or latent_w % 2:
            raise ValueError("El formato exige H y W pares; son {}x{}"
                             .format(latent_h, latent_w))
        if kind == "image" and latent_t != 1:
            raise ValueError("kind 'image' exige un unico fotograma latente.")

    return {
        "name": name,
        "kind": kind,
        "latent_h": latent_h,
        "latent_w": latent_w,
        "latent_t": latent_t,
        "mode": mode,
        "source": source,
        "source_shape": source_shape,
        "pool": pool,
        "optimize_steps": 0,
        "tags": list(tags or []),
        "description": description,
        "concept_type": concept_type,
        "_format_version": FORMAT_VERSION,
        "sample_rate": int(sample_rate),
    }


def _escribir(tensores, meta, ruta_sin_ext):
    """Escribe el safetensors de forma atomica. / Writes the safetensors atomically.

    A un temporal y luego os.replace: un fichero a medio escribir en
    models/refmods aparece igualmente en el desplegable de ComfyUI y revienta al
    cargarlo. / A half-written file still shows up in ComfyUI's dropdown.
    """
    destino = ruta_sin_ext + ".safetensors"
    carpeta = os.path.dirname(destino) or "."
    os.makedirs(carpeta, exist_ok=True)
    fd, temporal = tempfile.mkstemp(prefix=".refmod-", suffix=".tmp", dir=carpeta)
    os.close(fd)
    try:
        save_file({k: v.contiguous() for k, v in tensores.items()}, temporal,
                  metadata={META_KEY: json.dumps(meta)})
        os.replace(temporal, destino)
    finally:
        if os.path.exists(temporal):
            os.unlink(temporal)
    return destino


def guardar_bundle(miembros, name, ruta_sin_ext):
    """Varias referencias en UN fichero, en el formato de version 5.

    `miembros` es una lista de (latent, kind, kwargs) en el orden en que se
    quieren guardar. Los tensores van como ref_0, ref_1... emparejados por
    POSICION con la lista `members` de los metadatos; ese emparejamiento es todo
    el formato, asi que el orden no es cosmetico.

    `miembros` is a list of (latent, kind, kwargs). Tensors are stored as ref_0,
    ref_1... paired BY POSITION with the metadata's `members` list; that pairing
    is the whole format, so the order is not cosmetic.
    """
    if not miembros:
        raise ValueError("Un bundle necesita al menos una referencia.")
    if len(miembros) > BUNDLE_MAX_MIEMBROS:
        raise ValueError("Un bundle admite hasta {} miembros; hay {}."
                         .format(BUNDLE_MAX_MIEMBROS, len(miembros)))

    tensores, members = {}, []
    for i, (latent, kind, kw) in enumerate(miembros):
        members.append(_metadatos(latent, kind, **kw))
        tensores["ref_{}".format(i)] = latent

    meta = {
        "_format_version": BUNDLE_VERSION,
        "kind": "bundle",
        "name": name,
        "members": members,
    }
    return _escribir(tensores, meta, ruta_sin_ext)


def guardar(latent, kind, name, ruta_sin_ext, mode="encode", source="",
            source_shape="", pool="", description="", concept_type="generic",
            tags=None, sample_rate=32000):
    """Escribe {ruta}.safetensors con UNA referencia, en formato 4.

    Es el formato que leen todas las versiones del nodo. El bundle de version 5
    solo lo entienden las 0.2.6 en adelante, asi que este sigue siendo el
    predeterminado. / Version 4 is read by every version of the node; the
    version-5 bundle needs 0.2.6 or newer, so this stays the default.
    """
    meta = _metadatos(latent, kind, name, mode=mode, source=source,
                      source_shape=source_shape, pool=pool, description=description,
                      concept_type=concept_type, tags=tags, sample_rate=sample_rate)
    return _escribir({"latent": latent}, meta, ruta_sin_ext)


# ══════════════════════════════════════════════════════════════════════════
# Audio
# ══════════════════════════════════════════════════════════════════════════

def extraer_audio(fuentes, audio_vae, max_tokens=400, log=print):
    """Latente [1, 32, 2, T] a partir de las fuentes con pista de audio.

    Las referencias se concatenan en el tiempo y se recorta al presupuesto. Se
    recorta por el FINAL y no se remuestrea: un latente de audio remuestreado no
    suena mas corto, suena mal.

    [1, 32, 2, T] from whichever sources carry audio. References are
    concatenated in time and cut to budget from the END rather than resampled:
    a resampled audio latent does not sound shorter, it sounds wrong.
    """
    P = precache()
    trozos = []
    usados = []
    tope_latentes = max(1, max_tokens // 2) if max_tokens else None

    for ruta in fuentes:
        ext = os.path.splitext(ruta)[1].lower()
        if ext not in AUDIO_EXTS + VIDEO_EXTS:
            continue
        try:
            dur = P.audio_duration(ruta)
            if not dur or dur <= 0:
                continue
            # read_audio_pcm pide el audio en fotogramas de video a 24 fps: se
            # le pasa la duracion real del fichero, no una geometria de
            # entrenamiento, porque aqui no hay clip que sincronizar.
            # read_audio_pcm asks for audio in 24 fps video frames: the file's
            # real duration is passed, not a training geometry -- there is no
            # clip to stay in sync with here.
            frames = max(1, int(round(dur * 24.0)))
            pcm = P.read_audio_pcm(ruta, frames, 24.0)
            if pcm is None or pcm.size == 0:
                continue
            z = P.encode_audio_latent(audio_vae, pcm)     # [1, 32, 2T]
            if z.ndim != 3 or z.shape[1] != 32 or z.shape[2] % 2:
                log("[REFMOD] {}: latente inesperado {}, se salta"
                    .format(os.path.basename(ruta), tuple(z.shape)))
                continue
            t = z.shape[2] // 2
            # [1,32,2T] canal-mayor -> [1,32,2,T]. El orden coincide: las
            # primeras T posiciones son el canal izquierdo.
            # Channel-major [1,32,2T] -> [1,32,2,T]; the first T positions are
            # the left channel, which is what the node expects.
            trozos.append(z.reshape(1, 32, 2, t).float())
            usados.append("{} ({:.2f}s)".format(os.path.basename(ruta), t / 40.0))
            if tope_latentes and sum(x.shape[-1] for x in trozos) >= tope_latentes:
                break
        except Exception as exc:
            log("[REFMOD] {}: {}".format(os.path.basename(ruta), exc))

    if not trozos:
        return None, []

    latente = torch.cat(trozos, dim=-1)
    if tope_latentes and latente.shape[-1] > tope_latentes:
        log("[REFMOD] audio: {} latentes -> {} por el presupuesto de {} tokens"
            .format(latente.shape[-1], tope_latentes, max_tokens))
        latente = latente[..., :tope_latentes].clone()
    return latente.to(torch.float16), usados


# ══════════════════════════════════════════════════════════════════════════
# Visual
# ══════════════════════════════════════════════════════════════════════════

def preparar_video_vae(vae):
    """Lleva el VAE de video a CUDA en bf16. / Moves the video VAE to CUDA in bf16.

    load_h3_video_vae() lo deja en CPU y en float32: en la pre-cache es el
    llamante quien lo coloca. encode_clip_latent() manda la entrada a "cuda"
    incondicionalmente, asi que sin esto salta un "Input type
    (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the
    same" que no dice nada sobre lo que hay que arreglar.

    load_h3_video_vae() leaves it on CPU in float32 -- in the pre-cache the
    caller places it. encode_clip_latent() sends its input to "cuda"
    unconditionally, so without this you get a device-mismatch error that says
    nothing about what to fix.
    """
    if torch.cuda.is_available():
        vae = vae.to("cuda", dtype=torch.bfloat16)
    return vae.eval()


def _medidas(ruta, P):
    """(fotogramas, ancho, alto). Para una imagen, fotogramas = 1."""
    ext = os.path.splitext(ruta)[1].lower()
    if ext not in IMAGE_EXTS:
        info = P.probe_video(ruta)
        if info:
            return int(info[0]), int(info[1]), int(info[2])
    try:
        from PIL import Image
        with Image.open(ruta) as im:
            return 1, int(im.size[0]), int(im.size[1])
    except Exception:
        return 1, 0, 0


def _lienzo(ruta, short_edge, P):
    """(ancho, alto) multiplos de 32, lado corto <= short_edge, SOLO reduce.

    Nunca amplia: agrandar una referencia no le anade detalle, solo tokens, y
    los tokens se pagan en cada generacion.
    Never upscales: enlarging a reference adds no detail, only tokens, and
    tokens are paid on every generation.
    """
    _, w, h = _medidas(ruta, P)
    if not w or not h:
        w = h = short_edge
    escala = min(1.0, float(short_edge) / float(min(w, h)))
    return (max(32, int(round(w * escala / 32)) * 32),
            max(32, int(round(h * escala / 32)) * 32))


def extraer_visual(fuentes, video_vae, resolution=1024, max_tokens=1024, log=print):
    """Latente [1, 24, T, H/16, W/16] apilado en el tiempo.

    Todas las referencias comparten un unico lienzo espacial -- el de la
    primera -- porque el latente apilado tiene una sola H y una sola W.

    All references share one spatial canvas (the first one's) because the
    stacked latent has a single H and W.
    """
    P = precache()
    visuales = [r for r in fuentes
                if os.path.splitext(r)[1].lower() in IMAGE_EXTS + VIDEO_EXTS]
    if not visuales:
        return None, []

    ancho, alto = _lienzo(visuales[0], resolution, P)
    por_frame = (alto // 16 // 2) * (ancho // 16 // 2)
    if por_frame <= 0:
        raise ValueError("Lienzo invalido {}x{}".format(ancho, alto))
    tope_t = max(1, max_tokens // por_frame) if max_tokens else None
    log("[REFMOD] lienzo {}x{} px = {} tokens por fotograma latente{}"
        .format(ancho, alto, por_frame,
                "; caben {}".format(tope_t) if tope_t else ""))

    trozos, usados = [], []
    for ruta in visuales:
        try:
            nombre = os.path.basename(ruta)
            puestos = sum(x.shape[2] for x in trozos)
            if tope_t and puestos >= tope_t:
                break
            ext = os.path.splitext(ruta)[1].lower()

            if ext in IMAGE_EXTS:
                from PIL import Image
                with Image.open(ruta) as im:
                    img = im.convert("RGB").resize((ancho, alto))
                z = P.encode_video_latent(video_vae, img).float()      # [1,24,1,h,w]
            else:
                disponible = _medidas(ruta, P)[0]
                # Cuantos fotogramas de PIXEL corresponden a los latentes que
                # aun caben. h3_pixel_frames es la inversa exacta de la rejilla,
                # asi que no hay que adivinar: 5n+2 latentes <- 17n+5 pixeles.
                # How many PIXEL frames the remaining latent budget allows.
                # h3_pixel_frames is the exact inverse of the grid, so there is
                # nothing to guess: 5n+2 latents <- 17n+5 pixels.
                objetivo = 0
                if tope_t:
                    objetivo = P.h3_pixel_frames(max(2, tope_t - puestos))
                pedidos = P.h3_valid_frames(disponible, objetivo)
                if not pedidos:
                    log("[REFMOD] {}: solo {} fotogramas, por debajo del minimo "
                        "de 5 de la rejilla 17n+5; se salta".format(nombre, disponible))
                    continue
                frames = P.read_video_frames(ruta, pedidos, ancho, alto)
                z = P.encode_clip_latent(video_vae, frames).float()     # [1,24,T,h,w]

            trozos.append(z)
            usados.append("{} -> {} latentes de {}x{}".format(
                nombre, z.shape[2], z.shape[3], z.shape[4]))
            log("[REFMOD] {} -> {}".format(nombre, tuple(z.shape)))
        except Exception as exc:
            log("[REFMOD] {}: {}".format(os.path.basename(ruta), exc))

    if not trozos:
        return None, []

    latente = torch.cat(trozos, dim=2)
    if tope_t and latente.shape[2] > tope_t:
        # Se remuestrea uniformemente en vez de cortar por el final: en visual
        # los ultimos fotogramas informan tanto como los primeros, al reves que
        # en audio, donde cortar por el final solo acorta la muestra.
        # Uniformly resampled rather than cut from the end: in vision the last
        # frames carry as much as the first, unlike audio.
        idx = torch.linspace(0, latente.shape[2] - 1, tope_t).round().long()
        log("[REFMOD] visual: {} fotogramas latentes -> {} por el presupuesto de {} tokens"
            .format(latente.shape[2], tope_t, max_tokens))
        latente = latente[:, :, idx].clone()
    return latente.to(torch.float16), usados
