# -*- coding: utf-8 -*-
"""Carga, descarga y ejecucion del separador. / Loading, download and inference."""

import os
import subprocess

import numpy as np

# Repositorio y REVISION FIJA. Sin fijarla, una actualizacion del repo cambiaria
# los pesos bajo los pies de un dataset ya preparado y nadie sabria por que el
# resultado dejo de parecerse.
# Repo and PINNED revision. Without the pin, an update upstream would change the
# weights under an already-prepared dataset and nobody would know why the result
# stopped matching.
REPO_ID = "Kijai/MelBandRoFormer_comfy"
REVISION = "6251b3a2bd544aaa31400138e55abda4722735cc"
FICHERO = "MelBandRoformer_fp16.safetensors"
CARPETA = "./MelBandRoFormer"

# 44,1 kHz es la tasa con la que se entreno. Separar a 32 k -- la del VAE de H3 --
# daria peor resultado, asi que se remuestrea DESPUES, no antes.
# 44.1 kHz is what it was trained at. Separating at H3's 32 kHz would be worse,
# so the resample happens AFTER, not before.
SR = 44100

# Los cinco valores del troceado, tal como los usa el nodo de referencia:
# ventana de 8 s, medio solapamiento y un fundido de 0,8 s en los bordes. Sin el
# fundido se oye el corte en cada union.
# The five chunking values, as the reference node uses them: an 8 s window, half
# overlap and a 0.8 s fade at the edges. Without the fade the seams are audible.
VENTANA = 352800          # 8 s a 44100
PASOS = 2
FADE = VENTANA // 10

CONFIG = {
    "dim": 384, "depth": 6, "stereo": True, "num_stems": 1,
    "time_transformer_depth": 1, "freq_transformer_depth": 1,
    "num_bands": 60, "dim_head": 64, "heads": 8,
    "attn_dropout": 0, "ff_dropout": 0, "flash_attn": True,
    "dim_freqs_in": 1025, "sample_rate": SR,
    "stft_n_fft": 2048, "stft_hop_length": 441, "stft_win_length": 2048,
    "stft_normalized": False, "mask_estimator_depth": 2,
    "multi_stft_resolution_loss_weight": 1.0,
    "multi_stft_resolutions_window_sizes": (4096, 2048, 1024, 512, 256),
    "multi_stft_hop_size": 147, "multi_stft_normalized": False,
}

# Solo dos, y ambos diminutos. librosa NO hace falta: mel_converter.py trae el
# banco de filtros mel reimplementado en numpy ("following is from librosa"),
# precisamente para no arrastrar numba y scipy por una funcion opcional.
# Only two, both tiny. librosa is NOT needed: mel_converter.py carries the mel
# filter bank reimplemented in numpy, precisely to avoid dragging numba and scipy
# in for an optional feature.
FALTAN = ("einops", "rotary_embedding_torch")


def dependencias_que_faltan():
    """Las que no estan instaladas. Se comprueban ANTES de descargar 600 MB.

    Which ones are missing. Checked BEFORE downloading 600 MB.
    """
    import importlib
    fuera = []
    for m in FALTAN:
        try:
            importlib.import_module(m)
        except Exception:
            fuera.append(m)
    return fuera


def ruta_pesos(carpeta=CARPETA):
    return os.path.join(carpeta, FICHERO)


def descargar(carpeta=CARPETA, log=print):
    """Baja los pesos si no estan. Devuelve la ruta, o None si fallo.

    Downloads the weights when missing. Returns the path, or None on failure.
    """
    destino = ruta_pesos(carpeta)
    if os.path.isfile(destino):
        return destino

    os.makedirs(carpeta, exist_ok=True)
    log("[MELBAND] Downloading {} ({}) from {} ... / Descargando ..."
        .format(FICHERO, "~600 MB", REPO_ID))
    try:
        from huggingface_hub import hf_hub_download
        traido = hf_hub_download(repo_id=REPO_ID, filename=FICHERO,
                                 revision=REVISION, local_dir=carpeta)
        log("[MELBAND] Ready / Listo: {}".format(traido))
        return traido
    except Exception as exc:
        log("[MELBAND][ERROR] Download failed / Fallo la descarga: {}".format(exc))
        return None


def cargar(carpeta=CARPETA, device="cuda", log=print):
    """Construye el modelo y le carga los pesos. / Builds the model and loads it."""
    import torch
    from safetensors.torch import load_file
    from .mel_band_roformer import MelBandRoformer

    pesos = descargar(carpeta, log)
    if not pesos:
        return None

    modelo = MelBandRoformer(**CONFIG).eval()
    modelo.load_state_dict(load_file(pesos), strict=True)
    modelo.requires_grad_(False)
    return modelo.to(device)


def _ventana_fundido(n, fade, device):
    import torch
    w = torch.ones(n)
    w[:fade] *= torch.linspace(0, 1, fade)
    w[-fade:] *= torch.linspace(1, 0, fade)
    return w.to(device)


def separar_voz(modelo, pcm, device="cuda"):
    """[2, N] a 44,1 kHz -> [2, N] con solo la voz.

    Se procesa en ventanas solapadas y se suman con pesos de fundido, dividiendo
    al final por la suma de los pesos: asi cada muestra queda normalizada aunque
    pertenezca a dos ventanas. El relleno reflejado de los extremos evita que el
    modelo vea un salto a silencio donde no lo hay.

    [2, N] at 44.1 kHz -> [2, N] of voice only. Processed in overlapping windows
    summed with fade weights and divided by the weight sum, so every sample is
    normalised even when it belongs to two windows. The reflected padding at the
    ends keeps the model from seeing a jump to silence that is not there.
    """
    import torch
    import torch.nn.functional as F

    x = torch.as_tensor(pcm, dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.shape[0] == 1:
        x = x.repeat(2, 1)                      # el modelo es estereo

    largo = x.shape[1]
    paso = VENTANA // PASOS
    borde = VENTANA - paso
    relleno = largo > 2 * borde and borde > 0
    if relleno:
        x = F.pad(x, (borde, borde), mode="reflect")

    x = x.to(device)
    voz = torch.zeros_like(x, dtype=torch.float32)
    peso = torch.zeros_like(x, dtype=torch.float32)
    total = x.shape[1]
    w0 = _ventana_fundido(VENTANA, FADE, device)

    with torch.inference_mode():
        for i in range(0, total, paso):
            trozo = x[:, i:i + VENTANA]
            n = trozo.shape[-1]
            if n < VENTANA:
                modo = "reflect" if n > VENTANA // 2 + 1 else "constant"
                trozo = F.pad(trozo, (0, VENTANA - n), mode=modo)

            salida = modelo(trozo.unsqueeze(0))[0]

            w = w0.clone()
            if i == 0:
                w[:FADE] = 1                    # el primer trozo no entra fundido
            elif i + VENTANA >= total:
                w[-FADE:] = 1                   # ni el ultimo sale fundido

            voz[..., i:i + n] += salida[..., :n] * w[..., :n]
            peso[..., i:i + n] += w[..., :n]

    voz = voz / peso.clamp_min(1e-8)
    if relleno:
        voz = voz[..., borde:-borde]
    return voz.cpu().numpy()


def leer_pcm(path, ffmpeg, sr=SR):
    """Fichero -> [2, N] float32 a `sr`. / File -> [2, N] float32 at `sr`."""
    raw = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-f", "f32le",
         "-ac", "2", "-ar", str(sr), "-"],
        capture_output=True, timeout=1800).stdout
    a = np.frombuffer(raw, dtype=np.float32)
    return np.ascontiguousarray(a[: (a.size // 2) * 2].reshape(-1, 2).T)


def escribir_pcm(pcm, path, ffmpeg, sr=SR):
    """[2, N] -> fichero, con el mismo formato que espera la pre-cache."""
    datos = np.ascontiguousarray(pcm.T.astype(np.float32)).tobytes()
    subprocess.run(
        [ffmpeg, "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "2",
         "-i", "-", "-ar", "32000", "-ac", "2", str(path)],
        input=datos, capture_output=True, timeout=1800, check=True)
