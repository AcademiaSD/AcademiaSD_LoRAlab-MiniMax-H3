# -*- coding: utf-8 -*-
"""
1_pre_cache_MiniMaxH3.py  (v3 - diagnostics build)

MiniMax-H3 pre-cache for LoRA training (image dataset).
Pre-cache de MiniMax-H3 para entrenamiento LoRA (dataset de imagenes).

WHAT CHANGED vs v2 / QUE HA CAMBIADO respecto a v2:
  1. RoPE inv_freq diagnostic + rebuild. Computed (non-checkpoint) buffers born on the
     meta device were previously zero-filled, which silently killed positional encoding.
     Diagnostico y reconstruccion de inv_freq (RoPE). Los buffers calculados que nacian
     en meta se rellenaban con ceros y eso mataba en silencio la codificacion posicional.
  2. materialize_meta_tensors() now FAILS instead of zero-filling (language path).
     Ahora ABORTA en vez de rellenar con ceros (ruta del modelo de lenguaje).
  3. Hard verification of the H3 video VAE state_dict load.
     Verificacion dura de la carga del state_dict del VAE de video H3.
  4. Self-checks: latent statistics, prompt-permutation test, caption coverage.
     Auto-tests: estadisticas del latente, test de permutacion de prompt, cobertura de captions.
  5. info.json now records num_frames = 1 (the real encoded frame count).
     info.json ahora guarda num_frames = 1 (el numero real de frames codificados).
  6. Audio latent channels default 32 (was 128).
     Canales del latente de audio por defecto 32 (antes 128).
  7. All on-screen messages are bilingual EN / ES on a single line.
     Todos los mensajes en pantalla son bilingues EN / ES en una sola linea.

OUTPUT FOR REVIEW / SALIDA PARA REVISAR:
  <CACHE_DIR>/_diagnostics.json  <- send this file back / devuelve este fichero

LOGS:
  LOGS_DEV = 1 -> verbose / detallado
  LOGS_DEV = 0 -> quiet / silencioso
"""

import os
import gc
import json
import math
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F_vision
from PIL import Image
from safetensors import safe_open

# ============================================================================
# BILINGUAL LOGGING / LOGS BILINGUES
# ============================================================================
LOGS_DEV = 1


class _Bi(str):
    """Bilingual string that formats each half separately.
    Cadena bilingue que formatea cada mitad por separado.

    Behaves like the joined "EN / ES" string, but .format(*args) applies the SAME args to
    the English half and to the Spanish half, instead of trying to consume them twice.
    Se comporta como la cadena unida "EN / ES", pero .format(*args) aplica los MISMOS
    argumentos a la mitad inglesa y a la espanola, en vez de intentar consumirlos dos veces.
    """

    def __new__(cls, en, es):
        obj = super().__new__(cls, u"{} / {}".format(en, es))
        obj._en = en
        obj._es = es
        return obj

    def format(self, *args, **kwargs):
        return u"{} / {}".format(self._en.format(*args, **kwargs),
                                 self._es.format(*args, **kwargs))


def L(en, es):
    """Bilingual single-line message / Mensaje bilingue en una sola linea."""
    return _Bi(en, es)


def log_dev(msg, level=1):
    if LOGS_DEV >= level:
        print(msg, flush=True)


def log_error(msg):
    print(msg, flush=True)


# Global diagnostics bag / Bolsa global de diagnosticos
DIAG = {
    "schema": "minimax_h3_precache_diagnostics/1",
    "generated_at": None,
    "env": {},
    "config": {},
    "text_encoder": {},
    "rope": {},
    "vae": {},
    "dataset": {},
    "selftests": {},
    "warnings": [],
    "errors": [],
}


def diag_warn(msg):
    DIAG["warnings"].append(msg)
    log_error(u"[WARN] " + msg)


def diag_error(msg):
    DIAG["errors"].append(msg)
    log_error(u"[ERROR] " + msg)


# ============================================================================
# PHYSICAL RAM DETECTION / DETECCION DE RAM FISICA
# ============================================================================
def _detect_system_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass
    if sys.platform.startswith("win"):
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024 ** 3)
        except Exception:
            pass
    return None


SYSTEM_RAM_GB = _detect_system_ram_gb()

# ============================================================================
# MINIMAX H3 VIDEO VAE - ENCODE ONLY (REFERENCE IMPLEMENTATION)
# ============================================================================
# H3 image training uses the video VAE encoder with T=1: 24 channels, 16x spatial
# downscale, ImageNet-normalized RGB over a [0,1] base, then H3 latent mean/std.
# El entrenamiento de imagen de H3 usa el encoder del VAE de video con T=1: 24 canales,
# reduccion espacial 16x, RGB normalizado ImageNet sobre base [0,1] y luego mean/std H3.

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608886, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.4498890042304993, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235595, 3.0496184825897216, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811524,
]


class H3CausalConv3d(nn.Conv3d):
    """Reflect spatial padding, causal front-only temporal padding.
    Padding espacial reflect, padding temporal causal solo por delante."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.causal_padding = (padding,) * 3 if isinstance(padding, int) else tuple(padding)
        k = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size,) * 3
        # The front-pad of cp[0]*2 is only equivalent to a causal pad for kernel 3.
        # El pad frontal de cp[0]*2 solo equivale a un pad causal con kernel 3.
        if self.causal_padding[0] != 0 and int(k[0]) != 3:
            raise ValueError(
                L("H3CausalConv3d temporal padding assumes kernel_size[0]==3",
                  "El padding temporal de H3CausalConv3d asume kernel_size[0]==3"))

    def forward(self, x):
        cp = self.causal_padding
        if sum(cp) == 0:
            return super().forward(x)
        x = torch.nn.functional.pad(x, (cp[2], cp[2], cp[1], cp[1], 0, 0), mode="reflect")
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, cp[0] * 2, 0), mode="constant")
        return super().forward(x)


class H3TemporalIsolatedGroupNorm(nn.GroupNorm):
    """GroupNorm with per-frame statistics / GroupNorm con estadisticas por frame."""

    def forward(self, x):
        if x.dim() == 5:
            b, c, t, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, 1, h, w)
            x = super().forward(x)
            return x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        return super().forward(x)


def h3_group_norm_3d(num_channels):
    return H3TemporalIsolatedGroupNorm(num_groups=32, num_channels=num_channels, eps=1e-6, affine=True)


class H3Downsample3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_stride=1, space_stride=2):
        super().__init__()
        self.space_stride = space_stride
        self.conv = H3CausalConv3d(
            in_channels, out_channels, kernel_size=3, padding=(1, 0, 0),
            stride=(time_stride, space_stride, space_stride)
        )

    def forward(self, x):
        if self.space_stride == 2:
            x = torch.nn.functional.pad(x, (0, 1, 0, 1, 0, 0), mode="reflect")
        return self.conv(x)


class H3ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = h3_group_norm_3d(in_channels)
        self.norm2 = h3_group_norm_3d(out_channels)
        self.conv1 = H3CausalConv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = H3CausalConv3d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = H3CausalConv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        h = self.conv2(torch.nn.functional.silu(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return h + x


class H3EncoderFCN3D(nn.Module):
    def __init__(self, ch=128, ch_mult=(1, 2, 2, 4, 4, 8), space_down=(2, 2, 2, 2, 1, 1),
                 time_down=(1, 2, 2, 1, 1, 1), num_res_blocks=2, in_channels=3, z_channels=24):
        super().__init__()
        num_levels = len(ch_mult)
        block_mid = [ch * ch_mult[i] for i in range(num_levels)]
        block_in = [block_mid[0]] + block_mid[:-1]
        self.conv_in = H3CausalConv3d(in_channels, block_in[0], kernel_size=3, padding=1)
        self.down = nn.ModuleList()
        for i_level in range(num_levels):
            down = nn.Module()
            down.block = nn.ModuleList()
            for i in range(num_res_blocks):
                down.block.append(H3ResnetBlock3D(
                    block_in[i_level] if i == 0 else block_mid[i_level], block_mid[i_level]
                ))
            if space_down[i_level] * time_down[i_level] > 1:
                down.downsample = H3Downsample3D(
                    block_mid[i_level], block_mid[i_level],
                    time_stride=time_down[i_level], space_stride=space_down[i_level]
                )
            self.down.append(down)
        self.norm_out = h3_group_norm_3d(block_mid[-1])
        self.conv_out = H3CausalConv3d(block_mid[-1], 2 * z_channels, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for down in self.down:
            for block in down.block:
                h = block(h)
            if hasattr(down, "downsample"):
                h = down.downsample(h)
        return self.conv_out(torch.nn.functional.silu(self.norm_out(h)))


# Geometria temporal del VAE de H3, tomada de AutoencoderKLMiniMaxH3:
#
#   "The temporal geometry is fixed by clip_length (17 pixel frames per encoder
#    chunk) and token_drop (3 trailing latent frames dropped per encode):
#    17 * n + 5 pixel frames map to 5 * n + 2 latent frames."
#
# El numero de fotogramas de un clip NO es libre: tiene que ser 17n+5. Ojo, no
# es el patron 4k+1 de Wan o Hunyuan, donde el primer fotograma va aparte.
#
# H3's temporal geometry, from the reference VAE. A clip's frame count is not
# free: it must be 17n+5. Note this is NOT the 4k+1 pattern of Wan or Hunyuan,
# where the first frame is handled separately.
H3_CLIP_LENGTH = 17
H3_TOKEN_DROP = 3
H3_BASE_FRAMES = 5


def h3_valid_frames(count, target=0):
    """Fotogramas a conservar de un clip de `count`.

    Con `target` (el campo Frames de la pre-cache) se pide un numero concreto y
    se usa si el clip da para tanto; si no, el mayor 17n+5 que quepa. El objetivo
    tambien se baja a la rejilla, para que un valor raro en el JSON no cuele una
    geometria que el VAE no puede producir.

    Frames to keep from a clip of `count`. With `target` a specific count is
    requested and used if the clip is long enough, otherwise the largest 17n+5
    that fits. The target is snapped to the grid too, so an odd value in the JSON
    cannot sneak in a geometry the VAE cannot produce.
    """
    if count < H3_BASE_FRAMES:
        return None
    mayor = H3_CLIP_LENGTH * ((count - H3_BASE_FRAMES) // H3_CLIP_LENGTH) + H3_BASE_FRAMES
    if target and target >= H3_BASE_FRAMES:
        objetivo = H3_CLIP_LENGTH * ((target - H3_BASE_FRAMES) // H3_CLIP_LENGTH) + H3_BASE_FRAMES
        return min(objetivo, mayor)
    return mayor


def h3_latent_frames(count):
    """Latentes que produce un clip de `count` fotogramas 17n+5."""
    return 5 * ((count - H3_BASE_FRAMES) // H3_CLIP_LENGTH) + 2


def h3_pixel_frames(latents):
    """Inversa: fotogramas de pixel que produjeron `latents` latentes.
    Inverse: pixel frames that produced `latents` latent frames."""
    return H3_CLIP_LENGTH * ((latents - 2) // 5) + H3_BASE_FRAMES


class MiniMaxH3VideoVAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = H3EncoderFCN3D()
        self.quant_conv = nn.Conv3d(48, 48, 1)
        self.register_buffer("latents_mean", torch.tensor(LATENTS_MEAN))
        self.register_buffer("latents_std", torch.tensor(LATENTS_STD))
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)

    @torch.no_grad()
    def encode(self, x):
        """x en [-1,1] [B,3,T,H,W] -> latente normalizado [B,24,T',H/16,W/16].

        Una imagen (T=1) pasa por el camino espacial y da un latente. Un clip se
        codifica POR TROZOS de `clip_length` fotogramas, que es lo que hace el
        VAE de referencia y NO una optimizacion:

          - las convoluciones temporales son causales y se reinician en cada
            trozo, asi que el contexto que ve el encoder no es el mismo si se le
            pasa el clip entero de una vez;
          - los recuentos no coinciden. Medido con este mismo encoder: 73
            fotogramas dan 22 latentes troceando y 19 de una tirada. Solo el
            primero cuadra con la geometria del modelo (17n+5 -> 5n+2).

        Codificarlo del tiron "funciona" —salen latentes y no falla nada— y
        produce un condicionamiento que el modelo nunca vio en entrenamiento.

        A still image (T=1) goes through the spatial path. A clip is encoded in
        `clip_length` CHUNKS, which is what the reference VAE does and is NOT an
        optimization: the temporal convolutions are causal and restart on every
        chunk, and the counts differ (73 frames give 22 latents chunked, 19 in
        one pass; only the former matches the model's 17n+5 -> 5n+2 geometry).
        Encoding in one pass "works" and yields conditioning the model was never
        trained on.
        """
        if x.ndim == 4:
            x = x.unsqueeze(2)
        if x.ndim != 5:
            raise ValueError(
                L("MiniMax H3 pre-cache requires [B,3,T,H,W]",
                  "El pre-cache de MiniMax H3 requiere [B,3,T,H,W]"))

        x = (x + 1.0) * 0.5
        x = (x - self.pixel_mean.to(x)) / self.pixel_std.to(x)

        frames = x.shape[2]
        if frames == 1:
            moments = self.quant_conv(self.encoder(x))
            moments = moments[:, :, -1:, :, :]
        else:
            # Relleno hasta multiplo de clip_length repitiendo el ULTIMO
            # fotograma. Esos latentes de relleno son justo los que se lleva
            # `token_drop` despues: las dos cosas estan disenadas para
            # cancelarse, y por eso 17n+5 -> 5n+2 sale exacto.
            # Pad to a multiple of clip_length by repeating the LAST frame. Those
            # padding latents are exactly what `token_drop` removes afterwards:
            # the two are designed to cancel, which is why 17n+5 -> 5n+2 is exact.
            if frames % H3_CLIP_LENGTH:
                falta = (-frames) % H3_CLIP_LENGTH
                x = torch.cat([x, x[:, :, -1:].repeat(1, 1, falta, 1, 1)], dim=2)

            # Trozo a trozo y no todo junto: con 73 fotogramas a 576x576 el pico
            # de activaciones del encoder no cabria en una tarjeta de 16 GB.
            # Chunk by chunk rather than all at once: with 73 frames at 576x576
            # the encoder's activation peak would not fit on a 16 GB card.
            trozos = []
            for i in range(x.shape[2] // H3_CLIP_LENGTH):
                corte = x[:, :, i * H3_CLIP_LENGTH:(i + 1) * H3_CLIP_LENGTH]
                trozos.append(self.quant_conv(self.encoder(corte)))
            moments = torch.cat(trozos, dim=2)
            if H3_TOKEN_DROP > 0:
                moments = moments[:, :, :-H3_TOKEN_DROP]

        mean = torch.chunk(moments.float(), 2, dim=1)[0]
        lm = self.latents_mean.view(1, -1, 1, 1, 1).to(mean)
        ls = self.latents_std.view(1, -1, 1, 1, 1).to(mean)
        return (mean - lm) / ls


def load_h3_video_vae(nf4_model_id, original_model_id, strict=True):
    """Load the encode-only H3 video VAE and VERIFY the load.
    Carga el VAE de video H3 (solo encode) y VERIFICA la carga."""
    candidates = [
        os.path.join(nf4_model_id, "vae", "minimax_h3_video_vae.safetensors"),
        os.path.join(nf4_model_id, "vae", "minimax_h3_video_vae_fp16.safetensors"),
        os.path.join(nf4_model_id, "minimax_h3_video_vae.safetensors"),
        os.path.join(nf4_model_id, "minimax_h3_video_vae_fp16.safetensors"),
        os.path.join(original_model_id, "vae", "minimax_h3_video_vae.safetensors"),
        os.path.join(original_model_id, "vae", "minimax_h3_video_vae_fp16.safetensors"),
        os.path.join(original_model_id, "minimax_h3_video_vae.safetensors"),
        os.path.join(original_model_id, "minimax_h3_video_vae_fp16.safetensors"),
    ]
    ckpt = next((p for p in candidates if os.path.isfile(p)), None)
    if ckpt is None:
        raise FileNotFoundError(
            L("H3 Video VAE checkpoint not found. Searched:",
              "No se encontro el checkpoint del Video VAE H3. Buscado en:") + "\n" +
            "\n".join(candidates))

    log_dev(L("[VAE-H3] Loading H3 encoder from: {}",
              "[VAE-H3] Cargando encoder H3 desde: {}").format(os.path.abspath(ckpt)))
    DIAG["vae"]["checkpoint"] = os.path.abspath(ckpt)

    vae = MiniMaxH3VideoVAEEncoder()

    with safe_open(ckpt, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
        state = {k: f.get_tensor(k) for k in all_keys
                 if k.startswith("encoder.") or k.startswith("quant_conv.")
                 or k in ("latents_mean", "latents_std")}

    DIAG["vae"]["ckpt_total_keys"] = len(all_keys)
    DIAG["vae"]["ckpt_encoder_keys"] = len(state)
    DIAG["vae"]["latents_stats_in_ckpt"] = bool(
        "latents_mean" in state and "latents_std" in state)

    missing, unexpected = vae.load_state_dict(state, strict=False)
    # Only encoder/quant_conv/latents_* matter here; pixel_* are non-persistent.
    # Aqui solo importan encoder/quant_conv/latents_*; pixel_* no son persistentes.
    real_missing = [k for k in missing if not k.startswith("pixel_")]

    DIAG["vae"]["missing"] = len(real_missing)
    DIAG["vae"]["unexpected"] = len(unexpected)
    DIAG["vae"]["missing_first"] = real_missing[:15]
    DIAG["vae"]["unexpected_first"] = list(unexpected)[:15]

    log_dev(L("[VAE-H3] load_state_dict -> missing={} unexpected={}",
              "[VAE-H3] load_state_dict -> faltantes={} inesperados={}")
            .format(len(real_missing), len(unexpected)))

    if real_missing:
        msg = ("H3 Video VAE: {} weights did NOT load -> the encoder is partially random "
               "and every latent would be garbage. First: {} / "
               "VAE de video H3: {} pesos NO se cargaron -> el encoder queda parcialmente "
               "aleatorio y todos los latentes serian basura. Primeros: {}"
               ).format(len(real_missing), real_missing[:10], len(real_missing), real_missing[:10])
        if strict:
            raise RuntimeError(msg)
        diag_warn(msg)

    if not DIAG["vae"]["latents_stats_in_ckpt"]:
        diag_warn(L("H3 Video VAE: latents_mean/std not present in the checkpoint; "
                    "using the hardcoded table. Verify against vae/config.json.",
                    "VAE de video H3: latents_mean/std no estan en el checkpoint; "
                    "se usa la tabla hardcodeada. Verificar contra vae/config.json."))

    # Record the actually-used normalization / Registrar la normalizacion realmente usada
    DIAG["vae"]["latents_mean_used"] = [round(float(v), 6) for v in vae.latents_mean.tolist()]
    DIAG["vae"]["latents_std_used"] = [round(float(v), 6) for v in vae.latents_std.tolist()]

    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


# ============================================================================
# CONFIG
# ============================================================================
DEFAULTS = {
    "model_id": "./MiniMax-H3",
    "nf4_model_id": "./MiniMax-H3-NF4",
    "dataset_path": "./dataset",
    "cache_dir": "./cached_data_minimaxh3_v3",
    "target_area": 512 * 512,
    "max_side": 1024,
    "multiple": 32,
    # Tokens de texto por caption. 100 son unas 75 palabras: de sobra para
    # un caption de dataset, y cada token cuesta VRAM en cada paso de
    # entrenamiento porque va en la secuencia empaquetada. Lo que pase de
    # aqui se trunca, asi que no tiene sentido generar captions mas largos.
    # Text tokens per caption. 100 is about 75 words: plenty for a dataset
    # caption, and every token costs VRAM on every training step because it
    # travels in the packed sequence. Anything past this is truncated, so
    # generating longer captions is pointless.
    "max_seq_len": 100,
    "frame_rate": 24.0,
    "num_frames": 17,
    "project_name": "",
    "trigger_word": "",
    "preview_custom_prompt": "",
    "text_encoder_hidden_layer": 50,
    "low_ram_threshold_gb": 48.0,
    "force_gpu": True,
    "allow_cpu_fallback": False,
    "logs_dev": 1,
    # --- new / nuevos ---
    "strict_load": True,          # abort on any weight that did not load / abortar si falta algun peso
    "require_captions": True,     # abort if any image has no .txt / abortar si alguna imagen no tiene .txt
    "write_audio_latent": True,   # keep writing *_audio_latent.pt for script 2 / seguir escribiendo para el script 2
    "audio_latent_channels": 32,  # H3 audio VAE latent dim / dim latente del VAE de audio H3
    "run_selftests": True,
    # Backfill for an NF4 export that only contains Linear layers.
    # Relleno para un export NF4 que solo contiene capas lineales.
    "fetch_missing_from_hub": True,
    "missing_tensors_repo": "MiniMaxAI/MiniMax-H3",
    "missing_tensors_subfolders": ["FL2VA/text_encoder", "text_encoder", ""],
}

CONFIG_PATH = "pre_cache_settings.json"

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

cfg = {}
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        log_dev(L("[OK] Configuration loaded: {}", "[OK] Configuracion cargada: {}").format(CONFIG_PATH))
    except Exception as e:
        log_error(L("[ERROR] Could not read {}: {}", "[ERROR] No se pudo leer {}: {}").format(CONFIG_PATH, e))
        cfg = {}
else:
    log_dev(L("[!] {} does not exist; using defaults.",
              "[!] No existe {}; usando valores por defecto.").format(CONFIG_PATH))


def cfg_get(key, default):
    if not isinstance(cfg, dict):
        return default
    for candidate in (key, key + " ", " " + key, " " + key + " "):
        if candidate in cfg:
            return cfg[candidate]
    return default


def _cfg_bool(key, default):
    value = cfg_get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _parse_logs_dev(value):
    try:
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, (int, float)):
            return 1 if int(value) != 0 else 0
        if isinstance(value, str):
            return 1 if value.strip().lower() in ("1", "true", "yes", "y", "on") else 0
    except Exception:
        pass
    return 1


MODEL_ID = str(cfg_get("model_id", DEFAULTS["model_id"])).strip()
NF4_MODEL_ID = str(cfg_get("nf4_model_id", DEFAULTS["nf4_model_id"])).strip()
DATASET_PATH = str(cfg_get("dataset_path", DEFAULTS["dataset_path"])).strip()
TARGET_AREA = int(cfg_get("target_area", DEFAULTS["target_area"]))
MAX_SIDE = int(cfg_get("max_side", DEFAULTS["max_side"]))
MULTIPLE = int(cfg_get("multiple", DEFAULTS["multiple"]))
MAX_SEQ_LEN = int(cfg_get("max_seq_len", DEFAULTS["max_seq_len"]))
FRAME_RATE = float(cfg_get("frame_rate", DEFAULTS["frame_rate"]))
NUM_FRAMES = int(cfg_get("num_frames", DEFAULTS["num_frames"]))
TRIGGER_WORD = str(cfg_get("trigger_word", DEFAULTS["trigger_word"])).strip()
PROJECT_NAME = str(cfg_get("project_name", DEFAULTS["project_name"])).strip()
PREVIEW_CUSTOM_PROMPT = str(cfg_get("preview_custom_prompt", DEFAULTS["preview_custom_prompt"])).strip()
TEXT_ENCODER_HIDDEN_LAYER = int(cfg_get("text_encoder_hidden_layer", DEFAULTS["text_encoder_hidden_layer"]))
LOW_RAM_THRESHOLD_GB = float(cfg_get("low_ram_threshold_gb", DEFAULTS["low_ram_threshold_gb"]))

STRICT_LOAD = _cfg_bool("strict_load", DEFAULTS["strict_load"])
REQUIRE_CAPTIONS = _cfg_bool("require_captions", DEFAULTS["require_captions"])
WRITE_AUDIO_LATENT = _cfg_bool("write_audio_latent", DEFAULTS["write_audio_latent"])
AUDIO_LATENT_CHANNELS = int(cfg_get("audio_latent_channels", DEFAULTS["audio_latent_channels"]))
RUN_SELFTESTS = _cfg_bool("run_selftests", DEFAULTS["run_selftests"])
FETCH_MISSING_FROM_HUB = _cfg_bool("fetch_missing_from_hub", DEFAULTS["fetch_missing_from_hub"])
MISSING_TENSORS_REPO = str(cfg_get("missing_tensors_repo",
                                   DEFAULTS["missing_tensors_repo"])).strip()
MISSING_TENSORS_SUBFOLDERS = cfg_get("missing_tensors_subfolders",
                                     DEFAULTS["missing_tensors_subfolders"])
if isinstance(MISSING_TENSORS_SUBFOLDERS, str):
    MISSING_TENSORS_SUBFOLDERS = [MISSING_TENSORS_SUBFOLDERS]
MISSING_TENSORS_SUBFOLDERS = list(MISSING_TENSORS_SUBFOLDERS or ["FL2VA/text_encoder",
                                                                "text_encoder", ""])

FORCE_GPU = True
ALLOW_CPU_FALLBACK = False
if isinstance(cfg, dict):
    FORCE_GPU = _cfg_bool("force_gpu", True)
    ALLOW_CPU_FALLBACK = _cfg_bool("allow_cpu_fallback", False)

LOGS_DEV = _parse_logs_dev(cfg_get("logs_dev", LOGS_DEV))

LOW_RAM_MODE = bool(
    LOW_RAM_THRESHOLD_GB > 0
    and SYSTEM_RAM_GB is not None
    and SYSTEM_RAM_GB <= LOW_RAM_THRESHOLD_GB
)

if PROJECT_NAME:
    CACHE_DIR = "./cached_data_minimaxh3_{}".format(PROJECT_NAME)
else:
    CACHE_DIR = str(cfg_get("cache_dir", DEFAULTS["cache_dir"])).strip()

# The cache path is used EXACTLY as configured - no automatic suffix.
# La ruta de cache se usa EXACTAMENTE como esta configurada - sin sufijo automatico.
# Reuse of a stale cache is prevented by the format-version check in preprocess_minimaxh3()
# instead of by renaming the folder.
# La reutilizacion de una cache obsoleta se evita con la comprobacion de version de formato
# en preprocess_minimaxh3(), no renombrando la carpeta.
# v4: la cache puede contener clips. Los .pt de video pasan de [1,24,1,h,w] a
# [1,24,T',h,w], asi que una cache v3 mezclada con una v4 daria geometrias
# distintas en el mismo entrenamiento. Subir la version fuerza a regenerarla.
# v4: the cache can hold clips. Video .pt files go from [1,24,1,h,w] to
# [1,24,T',h,w], so mixing a v3 cache with a v4 one would feed two different
# geometries into the same run. Bumping the version forces a rebuild.
CACHE_FORMAT_VERSION = 4

MULTIPLE = max(32, MULTIPLE)

if LOW_RAM_MODE and MAX_SEQ_LEN > 256:
    log_error(L("[WARN] Low RAM mode is capping max_seq_len {} -> 256. Long captions WILL be truncated.",
                "[WARN] El modo Low RAM limita max_seq_len {} -> 256. Los captions largos SE TRUNCARAN.")
              .format(MAX_SEQ_LEN))
    MAX_SEQ_LEN = 256
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")

# ============================================================================
# BANNER
# ============================================================================
print(L("[BOOT] 1_pre_cache_MiniMaxH3.py started",
        "[BOOT] 1_pre_cache_MiniMaxH3.py iniciado"), flush=True)
# El volcado [BOOT] de seis valores se elimino: MODEL_ID, NF4_MODEL_ID y
# STRICT_LOAD ya salen en el resumen de configuracion de preprocess_minimaxh3(),
# alli en bilingue y alineados, y los otros tres se han anadido a ese mismo
# resumen. Era ruido de depuracion duplicado y solo en ingles.
# The six-value [BOOT] dump was removed: it duplicated the configuration summary
# printed later by preprocess_minimaxh3(), and it was English-only debug noise.
print("", flush=True)


# ============================================================================
# UTILITIES / UTILIDADES
# ============================================================================
def ensure_nf4_model_exists(nf4_model_id, repo_id="AcademiaSD/MiniMax-H3-NF4"):
    """Download the NF4 repo if the local folder is missing or empty.
    Descarga el repo NF4 si la carpeta local no existe o esta vacia."""
    if not os.path.exists(nf4_model_id) or not os.path.isdir(nf4_model_id) or not os.listdir(nf4_model_id):
        log_dev("")
        log_dev("=" * 90)
        log_dev(L("[DOWNLOAD] NF4 folder missing or empty: {}",
                  "[DOWNLOAD] La carpeta NF4 no existe o esta vacia: {}").format(os.path.abspath(nf4_model_id)))
        log_dev(L("[DOWNLOAD] Downloading from Hugging Face ({}) ...",
                  "[DOWNLOAD] Descargando desde Hugging Face ({}) ...").format(repo_id))
        log_dev("=" * 90)
        os.makedirs(nf4_model_id, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=repo_id, local_dir=nf4_model_id)
            log_dev(L("[DOWNLOAD] Download completed at {}",
                      "[DOWNLOAD] Descarga completada en {}").format(os.path.abspath(nf4_model_id)))
        except Exception as e:
            diag_error("HF download failed / Fallo la descarga HF ({}): {}".format(repo_id, e))
            raise
    else:
        log_dev(L("[OK] NF4 folder found: {}",
                  "[OK] Carpeta NF4 encontrada: {}").format(os.path.abspath(nf4_model_id)))


def gc_cuda():
    """Collect + empty the CUDA cache / Recolecta y vacia la cache CUDA."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def vram_gb():
    return torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def vram_peak_gb():
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def json_safe(value):
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Size):
        return list(value)
    if torch.is_tensor(value):
        return {"tensor": True, "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def atomic_json(data, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def bucket_size(width, height):
    ar = width / height
    bh = math.sqrt(TARGET_AREA / ar)
    bw = ar * bh
    bw = max(MULTIPLE, round(bw / MULTIPLE) * MULTIPLE)
    bh = max(MULTIPLE, round(bh / MULTIPLE) * MULTIPLE)
    if max(bw, bh) > MAX_SIDE:
        scale = MAX_SIDE / max(bw, bh)
        bw = max(MULTIPLE, int(bw * scale) // MULTIPLE * MULTIPLE)
        bh = max(MULTIPLE, int(bh * scale) // MULTIPLE * MULTIPLE)
    bw, bh = int(bw), int(bh)
    # 16x VAE * 2x2 transformer patch -> must be a multiple of 32.
    # 16x del VAE * patch 2x2 del transformer -> debe ser multiplo de 32.
    assert bw % 32 == 0 and bh % 32 == 0, "bucket must be a multiple of 32 / el bucket debe ser multiplo de 32"
    return bw, bh


def read_prompt(base_name):
    """Return (prompt, had_caption_file) / Devuelve (prompt, habia_fichero_caption)."""
    path = os.path.join(DATASET_PATH, base_name + ".txt")
    prompt = ""
    had = False
    if os.path.exists(path):
        had = True
        with open(path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
    if TRIGGER_WORD and TRIGGER_WORD.lower() not in prompt.lower():
        prompt = "{}, {}".format(TRIGGER_WORD, prompt).strip(", ")
    return prompt, had


def save_prompt_result(result, prefix):
    def recurse(obj, path):
        if torch.is_tensor(obj):
            safe_path = path.replace(".", "_").replace("/", "_").replace("\\", "_")
            filename = "{}_{}.pt".format(prefix, safe_path)
            torch.save(obj.detach().cpu(), os.path.join(CACHE_DIR, filename))
            return {"type": "tensor", "file": filename,
                    "shape": list(obj.shape), "dtype": str(obj.dtype)}
        if isinstance(obj, dict):
            return {"type": "dict",
                    "items": {str(k): recurse(v, "{}_{}".format(path, k)) for k, v in obj.items()}}
        if isinstance(obj, (tuple, list)):
            return {"type": "tuple" if isinstance(obj, tuple) else "list",
                    "items": [recurse(v, "{}_{}".format(path, i)) for i, v in enumerate(obj)]}
        return {"type": "value", "value": json_safe(obj)}

    structure = recurse(result, "root")
    atomic_json(structure, os.path.join(CACHE_DIR, "{}_structure.json".format(prefix)))
    return structure


def extract_prompt_tensors(result):
    found = []

    def recurse(obj, path="root"):
        if torch.is_tensor(obj):
            found.append((path, obj.detach().cpu()))
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                recurse(v, "{}.{}".format(path, k))
            return
        if isinstance(obj, (tuple, list)):
            for i, v in enumerate(obj):
                recurse(v, "{}.{}".format(path, i))

    recurse(result)
    return found


def get_parent_module(root, name):
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def set_child_module(parent, child_name, new_module):
    if child_name.isdigit():
        parent[int(child_name)] = new_module
    else:
        setattr(parent, child_name, new_module)


# ============================================================================
# *** ROPE DIAGNOSTIC + REBUILD / DIAGNOSTICO Y RECONSTRUCCION DE ROPE ***
# ============================================================================
# THE BUG THIS FIXES / EL BUG QUE ESTO ARREGLA:
#   rotary_emb.inv_freq is a COMPUTED buffer (register_buffer(..., persistent=False)),
#   so it is NOT in the checkpoint state_dict and never loads from the NF4 index. Built
#   under torch.device("meta") it stays on meta, and the old materialize_meta_tensors()
#   filled it with ZEROS -> cos=1, sin=0 -> RoPE becomes the identity -> the text encoder
#   loses ALL positional information and behaves like a bag of words. No error, no NaN.
#
#   rotary_emb.inv_freq es un buffer CALCULADO (register_buffer(..., persistent=False)),
#   asi que NO esta en el state_dict del checkpoint y nunca se carga desde el indice NF4.
#   Construido bajo torch.device("meta") se queda en meta, y el antiguo
#   materialize_meta_tensors() lo rellenaba con CEROS -> cos=1, sin=0 -> RoPE se vuelve la
#   identidad -> el text encoder pierde TODA la informacion posicional y se comporta como
#   una bolsa de palabras. Sin error, sin NaN.
# ============================================================================
def _tensor_health(t):
    if t is None or not torch.is_tensor(t):
        return {"present": False}
    if t.is_meta:
        return {"present": True, "meta": True, "shape": list(t.shape), "numel": int(t.numel())}
    f = t.detach().float()
    total = float(f.abs().sum())
    return {
        "present": True,
        "meta": False,
        "numel": int(f.numel()),
        "abs_sum": total,
        "all_zeros": bool(total == 0.0),
        "min": float(f.min()),
        "max": float(f.max()),
        "first5": [float(v) for v in f.flatten()[:5].tolist()],
    }


def report_rope_state(model, tag):
    """Inspect every inv_freq buffer / Inspecciona cada buffer inv_freq."""
    rows = []
    for name, mod in model.named_modules():
        buf = getattr(mod, "inv_freq", None)
        if buf is None or not torch.is_tensor(buf):
            continue
        info = {"module": name}
        info.update(_tensor_health(buf))
        info["attention_scaling"] = float(getattr(mod, "attention_scaling", 1.0) or 1.0)
        rows.append(info)

    DIAG["rope"][tag] = rows

    log_dev("")
    log_dev("-" * 90)
    log_dev(L("[ROPE][{}] inv_freq buffers found: {}",
              "[ROPE][{}] buffers inv_freq encontrados: {}").format(tag, len(rows)))
    if not rows:
        diag_warn(L("No inv_freq buffer found in the text encoder - cannot verify RoPE.",
                    "No se encontro ningun buffer inv_freq en el text encoder - no se puede verificar RoPE."))
    for r in rows:
        if r.get("meta"):
            status = "META (never loaded / nunca cargado)"
        elif r.get("all_zeros"):
            status = "ALL-ZEROS  <== BROKEN / ROTO"
        else:
            status = "OK  min={:.3e} max={:.3e}".format(r["min"], r["max"])
        log_dev("[ROPE][{}] {:<52} -> {}".format(tag, r["module"], status))
    log_dev("-" * 90)
    return rows


VISION_ROPE_PREFIXES = ("visual.", "model.visual.")


def _theta_from_config(cfg):
    """Read rope_theta across transformers layouts / Lee rope_theta en cualquier layout.

    transformers >= 5 moved it into config.rope_parameters = {"rope_type", "rope_theta", ...};
    older versions expose config.rope_theta directly.
    transformers >= 5 lo movio a config.rope_parameters = {"rope_type", "rope_theta", ...};
    las versiones antiguas exponen config.rope_theta directamente.
    """
    if cfg is None:
        return None, None
    rp = getattr(cfg, "rope_parameters", None)
    if isinstance(rp, dict):
        for k in ("rope_theta", "theta", "base"):
            if rp.get(k):
                return float(rp[k]), "rope_parameters[{}]".format(k)
    for k in ("rope_theta", "theta", "rope_base"):
        v = getattr(cfg, k, None)
        if isinstance(v, (int, float)) and v:
            return float(v), "config.{}".format(k)
    return None, None


def _rope_theta_for(name, mod, fallback_config):
    """Pick the right theta for this rotary module / Elige el theta correcto para este modulo."""
    sources = [("module.config", getattr(mod, "config", None))]

    if name.startswith(VISION_ROPE_PREFIXES):
        # The vision tower has its OWN rope base (10000 for Qwen3-VL). It must never inherit
        # the language model's 5e6 — that would be a different positional geometry entirely.
        # La torre de vision tiene su PROPIA base de rope (10000 en Qwen3-VL). Nunca debe
        # heredar el 5e6 del modelo de lenguaje: seria otra geometria posicional distinta.
        sources.append(("vision_config", getattr(fallback_config, "vision_config", None)))
        for label, cfg in sources:
            theta, how = _theta_from_config(cfg)
            if theta:
                return theta, "{}.{}".format(label, how)
        return 10000.0, "vision-default(10000)"

    sources.append(("text_config", getattr(fallback_config, "text_config", None)))
    sources.append(("fallback_config", fallback_config))
    for label, cfg in sources:
        theta, how = _theta_from_config(cfg)
        if theta:
            return theta, "{}.{}".format(label, how)
    return 10000.0, "DEFAULT-NOT-FOUND"


def _inv_freq_from_official(mod, name, fallback_config, expect_numel):
    """Try the official transformers rope init (honours rope_scaling).
    Intenta el init oficial de rope de transformers (respeta rope_scaling)."""
    if name.startswith(VISION_ROPE_PREFIXES):
        return None, None
    cfg = getattr(mod, "config", None) or getattr(fallback_config, "text_config", None) or fallback_config
    if cfg is None:
        return None, None
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    except Exception:
        return None, None
    try:
        # transformers >= 5 renamed rope_scaling -> rope_parameters
        # transformers >= 5 renombro rope_scaling -> rope_parameters
        rs = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
        if not isinstance(rs, dict):
            rs = {}
        rope_type = rs.get("rope_type", rs.get("type", "default")) or "default"
        if rope_type not in ROPE_INIT_FUNCTIONS:
            rope_type = "default"
        inv_freq, _ = ROPE_INIT_FUNCTIONS[rope_type](cfg, device="cpu")
        inv_freq = inv_freq.detach().float().cpu()
        if int(inv_freq.numel()) != int(expect_numel):
            return None, None
        if float(inv_freq.abs().sum()) == 0.0:
            return None, None
        return inv_freq, "rope_init:{}".format(rope_type)
    except Exception:
        return None, None


def _inv_freq_manual(theta, n):
    """Standard RoPE table: inv_freq[i] = theta ** (-i/n), n = dim/2.
    Tabla RoPE estandar: inv_freq[i] = theta ** (-i/n), n = dim/2."""
    idx = torch.arange(int(n), dtype=torch.float32)
    return 1.0 / (float(theta) ** (idx / float(n)))


def rebuild_rope_buffers(model, fallback_config=None):
    """Rebuild inv_freq for any rotary module left on meta or zeroed.
    Reconstruye inv_freq para cualquier modulo rotary en meta o a ceros.

    Does NOT call the module's constructor: rotary classes disagree on their signature
    (config-based for the language model, (dim, theta) for the vision tower). Instead the
    RoPE table is recomputed from the buffer's own size and the config's theta.
    NO llama al constructor del modulo: las clases rotary no coinciden en su firma (basada
    en config para el modelo de lenguaje, (dim, theta) para la torre de vision). En su lugar
    se recalcula la tabla RoPE a partir del tamano del propio buffer y del theta del config.
    """
    fixed, details, failed = [], [], []

    for name, mod in model.named_modules():
        buf = getattr(mod, "inv_freq", None)
        if buf is None or not torch.is_tensor(buf):
            continue
        broken = bool(buf.is_meta) or (not buf.is_meta and float(buf.detach().abs().sum()) == 0.0)
        if not broken:
            continue

        n = int(buf.shape[-1])
        load_bearing = not name.startswith(VISION_ROPE_PREFIXES)

        try:
            new_buf, method = _inv_freq_from_official(mod, name, fallback_config, n)
            if new_buf is None:
                theta, theta_src = _rope_theta_for(name, mod, fallback_config)
                # A guessed theta silently produces a WRONG positional encoding. Qwen3-VL-32B
                # uses 5e6, not the 1e4 default: never let a load-bearing rope fall back.
                # Un theta adivinado produce en silencio una codificacion posicional ERRONEA.
                # Qwen3-VL-32B usa 5e6, no el 1e4 por defecto: nunca dejar que la rope critica
                # caiga al valor por defecto.
                if theta_src == "DEFAULT-NOT-FOUND" and load_bearing:
                    raise RuntimeError(
                        L("rope_theta not found in the config for '{}'. Falling back to 10000 "
                          "would build a WRONG RoPE table (Qwen3-VL-32B uses 5e6). Checked "
                          "config.rope_parameters and config.rope_theta.",
                          "No se encontro rope_theta en el config de '{}'. Caer a 10000 crearia "
                          "una tabla RoPE ERRONEA (Qwen3-VL-32B usa 5e6). Comprobados "
                          "config.rope_parameters y config.rope_theta.").format(name))
                new_buf = _inv_freq_manual(theta, n)
                method = "manual(theta={:g} from {})".format(theta, theta_src)

            # Sanity: standard RoPE tables start at exactly 1.0 and decrease monotonically.
            # Cordura: las tablas RoPE estandar empiezan en 1.0 y decrecen monotonamente.
            if int(new_buf.numel()) != n:
                raise RuntimeError("numel mismatch {} != {}".format(int(new_buf.numel()), n))
            if float(new_buf.abs().sum()) == 0.0:
                raise RuntimeError("rebuilt table is all zeros / la tabla reconstruida es todo ceros")
            if n > 1 and not bool((new_buf[1:] <= new_buf[:-1]).all()):
                raise RuntimeError("table is not monotonically decreasing / la tabla no decrece")

            mod.register_buffer("inv_freq", new_buf.clone(), persistent=False)
            orig = getattr(mod, "original_inv_freq", None)
            if torch.is_tensor(orig):
                mod.original_inv_freq = new_buf.clone()

            fixed.append(name)
            details.append({
                "module": name, "numel": n, "method": method, "load_bearing": load_bearing,
                "first": round(float(new_buf[0]), 8),
                "last": round(float(new_buf[-1]), 12),
                "attention_scaling": float(getattr(mod, "attention_scaling", 1.0) or 1.0),
            })
            log_dev(L("[ROPE] Rebuilt {:<44} n={:<4} via {}",
                      "[ROPE] Reconstruido {:<44} n={:<4} via {}").format(name, n, method))

        except Exception as e:
            failed.append({"module": name, "error": str(e), "load_bearing": load_bearing})
            if load_bearing:
                diag_error("Could not rebuild the LOAD-BEARING rope buffer / "
                           "No se pudo reconstruir el buffer rope CRITICO {}: {}".format(name, e))
                raise
            # The vision tower is never used for text-only captions; it will be zero-filled.
            # La torre de vision no se usa con captions de solo texto; se rellenara con ceros.
            diag_warn("Vision rope buffer not rebuilt (unused for text-only) / "
                      "Buffer rope de vision no reconstruido (no se usa en solo-texto) {}: {}"
                      .format(name, e))

    DIAG["rope"]["rebuilt_modules"] = fixed
    DIAG["rope"]["rebuilt_count"] = len(fixed)
    DIAG["rope"]["rebuilt_details"] = details
    DIAG["rope"]["rebuild_failures"] = failed
    return fixed


# ============================================================================
# META TENSOR HANDLING (STRICT) / MANEJO DE TENSORES META (ESTRICTO)
# ============================================================================
# Anything under these prefixes is allowed to stay uninitialized: the vision tower and the
# LM head are never used for text-only captions (same allowlist ai-toolkit uses).
# Lo que este bajo estos prefijos puede quedar sin inicializar: la torre de vision y la
# cabeza LM no se usan para captions de solo texto (misma allowlist que usa ai-toolkit).
META_ALLOWED_PREFIXES = ("visual.", "model.visual.", "lm_head")


def audit_meta_tensors(module, name):
    """Return (blocking, allowed) meta tensor names / Devuelve nombres de tensores meta."""
    blocking, allowed = [], []
    for pname, p in module.named_parameters():
        if p.device.type == "meta":
            (allowed if pname.startswith(META_ALLOWED_PREFIXES) else blocking).append(pname)
    for bname, b in module.named_buffers():
        if b.device.type == "meta":
            (allowed if bname.startswith(META_ALLOWED_PREFIXES) else blocking).append(bname)

    DIAG.setdefault("meta", {})[name] = {
        "blocking_count": len(blocking),
        "blocking_first": blocking[:25],
        "allowed_count": len(allowed),
    }
    return blocking, allowed


def materialize_allowed_meta(module):
    """Zero-fill ONLY the allowlisted (unused) meta tensors.
    Rellena con ceros SOLO los tensores meta de la allowlist (no usados)."""
    n = 0
    for name, param in list(module.named_parameters()):
        if param.device.type == "meta" and name.startswith(META_ALLOWED_PREFIXES):
            parent, child = get_parent_module(module, name)
            new_p = nn.Parameter(torch.zeros(param.shape, dtype=param.dtype, device="cpu"),
                                 requires_grad=False)
            set_child_module(parent, child, new_p) if child.isdigit() else setattr(parent, child, new_p)
            n += 1
    for name, buf in list(module.named_buffers()):
        if buf.device.type == "meta" and name.startswith(META_ALLOWED_PREFIXES):
            parent, child = get_parent_module(module, name)
            parent.register_buffer(child, torch.zeros(buf.shape, dtype=buf.dtype, device="cpu"))
            n += 1
    if n:
        log_dev(L("[META] Zero-filled {} unused (vision/lm_head) tensors.",
                  "[META] Rellenados con ceros {} tensores no usados (vision/lm_head).").format(n))
    return n


def force_cuda_or_die(module, name, strict_meta=True):
    """Move a module to CUDA, refusing to proceed with uninitialized weights.
    Mueve un modulo a CUDA, negandose a continuar con pesos sin inicializar."""
    if module is None:
        raise RuntimeError(L("{} is None and cannot be moved to CUDA.",
                             "{} es None y no puede moverse a CUDA.").format(name))

    blocking, allowed = audit_meta_tensors(module, name)
    if blocking:
        msg = ("{}: {} tensors are still on the META device and would be silently zero-filled "
               "(this destroys the model). First: {} / "
               "{}: {} tensores siguen en META y se rellenarian con ceros en silencio "
               "(esto destruye el modelo). Primeros: {}"
               ).format(name, len(blocking), blocking[:10], name, len(blocking), blocking[:10])
        if strict_meta and STRICT_LOAD:
            raise RuntimeError(msg)
        diag_warn(msg)

    materialize_allowed_meta(module)

    log_dev(L("[GPU] Moving {} to CUDA...", "[GPU] Moviendo {} a CUDA...").format(name))
    try:
        module.to("cuda")
    except Exception as e:
        if FORCE_GPU and not ALLOW_CPU_FALLBACK:
            raise RuntimeError(L("{} could not be moved to CUDA (ALLOW_CPU_FALLBACK=False): {}",
                                 "{} no se pudo mover a CUDA (ALLOW_CPU_FALLBACK=False): {}").format(name, e)) from e
        diag_warn("{} stays on CPU / se queda en CPU: {}".format(name, e))
        return False

    bad = [("{}:{}".format(n, p.device)) for n, p in module.named_parameters() if p.device.type != "cuda"]
    bad += [("{}:{}".format(n, b.device)) for n, b in module.named_buffers() if b.device.type != "cuda"]
    if bad:
        if FORCE_GPU and not ALLOW_CPU_FALLBACK:
            raise RuntimeError(L("{} is not 100% on CUDA. First off-device: {}",
                                 "{} no quedo 100% en CUDA. Primeros fuera: {}").format(name, bad[:20]))
        diag_warn("{} not fully on CUDA / no quedo 100% en CUDA: {}".format(name, bad[:10]))
        return False

    log_dev(L("[GPU] {} is 100% on CUDA. VRAM: {:.2f} GB",
              "[GPU] {} esta 100% en CUDA. VRAM: {:.2f} GB").format(name, vram_gb()))
    return True


# ============================================================================
# NF4 TEXT ENCODER LOADING / CARGA NF4 DEL TEXT ENCODER
# ============================================================================
def _find_nf4_index(nf4_te_dir):
    for candidate in ("index.json", "config_nf4.json"):
        p = os.path.join(nf4_te_dir, candidate)
        if os.path.exists(p):
            log_dev(L("[NF4-TE] Index found: {}", "[NF4-TE] Indice encontrado: {}").format(p))
            with open(p, "r", encoding="utf-8") as f:
                return p, json.load(f)
    return None, None


def _norm_key(s):
    """Normalize a name/filename for fuzzy comparison / Normaliza nombre o fichero para comparar."""
    s = os.path.basename(str(s))
    for ext in (".safetensors", ".pt", ".bin"):
        if s.lower().endswith(ext):
            s = s[: -len(ext)]
    return "".join(ch for ch in s.lower() if ch.isalnum())


class WeightInventory(object):
    """Everything we can learn about where each tensor physically lives.
    Todo lo que podemos averiguar sobre donde vive fisicamente cada tensor.

    This NF4 export stores ONE FILE PER TENSOR with generic keys ("weight", "bias",
    "quant_state.*"); the tensor identity is in the FILENAME and in index.json, not in the
    key. Standard HF shards instead use full-path keys. Both layouts are indexed here.
    Este export NF4 guarda UN FICHERO POR TENSOR con claves genericas ("weight", "bias",
    "quant_state.*"); la identidad del tensor esta en el NOMBRE DEL FICHERO y en index.json,
    no en la clave. Los shards HF estandar usan claves con ruta completa. Aqui se indexan ambos.
    """

    def __init__(self):
        self.name_to_file = {}    # index.json name            -> path
        self.fullkey = {}         # full-path tensor key       -> {file, shape, dtype}
        self.by_basename = {}     # normalized filename        -> path
        self.by_shape = {}        # shape tuple                -> [(path, key)]
        self.files = []
        self.generic_files = 0


def build_weight_inventory(index, weights_dir, *extra_dirs):
    """Scan every .safetensors and index it by name, by key, by filename and by shape.
    Escanea todos los .safetensors y los indexa por nombre, clave, fichero y shape."""
    inv = WeightInventory()

    # --- 1) name -> file, straight from index.json (authoritative when present) ---
    # --- 1) nombre -> fichero, directo de index.json (autoritativo cuando existe) ---
    for section in ("quantized", "unquantized", "other"):
        sec = index.get(section) or {}
        for name, info in sec.items():
            fn = info.get("file") if isinstance(info, dict) else info
            if not fn:
                continue
            inv.name_to_file[name] = fn if os.path.isabs(fn) else os.path.join(weights_dir, fn)
    for alias, real in (index.get("aliases") or {}).items():
        if real in inv.name_to_file:
            inv.name_to_file[alias] = inv.name_to_file[real]

    # --- 2) walk the directories (deduplicated: weights_dir usually lives inside nf4_te_dir,
    #        so a naive walk of both counts every file twice)
    # --- 2) recorrer los directorios (deduplicado: weights_dir suele estar dentro de
    #        nf4_te_dir, asi que recorrer ambos contaria cada fichero dos veces)
    _seen_paths = set()
    for d in (weights_dir,) + tuple(extra_dirs):
        if not d or not os.path.isdir(d):
            continue
        for root, _, names in os.walk(d):
            for n in sorted(names):
                if n.lower().endswith(".safetensors"):
                    p = os.path.normcase(os.path.abspath(os.path.join(root, n)))
                    if p in _seen_paths:
                        continue
                    _seen_paths.add(p)
                    inv.files.append(os.path.join(root, n))

    for p in inv.files:
        inv.by_basename.setdefault(_norm_key(p), p)
        try:
            with safe_open(p, framework="pt", device="cpu") as f:
                keys = [k for k in f.keys() if not k.startswith("quant_state.")]
                generic_only = all(k in ("weight", "bias") for k in keys) if keys else False
                if generic_only:
                    inv.generic_files += 1
                for k in keys:
                    try:
                        sl = f.get_slice(k)
                        shape, dtype = tuple(sl.get_shape()), str(sl.get_dtype())
                    except Exception:
                        t = f.get_tensor(k)
                        shape, dtype = tuple(t.shape), str(t.dtype)
                    if "." in k and k not in ("weight", "bias"):
                        inv.fullkey.setdefault(k, {"file": p, "shape": shape, "dtype": dtype})
                    inv.by_shape.setdefault(shape, []).append((p, k))
        except Exception as e:
            diag_warn("could not scan / no se pudo escanear {}: {}".format(os.path.basename(p), e))

    log_dev(L("[SCAN] {} files | {} index names | {} full-path keys | {} per-tensor generic-key files",
              "[SCAN] {} ficheros | {} nombres de indice | {} claves ruta-completa | {} ficheros de clave generica")
            .format(len(inv.files), len(inv.name_to_file), len(inv.fullkey), inv.generic_files))

    DIAG["text_encoder"]["inventory"] = {
        "files": len(inv.files),
        "index_names": len(inv.name_to_file),
        "fullpath_keys": len(inv.fullkey),
        "generic_key_files": inv.generic_files,
        "sample_filenames": [os.path.basename(p) for p in inv.files[:50]],
        "sample_index_names": sorted(inv.name_to_file.keys())[:30],
        "index_names_with_norm": [k for k in sorted(inv.name_to_file) if "norm" in k][:15],
        "index_names_with_embed": [k for k in sorted(inv.name_to_file) if "embed" in k][:15],
        "sample_fullpath_keys": sorted(inv.fullkey.keys())[:30],
    }
    return inv


def _read_matching_tensor(path, wanted_name, wanted_shape):
    """Read the tensor from `path` whose shape matches, regardless of its key name.
    Lee del fichero `path` el tensor cuya shape coincide, sea cual sea su clave."""
    if not os.path.isfile(path):
        return None, "file missing / falta el fichero"
    leaf = wanted_name.rsplit(".", 1)[-1]
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = [k for k in f.keys() if not k.startswith("quant_state.")]
        # Preference order: exact full name, then the leaf ("weight"/"bias"), then anything.
        # Orden de preferencia: nombre completo, luego la hoja ("weight"/"bias"), luego cualquiera.
        ordered = ([wanted_name] if wanted_name in keys else []) \
            + [k for k in keys if k == leaf] \
            + [k for k in keys if k not in (wanted_name, leaf)]
        for k in ordered:
            t = f.get_tensor(k)
            if tuple(t.shape) == tuple(wanted_shape):
                return t, k
        return None, "no key with shape {} (keys: {}) / ninguna clave con shape {} (claves: {})".format(
            list(wanted_shape), keys[:6], list(wanted_shape), keys[:6])


def _name_candidates(model_name):
    """Checkpoint key candidates for a model parameter name.
    Candidatos de clave de checkpoint para un nombre de parametro del modelo.

    Qwen3VLForConditionalGeneration expects `model.language_model.*` / `model.visual.*`,
    while single-file H3 checkpoints commonly store `model.*` / `visual.*`.
    Qwen3VLForConditionalGeneration espera `model.language_model.*` / `model.visual.*`,
    mientras los checkpoints H3 de fichero unico suelen guardar `model.*` / `visual.*`.
    """
    out = [model_name]
    rules = [
        ("model.language_model.", "model."),
        ("model.language_model.", ""),
        ("model.language_model.", "language_model."),
        ("model.language_model.", "model.model."),
        ("model.visual.", "visual."),
        ("model.", ""),
        ("language_model.", ""),
    ]
    for pre, rep in rules:
        if model_name.startswith(pre):
            out.append(rep + model_name[len(pre):])
    out.append("model." + model_name)
    out.append("model.language_model." + model_name)
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


_QUANT_DTYPES = ("int8", "uint8", "torch.int8", "torch.uint8", "I8", "U8", "F8_E4M3", "F8_E5M2")


def resolve_meta_from_checkpoint(model, inv, compute_dtype=torch.bfloat16):
    """Fill every remaining meta parameter/buffer, trying four location strategies.
    Rellena todo parametro/buffer meta restante, probando cuatro estrategias de localizacion.

    S1 index.json name -> file (key-agnostic: matches by shape inside the file)
    S2 full-path tensor key (standard HF shards)
    S3 normalized filename match
    S4 globally unique shape
    """
    targets = []
    for n, p in model.named_parameters():
        if p.device.type == "meta":
            targets.append((n, p, True))
    for n, b in model.named_buffers():
        if b.device.type == "meta":
            targets.append((n, b, False))

    matched, by_rule, unmatched, quant_blocked = 0, {}, [], []

    for name, tensor, is_param in targets:
        want = tuple(tensor.shape)
        cands = _name_candidates(name)
        w = None
        rule = None
        notes = []

        # --- S1: index.json name -> file, then match by shape inside that file ---
        for c in cands:
            path = inv.name_to_file.get(c)
            if path is None:
                continue
            t, info = _read_matching_tensor(path, c, want)
            if t is not None:
                w, rule = t, "S1 index[{}] key='{}'".format("exact" if c == name else "mapped", info)
                break
            notes.append("S1 {}: {}".format(os.path.basename(path), info))

        # --- S2: full-path tensor key (standard HF shards) ---
        if w is None:
            for c in cands:
                info = inv.fullkey.get(c)
                if info is not None and tuple(info["shape"]) == want:
                    if any(q in info["dtype"] for q in _QUANT_DTYPES):
                        quant_blocked.append({"name": name, "ckpt_key": c, "dtype": info["dtype"],
                                              "file": os.path.basename(info["file"])})
                        break
                    with safe_open(info["file"], framework="pt", device="cpu") as f:
                        w = f.get_tensor(c)
                    rule = "S2 fullkey[{}]".format("exact" if c == name else "mapped")
                    break

        # --- S3: normalized filename match ---
        if w is None:
            for c in cands:
                path = inv.by_basename.get(_norm_key(c))
                if path is None:
                    continue
                t, info = _read_matching_tensor(path, c, want)
                if t is not None:
                    w, rule = t, "S3 filename"
                    break

        # --- S4: globally unique shape (safe only when exactly one candidate exists) ---
        if w is None:
            hits = inv.by_shape.get(want, [])
            if len(hits) == 1:
                path, key = hits[0]
                with safe_open(path, framework="pt", device="cpu") as f:
                    w = f.get_tensor(key)
                rule = "S4 unique-shape"

        if w is None:
            unmatched.append({"name": name, "shape": list(want),
                              "tried": cands[:5],
                              "shape_candidates": len(inv.by_shape.get(want, [])),
                              "notes": notes[:2]})
            continue

        if not w.is_floating_point() and w.dtype not in (torch.bool,):
            quant_blocked.append({"name": name, "dtype": str(w.dtype), "rule": rule})
            continue

        w = w.to(compute_dtype)
        parent, child = get_parent_module(model, name)
        if is_param:
            setattr(parent, child, nn.Parameter(w, requires_grad=False))
        else:
            parent.register_buffer(child, w)
        by_rule[rule.split(" key=")[0]] = by_rule.get(rule.split(" key=")[0], 0) + 1
        matched += 1

    report = {
        "targets": len(targets),
        "matched": matched,
        "by_rule": by_rule,
        "unmatched_count": len(unmatched),
        "unmatched_first": unmatched[:25],
        "quantized_blocked_count": len(quant_blocked),
        "quantized_blocked_first": quant_blocked[:15],
    }
    DIAG["text_encoder"]["meta_resolution"] = report

    log_dev("")
    log_dev(L("[RESOLVE] meta tensors: {} | resolved: {} | unmatched: {} | quantized-blocked: {}",
              "[RESOLVE] tensores meta: {} | resueltos: {} | sin resolver: {} | bloqueados por cuantizacion: {}")
            .format(len(targets), matched, len(unmatched), len(quant_blocked)))
    for rule, cnt in sorted(by_rule.items(), key=lambda kv: -kv[1]):
        log_dev(L("[RESOLVE]   {} -> {} tensors", "[RESOLVE]   {} -> {} tensores").format(rule, cnt))
    for u in unmatched[:12]:
        log_dev(L("[RESOLVE]   UNMATCHED {} shape={} shape_candidates={}",
                  "[RESOLVE]   SIN RESOLVER {} shape={} candidatos_por_shape={}")
                .format(u["name"], u["shape"], u["shape_candidates"]))
        for nt in u["notes"]:
            log_dev("[RESOLVE]       {}".format(nt))
    for q in quant_blocked[:10]:
        log_error(L("[RESOLVE]   QUANTIZED (needs dequant) {} dtype={}",
                    "[RESOLVE]   CUANTIZADO (requiere dequant) {} dtype={}")
                  .format(q["name"], q.get("dtype")))

    if quant_blocked:
        diag_error(L("Some non-Linear tensors are stored in an integer dtype and need dequantizing.",
                     "Algunos tensores no-Lineales estan guardados en enteros y requieren dequantizacion."))
    return report


# ============================================================================
# HUB BACKFILL FOR AN INCOMPLETE NF4 EXPORT
# RELLENO DESDE EL HUB PARA UN EXPORT NF4 INCOMPLETO
# ============================================================================
# Some NF4 conversion scripts export only the Linear layers, leaving embed_tokens and every
# RMSNorm out of the index and off the disk entirely. Those tensors are tiny compared to the
# full checkpoint (embed_tokens ~1.6 GB bf16, all 200 norms together ~2 MB), so instead of
# pulling the 62 GB text encoder we parse each shard's safetensors header over HTTP and
# byte-range read ONLY the tensors we need, then cache them locally.
# Algunos scripts de conversion NF4 exportan solo las capas lineales y dejan fuera del indice
# (y del disco) embed_tokens y todas las RMSNorm. Esos tensores son minusculos comparados con
# el checkpoint completo (embed_tokens ~1,6 GB en bf16, las 200 normas juntas ~2 MB), asi que
# en vez de bajar el text encoder de 62 GB leemos la cabecera safetensors de cada shard por
# HTTP y descargamos SOLO los rangos de bytes necesarios, cacheandolos en local.
# ============================================================================
_ST_DTYPES = {
    "BOOL": torch.bool, "U8": torch.uint8, "I8": torch.int8,
    "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
    "F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32, "F64": torch.float64,
}

BACKFILL_CACHE_NAME = "nonlinear_backfill.safetensors"


def _read_st_header(fobj):
    """Parse a safetensors header from a seekable stream / Lee la cabecera safetensors."""
    n = int.from_bytes(fobj.read(8), "little")
    hdr = json.loads(fobj.read(n).decode("utf-8"))
    return hdr, 8 + n


def _layer_index_of(name):
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def backfill_missing_from_hub(model, nf4_te_dir, repo_id, subfolders, max_layer):
    """Byte-range fetch the still-missing non-Linear tensors and cache them locally.
    Descarga por rangos de bytes los tensores no-Lineales que faltan y los cachea en local."""
    wanted = []
    for n, p in model.named_parameters():
        if p.device.type == "meta" and not n.startswith(META_ALLOWED_PREFIXES):
            li = _layer_index_of(n)
            if li is None or li < max_layer:
                wanted.append((n, tuple(p.shape), True))
    for n, b in model.named_buffers():
        if b.device.type == "meta" and not n.startswith(META_ALLOWED_PREFIXES):
            li = _layer_index_of(n)
            if li is None or li < max_layer:
                wanted.append((n, tuple(b.shape), False))

    report = {"requested": len(wanted), "fetched": 0, "repo": repo_id,
              "subfolder": None, "shards": 0, "bytes": 0, "unmatched": []}
    DIAG["text_encoder"]["hub_backfill"] = report
    if not wanted:
        return report

    log_dev("")
    log_dev("=" * 90)
    log_dev(L("[BACKFILL] {} non-Linear tensors are missing from the NF4 export.",
              "[BACKFILL] Faltan {} tensores no-Lineales en el export NF4.").format(len(wanted)))
    log_dev(L("[BACKFILL] Fetching only their byte ranges from {} (not the whole checkpoint).",
              "[BACKFILL] Descargando solo sus rangos de bytes desde {} (no el checkpoint entero).")
            .format(repo_id))
    log_dev("=" * 90)

    from huggingface_hub import hf_hub_download, HfFileSystem

    index_data, chosen_sub = None, None
    for sub in subfolders:
        try:
            p = hf_hub_download(repo_id=repo_id, filename="model.safetensors.index.json",
                                subfolder=sub or None)
            with open(p, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            chosen_sub = sub
            log_dev(L("[BACKFILL] Weight map found at subfolder '{}'.",
                      "[BACKFILL] Mapa de pesos encontrado en la subcarpeta '{}'.").format(sub or "<root>"))
            break
        except Exception as e:
            log_dev(L("[BACKFILL] no weight map at '{}': {}",
                      "[BACKFILL] sin mapa de pesos en '{}': {}").format(sub or "<root>", e), level=2)
    if index_data is None:
        raise RuntimeError(
            L("Could not fetch model.safetensors.index.json from {} (tried subfolders {}).",
              "No se pudo obtener model.safetensors.index.json de {} (subcarpetas probadas {}).")
            .format(repo_id, subfolders))

    report["subfolder"] = chosen_sub
    weight_map = index_data.get("weight_map") or {}

    plan, unmatched = {}, []
    for name, shape, is_param in wanted:
        hit = next((c for c in _name_candidates(name) if c in weight_map), None)
        if hit is None:
            unmatched.append(name)
            continue
        plan.setdefault(weight_map[hit], []).append((name, hit, shape, is_param))

    report["unmatched"] = unmatched[:20]
    report["shards"] = len(plan)
    log_dev(L("[BACKFILL] {} tensors mapped across {} shards ({} unmatched).",
              "[BACKFILL] {} tensores mapeados en {} shards ({} sin mapear).")
            .format(len(wanted) - len(unmatched), len(plan), len(unmatched)))

    fs = HfFileSystem()
    fetched, total_bytes = {}, 0

    for shard, items in sorted(plan.items()):
        remote = "/".join(x for x in (repo_id, chosen_sub, shard) if x)
        log_dev(L("[BACKFILL] {} -> {} tensors", "[BACKFILL] {} -> {} tensores")
                .format(shard, len(items)))
        with fs.open(remote, "rb") as f:
            hdr, base = _read_st_header(f)
            for name, ckpt_key, shape, is_param in items:
                meta = hdr.get(ckpt_key)
                if meta is None:
                    unmatched.append(name)
                    continue
                dt = _ST_DTYPES.get(meta["dtype"])
                if dt is None:
                    diag_warn("unsupported dtype / dtype no soportado {} for {}".format(meta["dtype"], name))
                    continue
                s, e = meta["data_offsets"]
                f.seek(base + s)
                buf = bytearray(f.read(e - s))
                total_bytes += len(buf)
                t = torch.frombuffer(buf, dtype=dt).reshape(tuple(meta["shape"])).clone()
                if tuple(t.shape) != tuple(shape):
                    diag_warn("shape mismatch / shape distinta {}: {} vs {}"
                              .format(name, list(t.shape), list(shape)))
                    continue
                fetched[name] = t

    report["fetched"] = len(fetched)
    report["bytes"] = total_bytes
    log_dev(L("[BACKFILL] Fetched {} tensors, {:.2f} MB.",
              "[BACKFILL] Descargados {} tensores, {:.2f} MB.")
            .format(len(fetched), total_bytes / 1e6))

    if fetched:
        # Cache with FULL-PATH model keys so the next run resolves them via strategy S2.
        # Cachear con claves de ruta completa para que la proxima corrida las resuelva con S2.
        try:
            from safetensors.torch import save_file
            extra_dir = os.path.join(nf4_te_dir, "weights_extra")
            os.makedirs(extra_dir, exist_ok=True)
            cache_path = os.path.join(extra_dir, BACKFILL_CACHE_NAME)
            save_file({k: v.contiguous() for k, v in fetched.items()}, cache_path)
            report["cache_file"] = cache_path
            log_dev(L("[BACKFILL] Cached to {} - later runs will not re-download.",
                      "[BACKFILL] Cacheado en {} - las siguientes corridas no volveran a descargar.")
                    .format(cache_path))
        except Exception as e:
            diag_warn("could not cache backfill / no se pudo cachear el backfill: {}".format(e))

        for name, t in fetched.items():
            parent, child = get_parent_module(model, name)
            is_param = name in dict(model.named_parameters())
            val = t.to(torch.bfloat16) if t.is_floating_point() else t
            if is_param:
                setattr(parent, child, nn.Parameter(val, requires_grad=False))
            else:
                parent.register_buffer(child, val)

    return report


def verify_language_weights(model):
    """Sanity-check the tensors that silently destroyed the encoder when zero-filled.
    Comprueba los tensores que destruian el encoder en silencio al rellenarse con ceros."""
    checks = {}
    named = dict(model.named_parameters())

    emb = next((v for k, v in named.items() if k.endswith("embed_tokens.weight")), None)
    if emb is None:
        diag_error(L("embed_tokens.weight not found in the model.",
                     "no se encontro embed_tokens.weight en el modelo."))
    elif emb.is_meta:
        checks["embed_tokens"] = {"shape": list(emb.shape), "meta": True}
        diag_error(L("embed_tokens is still on META - it never loaded.",
                     "embed_tokens sigue en META - nunca se cargo."))
    else:
        e = emb.detach().float()
        checks["embed_tokens"] = {
            "shape": list(emb.shape), "meta": False,
            "abs_mean": float(e.abs().mean()), "std": float(e.std()),
            "all_zeros": bool(float(e.abs().sum()) == 0.0),
        }
        if checks["embed_tokens"]["all_zeros"]:
            diag_error(L("embed_tokens is ALL ZEROS - every token maps to the null vector.",
                         "embed_tokens es TODO CEROS - cada token se convierte en el vector nulo."))

    norm_names = [k for k in named if k.endswith(("q_norm.weight", "k_norm.weight",
                                                  "input_layernorm.weight",
                                                  "post_attention_layernorm.weight"))]
    zero_norms = []
    for k in norm_names:
        v = named[k].detach()
        if not v.is_meta and float(v.abs().sum()) == 0.0:
            zero_norms.append(k)
    checks["rms_norms"] = {
        "total": len(norm_names),
        "all_zero_count": len(zero_norms),
        "all_zero_first": zero_norms[:10],
    }
    if norm_names and not named[norm_names[0]].is_meta:
        sample = named[norm_names[0]].detach().float()
        checks["rms_norms"]["sample_name"] = norm_names[0]
        checks["rms_norms"]["sample_mean"] = float(sample.mean())
    checks["rms_norms"]["still_meta_count"] = sum(
        1 for k in norm_names if named[k].is_meta)

    DIAG["text_encoder"]["weight_checks"] = checks

    log_dev("")
    log_dev("[VERIFY] embed_tokens: {}".format(checks.get("embed_tokens", "MISSING")))
    log_dev(L("[VERIFY] RMSNorm weights: {} total, {} all-zero",
              "[VERIFY] pesos RMSNorm: {} en total, {} a cero")
            .format(checks["rms_norms"]["total"], checks["rms_norms"]["all_zero_count"]))

    if zero_norms:
        diag_error(L("{} RMSNorm weights are all zero - the layer output collapses to zero.",
                     "{} pesos RMSNorm estan a cero - la salida de la capa colapsa a cero.")
                   .format(len(zero_norms)))
    return checks


def new_layer_empty(cls, in_features, out_features, bias=False, **kwargs):
    """Crea una capa lineal SIN inicializar sus pesos.

    nn.Linear.__init__ reserva la matriz completa y la rellena con kaiming
    uniform. Ese relleno se descarta acto seguido, al asignarle el peso real del
    checkpoint, pero cuesta lo suyo: medido sobre los tamanos de este modelo son
    ~0,58 s por capa grande, y con cientos de capas es casi todo el tiempo de
    arranque. En el entrenador esto suponia 133 de los 133 segundos de carga.

    Construyendola en el dispositivo `meta` no se reserva memoria ni se rellena
    nada; `to_empty` reserva luego el hueco en CPU sin escribir en el. La capa
    queda identica.

    Creates a linear layer WITHOUT initializing its weights. nn.Linear.__init__
    allocates the full matrix and fills it with kaiming uniform; that fill is
    discarded immediately when the checkpoint weight is assigned, yet it costs
    ~0.58 s per large layer -- in the trainer it was 133 of the 133 seconds of
    load time. Building on `meta` allocates and fills nothing; `to_empty` then
    reserves CPU storage without writing to it.
    """
    with torch.device("meta"):
        layer = cls(in_features, out_features, bias=bias, **kwargs)
    return layer.to_empty(device="cpu")


def load_text_encoder_from_nf4(nf4_model_id, original_model_id, full_model=False):
    """Load the Qwen3-VL text encoder from the NF4 export.
    Carga el text encoder Qwen3-VL desde el export NF4.

    full_model=False (por defecto) da el modelo que necesita el pre-cache: el
    decoder truncado en TEXT_ENCODER_HIDDEN_LAYER y la lm_head sustituida por
    Identity, porque H3 solo consume hidden_states[50] y la cabeza son ~3,1 GB
    de peso muerto.

    full_model=True da el modelo COMPLETO, con sus 64 capas y su lm_head real.
    Hace falta para generar texto (el auto-captioning): un modelo truncado y sin
    cabeza no puede producir un solo token. El repo NF4 conserva ambas cosas.

    full_model=False (default) yields what the pre-cache needs: the decoder
    truncated at TEXT_ENCODER_HIDDEN_LAYER and lm_head replaced by Identity,
    since H3 only consumes hidden_states[50] and the head is ~3.1 GB of dead
    weight. full_model=True yields the COMPLETE model, all 64 layers and the real
    lm_head, which is what generating text (auto-captioning) requires: a
    truncated, headless model cannot emit a single token."""
    from bitsandbytes.nn import Linear4bit, Params4bit

    nf4_te_dir = os.path.join(nf4_model_id, "text_encoder")
    orig_te_dir = os.path.join(original_model_id, "text_encoder")

    if not os.path.isdir(nf4_te_dir):
        raise FileNotFoundError(L("NF4 text_encoder folder not found: {}",
                                  "No existe la carpeta NF4 del text_encoder: {}").format(nf4_te_dir))

    log_dev("")
    log_dev("=" * 90)
    log_dev(L("[NF4-TE] Loading text_encoder from NF4",
              "[NF4-TE] Cargando text_encoder desde NF4"))
    log_dev("=" * 90)

    config_path = os.path.join(nf4_te_dir, "config.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(orig_te_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(L("text_encoder config.json not found.",
                                  "No se encuentra config.json del text_encoder."))

    index_path, index = _find_nf4_index(nf4_te_dir)
    if index is None:
        raise FileNotFoundError(L("Neither index.json nor config_nf4.json found in {}",
                                  "No se encontro index.json ni config_nf4.json en {}").format(nf4_te_dir))

    quantized = index.get("quantized", {})
    unquantized = index.get("unquantized", {})
    other_tensors = index.get("other", {})
    aliases = index.get("aliases", {})
    weights_dir = os.path.join(nf4_te_dir, "weights")

    log_dev(L("[NF4-TE] NF4 layers: {} | BF16 layers: {} | non-Linear tensors: {} | aliases: {}",
              "[NF4-TE] Capas NF4: {} | Capas BF16: {} | Tensores no-Lineales: {} | aliases: {}")
            .format(len(quantized), len(unquantized), len(other_tensors), len(aliases)))

    DIAG["text_encoder"]["index_quantized"] = len(quantized)
    DIAG["text_encoder"]["index_unquantized"] = len(unquantized)
    DIAG["text_encoder"]["index_other"] = len(other_tensors)

    from transformers import AutoConfig
    import transformers as _tf

    te_config = AutoConfig.from_pretrained(os.path.dirname(config_path), trust_remote_code=True)

    # --- layer truncation, with a hard sanity check ---
    # --- truncado de capas, con comprobacion dura ---
    sub_cfg = getattr(te_config, "text_config", None)
    if sub_cfg is not None and hasattr(sub_cfg, "num_hidden_layers"):
        original_layers = int(sub_cfg.num_hidden_layers)
        target = sub_cfg
    else:
        original_layers = int(getattr(te_config, "num_hidden_layers", 0))
        target = te_config

    DIAG["text_encoder"]["original_num_hidden_layers"] = original_layers
    DIAG["text_encoder"]["requested_hidden_layer"] = TEXT_ENCODER_HIDDEN_LAYER

    if original_layers < TEXT_ENCODER_HIDDEN_LAYER:
        raise RuntimeError(
            L("Text encoder has only {} layers but layer {} was requested. Truncating UP would "
              "create uninitialized layers and ruin the conditioning.",
              "El text encoder solo tiene {} capas pero se pidio la capa {}. Truncar hacia ARRIBA "
              "crearia capas sin inicializar y arruinaria el condicionamiento.")
            .format(original_layers, TEXT_ENCODER_HIDDEN_LAYER))

    if full_model:
        log_dev(L("[NF4-TE] FULL model: {} layers kept, lm_head kept (text generation).",
                  "[NF4-TE] Modelo COMPLETO: {} capas y lm_head conservadas (generacion de texto).")
                .format(original_layers))
    else:
        target.num_hidden_layers = TEXT_ENCODER_HIDDEN_LAYER
        log_dev(L("[NF4-TE] Decoder truncated: {} -> {} layers (only hidden_states[{}] is consumed).",
                  "[NF4-TE] Decoder truncado: {} -> {} capas (solo se consume hidden_states[{}]).")
                .format(original_layers, TEXT_ENCODER_HIDDEN_LAYER, TEXT_ENCODER_HIDDEN_LAYER))

    architectures = getattr(te_config, "architectures", []) or []
    model_cls = None
    for arch in architectures:
        if hasattr(_tf, arch):
            model_cls = getattr(_tf, arch)
            break
    if model_cls is None:
        for candidate in ("Qwen3VLForConditionalGeneration", "AutoModelForCausalLM"):
            if hasattr(_tf, candidate):
                model_cls = getattr(_tf, candidate)
                break
    if model_cls is None:
        raise RuntimeError(L("Could not resolve the text_encoder class.",
                             "No se pudo resolver la clase del text_encoder."))

    log_dev(L("[NF4-TE] Detected class: {}", "[NF4-TE] Clase detectada: {}").format(model_cls.__name__))
    DIAG["text_encoder"]["class"] = model_cls.__name__

    def _strip_lm_head(te):
        """H3 uses hidden_states[50]; the LM head is dead weight (151936 x 5120).
        H3 usa hidden_states[50]; la cabeza LM es peso muerto (151936 x 5120).

        Left in place it gets zero-filled and pushed to the GPU — up to ~3.1 GB of VRAM
        holding nothing. nn.Identity keeps the wrapper's forward working either way.
        Si se deja, se rellena con ceros y se sube a la GPU: hasta ~3,1 GB de VRAM para
        nada. nn.Identity mantiene funcionando el forward del wrapper igualmente.
        """
        if full_model:
            # Con el modelo completo la cabeza es justo lo que hace falta.
            # With the full model the head is exactly what is needed.
            return 0
        head = getattr(te, "lm_head", None)
        if head is None or isinstance(head, nn.Identity):
            return 0
        try:
            n = int(getattr(head, "out_features", 0)) * int(getattr(head, "in_features", 0))
        except Exception:
            n = 0
        te.lm_head = nn.Identity()
        log_dev(L("[NF4-TE] lm_head replaced with Identity ({:.2f} GB of unused weights skipped).",
                  "[NF4-TE] lm_head sustituida por Identity ({:.2f} GB de pesos inutiles evitados).")
                .format(n * 4 / 1e9))
        DIAG["text_encoder"]["lm_head_stripped_params"] = n
        return n

    text_encoder = None
    try:
        with torch.device("meta"):
            text_encoder = model_cls(te_config)
        log_dev(L("[NF4-TE] Model created on the META device.",
                  "[NF4-TE] Modelo creado en el dispositivo META."))
    except Exception as e1:
        log_dev(L("[NF4-TE] META constructor failed: {}",
                  "[NF4-TE] El constructor en META fallo: {}").format(e1))
        text_encoder = model_cls(te_config)
        log_dev(L("[NF4-TE] Model created without META.",
                  "[NF4-TE] Modelo creado sin META."))

    # Drop the unused LM head BEFORE any loading, so it is never resolved, never zero-filled
    # and never moved to the GPU.
    # Quitar la cabeza LM no usada ANTES de cargar nada, para que no se resuelva, no se
    # rellene con ceros y no se suba a la GPU.
    _strip_lm_head(text_encoder)

    # ---- NF4 layers ----
    log_dev("")
    log_dev(L("[NF4-TE] Rebuilding {} NF4 layers...",
              "[NF4-TE] Reconstruyendo {} capas NF4...").format(len(quantized)))
    replaced, failed_nf4 = 0, []
    t0 = time.time()

    for name, info in quantized.items():
        filepath = os.path.join(weights_dir, info["file"])
        if not os.path.exists(filepath):
            failed_nf4.append((name, "missing file / falta el fichero"))
            continue
        try:
            parent, child_name = get_parent_module(text_encoder, name)
        except Exception as e:
            failed_nf4.append((name, "unresolved module / modulo no resuelto: {}".format(e)))
            continue
        try:
            with safe_open(filepath, framework="pt", device="cpu") as f:
                weight_data = f.get_tensor("weight")
                bias_data = f.get_tensor("bias") if info.get("bias", False) else None
                qs_dict = {k[len("quant_state."):]: f.get_tensor(k)
                           for k in f.keys() if k.startswith("quant_state.")}

            new_layer = new_layer_empty(
                Linear4bit,
                int(info["in_features"]), int(info["out_features"]),
                bias=info.get("bias", False),
                quant_type=info.get("quant_type", "nf4"),
                compute_dtype=torch.bfloat16,
            )
            try:
                new_weight = Params4bit.from_prequantized(
                    data=weight_data, quantized_stats=qs_dict,
                    requires_grad=False, device="cpu", module=new_layer)
            except TypeError:
                new_weight = Params4bit.from_prequantized(
                    data=weight_data, quantized_stats=qs_dict,
                    requires_grad=False, device="cpu")
            new_layer.weight = new_weight
            if bias_data is not None:
                new_layer.bias = nn.Parameter(bias_data.to(dtype=torch.bfloat16), requires_grad=False)
            set_child_module(parent, child_name, new_layer)
            replaced += 1
            if replaced % 100 == 0:
                log_dev(L("  [NF4-TE] ... {} / {} NF4 layers loaded on CPU",
                          "  [NF4-TE] ... {} / {} capas NF4 cargadas en CPU").format(replaced, len(quantized)))
        except Exception as e:
            failed_nf4.append((name, str(e)))

    log_dev(L("[NF4-TE] NF4 layers rebuilt: {} in {:.1f}s",
              "[NF4-TE] Capas NF4 reconstruidas: {} en {:.1f}s").format(replaced, time.time() - t0))

    DIAG["text_encoder"]["nf4_replaced"] = replaced
    DIAG["text_encoder"]["nf4_failed"] = len(failed_nf4)
    DIAG["text_encoder"]["nf4_failed_first"] = [list(x) for x in failed_nf4[:15]]

    # Layers named for a truncated-away index are expected to fail; anything else is not.
    # Las capas de indices truncados fallaran (esperado); cualquier otro fallo no lo es.
    def _is_truncated_layer(n):
        parts = n.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                return int(parts[i + 1]) >= TEXT_ENCODER_HIDDEN_LAYER
        return False

    hard_failures = [f for f in failed_nf4 if not _is_truncated_layer(f[0])]
    DIAG["text_encoder"]["nf4_hard_failures"] = len(hard_failures)
    DIAG["text_encoder"]["nf4_hard_failures_first"] = [list(x) for x in hard_failures[:15]]
    if hard_failures:
        msg = ("{} NF4 layers inside the kept 0..{} stack failed to load: {} / "
               "{} capas NF4 dentro del stack conservado 0..{} no se cargaron: {}"
               ).format(len(hard_failures), TEXT_ENCODER_HIDDEN_LAYER - 1, hard_failures[:5],
                        len(hard_failures), TEXT_ENCODER_HIDDEN_LAYER - 1, hard_failures[:5])
        if STRICT_LOAD:
            raise RuntimeError(msg)
        diag_warn(msg)

    # ---- BF16 layers ----
    if unquantized:
        log_dev(L("[NF4-TE] Loading {} BF16 layers on CPU...",
                  "[NF4-TE] Cargando {} capas BF16 en CPU...").format(len(unquantized)))
        for name, info in unquantized.items():
            filepath = os.path.join(weights_dir, info["file"])
            if not os.path.exists(filepath):
                continue
            try:
                parent, child_name = get_parent_module(text_encoder, name)
                with safe_open(filepath, framework="pt", device="cpu") as f:
                    weight = f.get_tensor("weight")
                    bias = f.get_tensor("bias") if info.get("bias", False) else None
                layer = new_layer_empty(nn.Linear,
                                        int(info["in_features"]), int(info["out_features"]),
                                        bias=info.get("bias", False))
                layer.weight = nn.Parameter(weight.to(dtype=torch.bfloat16), requires_grad=False)
                if bias is not None:
                    layer.bias = nn.Parameter(bias.to(dtype=torch.bfloat16), requires_grad=False)
                set_child_module(parent, child_name, layer)
            except Exception as e:
                diag_warn("BF16 layer failed / capa BF16 fallo {}: {}".format(name, e))

    # ---- non-Linear tensors (all kept on CPU) / tensores no-Lineales (todo en CPU) ----
    state_dict_other = {}
    if other_tensors:
        log_dev(L("[NF4-TE] Loading {} non-Linear tensors on CPU...",
                  "[NF4-TE] Cargando {} tensores no-Lineales en CPU...").format(len(other_tensors)))
        for tensor_name, tinfo in other_tensors.items():
            filepath = os.path.join(weights_dir, tinfo["file"])
            if not os.path.exists(filepath):
                continue
            try:
                with safe_open(filepath, framework="pt", device="cpu") as f:
                    if tensor_name in f.keys():
                        tensor = f.get_tensor(tensor_name)
                        if tensor.is_floating_point():
                            tensor = tensor.to(torch.bfloat16)
                        state_dict_other[tensor_name] = tensor
            except Exception as e:
                diag_warn("tensor failed / tensor fallo {}: {}".format(tensor_name, e))

    if not state_dict_other and os.path.isdir(weights_dir):
        other_files = sorted(f for f in os.listdir(weights_dir)
                             if f.lower().startswith("other-") and f.lower().endswith(".safetensors"))
        if other_files:
            log_dev(L("[NF4-TE] Index has no 'other'; reading {} other-*.safetensors files.",
                      "[NF4-TE] El indice no tiene 'other'; leyendo {} ficheros other-*.safetensors.")
                    .format(len(other_files)))
            for fname in other_files:
                try:
                    with safe_open(os.path.join(weights_dir, fname), framework="pt", device="cpu") as f:
                        for key in f.keys():
                            tensor = f.get_tensor(key)
                            if tensor.is_floating_point():
                                tensor = tensor.to(torch.bfloat16)
                            # FIX: stays on CPU (the old code moved these to CUDA only here).
                            # FIX: se queda en CPU (el codigo antiguo movia estos a CUDA solo aqui).
                            state_dict_other[key] = tensor
                except Exception as e:
                    diag_warn("could not read / no se pudo leer {}: {}".format(fname, e))

    if aliases:
        for alias_name, real_name in aliases.items():
            if real_name in state_dict_other:
                state_dict_other[alias_name] = state_dict_other[real_name]

    if state_dict_other:
        log_dev(L("[NF4-TE] Assigning {} non-Linear tensors...",
                  "[NF4-TE] Asignando {} tensores no-Lineales...").format(len(state_dict_other)))
        try:
            missing, unexpected = text_encoder.load_state_dict(state_dict_other, strict=False, assign=True)
            DIAG["text_encoder"]["other_unexpected"] = len(unexpected)
        except TypeError:
            text_encoder.load_state_dict(state_dict_other, strict=False)
        except Exception as e:
            diag_warn("load_state_dict failed / fallo: {}".format(e))

    # ---- ROBUST SWEEP: resolve anything still on meta by scanning the real files ----
    # ---- BARRIDO ROBUSTO: resolver lo que siga en meta escaneando los ficheros reales ----
    # The index-driven path above only works if its key names match the model layout exactly.
    # Qwen3VLForConditionalGeneration expects model.language_model.* / model.visual.*, while
    # single-file H3 checkpoints usually store model.* / visual.*. Rather than trust the names,
    # scan every safetensors file and match by (candidate name, exact shape).
    # La ruta basada en el indice solo funciona si sus nombres coinciden con el modelo.
    # Qwen3VLForConditionalGeneration espera model.language_model.* / model.visual.*, mientras
    # los checkpoints H3 de fichero unico suelen guardar model.* / visual.*. En vez de confiar
    # en los nombres, escaneamos todos los safetensors y emparejamos por (candidato, shape).
    log_dev("")
    log_dev("-" * 90)
    log_dev(L("[SCAN] Scanning weight files to resolve remaining meta tensors...",
              "[SCAN] Escaneando ficheros de pesos para resolver los tensores meta restantes..."))
    # The original (non-quantized) text_encoder folder is included as an extra source: if it is
    # present its standard HF shards carry full-path keys and everything resolves immediately.
    # La carpeta original (sin cuantizar) del text_encoder se incluye como fuente extra: si
    # existe, sus shards HF estandar llevan claves con ruta completa y todo se resuelve al vuelo.
    inv = build_weight_inventory(index, weights_dir, nf4_te_dir, orig_te_dir)
    resolve_meta_from_checkpoint(text_encoder, inv, compute_dtype=torch.bfloat16)

    # ---- export completeness verdict / veredicto de completitud del export ----
    _sections = {"quantized": len(quantized), "unquantized": len(unquantized),
                 "other": len(other_tensors)}
    _still_meta = [n for n, p in text_encoder.named_parameters()
                   if p.device.type == "meta" and not n.startswith(META_ALLOWED_PREFIXES)]
    DIAG["text_encoder"]["export_completeness"] = {
        "index_sections": _sections,
        "language_tensors_still_missing": len(_still_meta),
        "sample": _still_meta[:10],
        "verdict": "INCOMPLETE_EXPORT" if _still_meta else "COMPLETE",
    }
    if _still_meta:
        log_error("")
        log_error("*" * 90)
        log_error(L("[EXPORT] The NF4 export contains only Linear layers: {} language tensors "
                    "(embed_tokens + RMSNorms) were never written to disk.",
                    "[EXPORT] El export NF4 solo contiene capas lineales: {} tensores del modelo "
                    "de lenguaje (embed_tokens + RMSNorms) nunca se escribieron al disco.")
                  .format(len(_still_meta)))
        log_error(L("[EXPORT] Index sections -> quantized={} unquantized={} other={}",
                    "[EXPORT] Secciones del indice -> quantized={} unquantized={} other={}")
                  .format(_sections["quantized"], _sections["unquantized"], _sections["other"]))
        log_error(L("[EXPORT] Fix the NF4 conversion script to also export non-Linear tensors.",
                    "[EXPORT] Corrige el script de conversion NF4 para exportar tambien los "
                    "tensores no-Lineales."))
        log_error("*" * 90)

        if FETCH_MISSING_FROM_HUB:
            backfill_missing_from_hub(
                text_encoder, nf4_te_dir,
                repo_id=MISSING_TENSORS_REPO,
                subfolders=MISSING_TENSORS_SUBFOLDERS,
                # Con el modelo completo hay que rellenar TODAS las capas, no
                # solo hasta la 50: las normas de las capas 50-63 tambien hacen
                # falta para generar. Son unos cientos de KB.
                # With the full model every layer must be backfilled, not just up
                # to 50: the norms of layers 50-63 are needed to generate too.
                max_layer=(original_layers if full_model else TEXT_ENCODER_HIDDEN_LAYER),
            )
        else:
            diag_error(L("fetch_missing_from_hub is disabled, so the encoder cannot be completed.",
                         "fetch_missing_from_hub esta desactivado, el encoder no puede completarse."))
    log_dev("-" * 90)

    # ---- processor / tokenizer ----
    log_dev("")
    log_dev(L("[NF4-TE] Loading processor/tokenizer...",
              "[NF4-TE] Cargando processor/tokenizer..."))
    processor = None
    processor_candidates = [d for d in (orig_te_dir, nf4_te_dir) if os.path.isdir(d)]
    for cand in processor_candidates:
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(cand, trust_remote_code=True)
            log_dev(L("[NF4-TE] Processor loaded from: {}",
                      "[NF4-TE] Processor cargado desde: {}").format(cand))
            break
        except Exception:
            pass
    if processor is None:
        for cand in processor_candidates:
            try:
                from transformers import AutoTokenizer
                processor = AutoTokenizer.from_pretrained(cand, trust_remote_code=True)
                log_dev(L("[NF4-TE] Tokenizer loaded from: {}",
                          "[NF4-TE] Tokenizer cargado desde: {}").format(cand))
                break
            except Exception:
                pass
    if processor is None:
        raise RuntimeError(L("Could not load the text_encoder processor/tokenizer.",
                             "No se pudo cargar el processor/tokenizer del text_encoder."))

    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad_(False)

    log_dev(L("[NF4-TE] Text encoder NF4 loaded.",
              "[NF4-TE] Text encoder NF4 cargado."))
    return text_encoder, processor, te_config


def disable_final_norm(text_encoder):
    """H3 consumes the UNNORMALIZED layer-50 output -> replace the final norm with Identity.
    H3 consume la salida SIN NORMALIZAR de la capa 50 -> sustituir la norm final por Identity."""
    targets = [
        ("model.language_model", getattr(getattr(text_encoder, "model", None), "language_model", None)),
        ("model", getattr(text_encoder, "model", None)),
        ("root", text_encoder),
    ]
    for label, cand in targets:
        if cand is not None and isinstance(getattr(cand, "norm", None), nn.Module) \
                and not isinstance(cand.norm, nn.Identity):
            cand.norm = nn.Identity()
            DIAG["text_encoder"]["final_norm_disabled_at"] = label
            log_dev(L("[TEXT] Final norm disabled at '{}' -> RAW H3 conditioning.",
                      "[TEXT] Final norm desactivada en '{}' -> conditioning RAW de H3.").format(label))
            return True
    raise RuntimeError(
        L("Could not locate the text encoder final norm. H3 needs the UNNORMALIZED layer-50 "
          "output; running with the RMSNorm applied gives the wrong conditioning scale.",
          "No se pudo localizar la norm final del text encoder. H3 necesita la salida SIN "
          "NORMALIZAR de la capa 50; con la RMSNorm aplicada la escala del conditioning es erronea."))


# ============================================================================
# VAE LOADING / CARGA DE VAEs
# ============================================================================
def load_vaes(nf4_model_id):
    log_dev("")
    log_dev("=" * 90)
    log_dev(L("[VAE-H3] Loading the reference H3 VIDEO VAE",
              "[VAE-H3] Cargando el VIDEO VAE H3 de referencia"))
    log_dev("=" * 90)
    return load_h3_video_vae(nf4_model_id, MODEL_ID, strict=STRICT_LOAD), None


# ============================================================================
# PROMPT ENCODING (H3 = raw hidden_states[50] of Qwen3-VL)
# ============================================================================
def _te_forward(text_encoder, inputs):
    """Prefer the inner LM module (skips lm_head compute); fall back to the wrapper.
    Prefiere el modulo LM interno (evita calcular lm_head); si no, usa el wrapper."""
    inner = getattr(text_encoder, "model", None)
    if inner is not None and isinstance(inner, nn.Module):
        try:
            out = inner(**inputs, output_hidden_states=True, use_cache=False)
            if getattr(out, "hidden_states", None):
                return out, "inner"
        except Exception:
            pass
    out = text_encoder(**inputs, output_hidden_states=True, use_cache=False)
    return out, "wrapper"


def encode_prompt_minimax(text_encoder, processor, prompt, device="cuda", quiet=False,
                          input_ids=None):
    tokenizer = getattr(processor, "tokenizer", processor)
    if input_ids is not None:
        inputs = {"input_ids": input_ids,
                  "attention_mask": torch.ones(input_ids.shape, dtype=torch.long)}
    else:
        inputs = tokenizer(
            text=[prompt],
            add_special_tokens=False,   # H3: raw text, no special tokens, no chat template
            padding=False,
            return_tensors="pt",
            max_length=MAX_SEQ_LEN,
            truncation=True,
        )
    n_tok = int(inputs["input_ids"].shape[1])

    if n_tok == 0:
        pad_id = getattr(tokenizer, "pad_token_id", None) or 151643
        inputs["input_ids"] = torch.tensor([[pad_id]])
        inputs["attention_mask"] = torch.ones((1, 1), dtype=torch.long)
        n_tok = 1
        if not quiet:
            log_dev(L("    [TEXT] Empty prompt -> single pad token id={}",
                      "    [TEXT] Prompt vacio -> un solo pad token id={}").format(pad_id), level=2)

    truncated = bool(n_tok >= MAX_SEQ_LEN)
    if truncated and not quiet:
        log_error(L("[WARN] Caption truncated at max_seq_len={} tokens: {}...",
                    "[WARN] Caption truncado en max_seq_len={} tokens: {}...")
                  .format(MAX_SEQ_LEN, prompt[:60]))

    if "attention_mask" in inputs:
        real_attention_mask = inputs["attention_mask"][0].bool()
    else:
        real_attention_mask = torch.ones(n_tok, dtype=torch.bool)

    inputs = {k: v.to(device) for k, v in inputs.items()}

    t0 = time.time()
    with torch.inference_mode():
        outputs, path = _te_forward(text_encoder, inputs)
        hidden = outputs.hidden_states

        # hidden_states has num_hidden_layers + 1 entries. With the stack truncated to 50,
        # index 50 is the last entry, and since the final norm is Identity it IS the raw
        # layer-50 output that H3 expects.
        # hidden_states tiene num_hidden_layers + 1 entradas. Con el stack truncado a 50, el
        # indice 50 es la ultima entrada y, como la norm final es Identity, ES la salida cruda
        # de la capa 50 que espera H3.
        expected_len = TEXT_ENCODER_HIDDEN_LAYER + 1
        if len(hidden) != expected_len:
            raise RuntimeError(
                L("hidden_states has {} entries, expected {} (= layer {} + embedding). The layer "
                  "truncation did not take effect and hidden_states[{}] is NOT the layer-50 output.",
                  "hidden_states tiene {} entradas, se esperaban {} (= capa {} + embedding). El "
                  "truncado de capas no surtio efecto y hidden_states[{}] NO es la salida de la capa 50.")
                .format(len(hidden), expected_len, TEXT_ENCODER_HIDDEN_LAYER, TEXT_ENCODER_HIDDEN_LAYER))

        layer_idx = TEXT_ENCODER_HIDDEN_LAYER
        prompt_embeds = hidden[layer_idx]

    elapsed = time.time() - t0

    if not quiet:
        log_dev(L("    [TEXT] path={} | layer={} | shape={} | tokens={} | {:.2f}s",
                  "    [TEXT] ruta={} | capa={} | shape={} | tokens={} | {:.2f}s")
                .format(path, layer_idx, tuple(prompt_embeds.shape), n_tok, elapsed))

    if torch.isnan(prompt_embeds).any():
        diag_error("NaN in prompt_embeds / NaN en prompt_embeds: {}".format(prompt[:60]))

    return {
        "prompt_embeds": prompt_embeds.detach().to(torch.bfloat16).cpu().contiguous(),
        "attention_mask": real_attention_mask.cpu().contiguous(),
        "prompt": prompt,
        "hidden_layer_used": layer_idx,
        "num_tokens": n_tok,
        "truncated": truncated,
    }


# ============================================================================
# SELF-TEST: ROPE PERMUTATION / AUTO-TEST: PERMUTACION DE ROPE
# ============================================================================
def selftest_rope_permutation(text_encoder, processor):
    """Decisive functional test for a dead RoPE.
    Test funcional decisivo para detectar un RoPE muerto.

    With RoPE working, attention is position-aware, so shuffling the words changes the
    mean-pooled embedding. With inv_freq = 0, attention becomes permutation-EQUIVARIANT and
    the mean-pooled embedding is IDENTICAL for both orderings.

    Con RoPE funcionando, la atencion depende de la posicion, asi que barajar las palabras
    cambia el embedding promediado. Con inv_freq = 0, la atencion es EQUIVARIANTE a
    permutaciones y el embedding promediado es IDENTICO en ambos ordenes.
    """
    a = "the red cat sat quietly on a very blue mat near the old wooden door"

    tokenizer = getattr(processor, "tokenizer", processor)
    ids_a = tokenizer(text=[a], add_special_tokens=False, return_tensors="pt")["input_ids"]
    # Permute the TOKEN IDS, not the words: shuffling words changes the BPE merges and the
    # two token multisets stop matching, which makes the comparison meaningless.
    # Permutar los TOKEN IDS, no las palabras: barajar palabras cambia los merges BPE y los
    # dos multisets de tokens dejan de coincidir, lo que invalida la comparacion.
    ids_b = ids_a.flip(1).contiguous()

    ta, tb = ids_a[0].tolist(), ids_b[0].tolist()
    result = {"probe_a": a, "probe_b": "<same token ids, reversed order>",
              "tokens_a": len(ta), "tokens_b": len(tb),
              "same_token_multiset": sorted(ta) == sorted(tb)}

    ra = encode_prompt_minimax(text_encoder, processor, a, quiet=True, input_ids=ids_a)
    rb = encode_prompt_minimax(text_encoder, processor, a, quiet=True, input_ids=ids_b)

    ea = ra["prompt_embeds"].float().reshape(-1, ra["prompt_embeds"].shape[-1]).mean(0)
    eb = rb["prompt_embeds"].float().reshape(-1, rb["prompt_embeds"].shape[-1]).mean(0)

    cos = float(torch.nn.functional.cosine_similarity(ea, eb, dim=0))
    maxdiff = float((ea - eb).abs().max())
    rel = maxdiff / max(1e-8, float(ea.abs().max()))

    result.update({"cosine": cos, "max_abs_diff": maxdiff, "relative_diff": rel})

    # A dead RoPE gives cos ~= 1.0 and max_abs_diff ~= 0 when the token multisets match.
    # Un RoPE muerto da cos ~= 1.0 y max_abs_diff ~= 0 cuando los multisets coinciden.
    if result["same_token_multiset"]:
        result["verdict"] = "ROPE_DEAD" if (cos > 0.9995 and rel < 1e-3) else "ROPE_OK"
    else:
        result["verdict"] = "INCONCLUSIVE_TOKENIZATION"

    DIAG["selftests"]["rope_permutation"] = result

    log_dev("")
    log_dev("=" * 90)
    log_dev(L("[SELFTEST] RoPE permutation test", "[SELFTEST] Test de permutacion de RoPE"))
    log_dev(L("[SELFTEST] same token multiset: {} | cosine: {:.6f} | max|diff|: {:.6e} | rel: {:.3e}",
              "[SELFTEST] mismo multiset de tokens: {} | coseno: {:.6f} | max|dif|: {:.6e} | rel: {:.3e}")
            .format(result["same_token_multiset"], cos, maxdiff, rel))
    log_dev(L("[SELFTEST] VERDICT: {}", "[SELFTEST] VEREDICTO: {}").format(result["verdict"]))
    log_dev("=" * 90)

    if result["verdict"] == "ROPE_DEAD":
        diag_error(L("RoPE is NOT working: word order does not change the embedding. "
                     "The conditioning is a bag of words.",
                     "RoPE NO funciona: el orden de las palabras no cambia el embedding. "
                     "El conditioning es una bolsa de palabras."))
    return result


def selftest_prompt_discrimination(text_encoder, processor):
    """Two unrelated prompts must not produce near-identical embeddings.
    Dos prompts sin relacion no deben producir embeddings casi identicos."""
    p1 = "a close-up portrait photograph of a woman with green eyes"
    p2 = "an aerial view of an industrial harbour at night with cranes"
    r1 = encode_prompt_minimax(text_encoder, processor, p1, quiet=True)
    r2 = encode_prompt_minimax(text_encoder, processor, p2, quiet=True)
    e1 = r1["prompt_embeds"].float().reshape(-1, r1["prompt_embeds"].shape[-1]).mean(0)
    e2 = r2["prompt_embeds"].float().reshape(-1, r2["prompt_embeds"].shape[-1]).mean(0)
    cos = float(torch.nn.functional.cosine_similarity(e1, e2, dim=0))
    out = {"cosine": cos,
           "embed_norm_1": float(e1.norm()), "embed_norm_2": float(e2.norm()),
           "verdict": "SUSPICIOUS" if cos > 0.999 else "OK"}
    DIAG["selftests"]["prompt_discrimination"] = out
    log_dev(L("[SELFTEST] Prompt discrimination cosine: {:.6f} -> {}",
              "[SELFTEST] Coseno de discriminacion de prompts: {:.6f} -> {}").format(cos, out["verdict"]))
    if out["verdict"] == "SUSPICIOUS":
        diag_error(L("Unrelated prompts give nearly identical embeddings.",
                     "Prompts sin relacion dan embeddings casi identicos."))
    return out


# ============================================================================
# LATENT ENCODING / CODIFICACION DE LATENTES
# ============================================================================
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv", ".avi")
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a")


def is_video(filename):
    return filename.lower().endswith(VIDEO_EXTS)


def is_audio(filename):
    """Reservado: el audio aun no se cachea, pero la rama existe para que
    anadirlo sea una funcion mas y no rehacer el recorrido del dataset.
    Reserved: audio is not cached yet, but the branch exists so that adding it
    is one more function rather than reworking the dataset walk."""
    return filename.lower().endswith(AUDIO_EXTS)


def _ffbin(name):
    import shutil
    return shutil.which(name) or shutil.which(name + ".exe")


def probe_video(path):
    """(fotogramas, ancho, alto) contados de verdad, o None.

    Se cuentan con -count_frames en vez de leer nb_frames del contenedor, que en
    muchos mp4 viene vacio o mentiroso; aqui un recuento erroneo rompe la
    geometria 17n+5 sin avisar.
    Counted with -count_frames instead of the container's nb_frames, which is
    often missing or wrong in mp4; here a bad count silently breaks the 17n+5
    geometry."""
    probe = _ffbin("ffprobe")
    if not probe:
        return None
    try:
        out = subprocess.run(
            [probe, "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames,width,height",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=180,
        ).stdout.strip()
        w, h, frames = out.split(",")
        return int(frames), int(w), int(h)
    except Exception:
        return None


def read_video_frames(path, count, width, height):
    """Los `count` primeros fotogramas, ya escalados y recortados a width x height.

    Escala y recorte los hace ffmpeg y no PIL: son 73 fotogramas por clip y
    hacerlo en Python multiplica por diez el tiempo de la fase 2 sin ganar nada.
    `increase` mantiene la proporcion cubriendo el destino, y `crop` centra.

    The first `count` frames, already scaled and cropped. ffmpeg does the scale
    and crop rather than PIL: it is 73 frames per clip and doing it in Python
    makes phase 2 ten times slower for nothing.
    """
    ffmpeg = _ffbin("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(L("ffmpeg not found in PATH; it is needed to read video.",
                             "No se encuentra ffmpeg en el PATH; hace falta para leer video."))

    vf = "scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}".format(
        w=int(width), h=int(height))
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-vf", vf,
         "-frames:v", str(int(count)), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, timeout=600,
    )
    if proc.returncode != 0:
        detalle = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError("ffmpeg: {}".format(detalle[-1] if detalle else "exit {}".format(proc.returncode)))

    esperado = int(count) * int(height) * int(width) * 3
    if len(proc.stdout) != esperado:
        raise RuntimeError(L("ffmpeg returned {} bytes, {} expected ({} frames of {}x{})",
                             "ffmpeg devolvio {} bytes, se esperaban {} ({} fotogramas de {}x{})")
                           .format(len(proc.stdout), esperado, count, width, height))

    arr = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(int(count), int(height), int(width), 3)
    return torch.from_numpy(arr.copy())          # [T,H,W,3] uint8


def encode_video_latent(vae, image):
    """One still image = one video frame (T=1) / Una imagen fija = un frame de video (T=1)."""
    vae_dtype = next(vae.parameters()).dtype
    image_tensor = F_vision.pil_to_tensor(image).float() / 127.5 - 1.0
    image_tensor = image_tensor.unsqueeze(0).to("cuda", dtype=vae_dtype)
    with torch.inference_mode():
        latent = vae.encode(image_tensor)   # [B,24,1,H/16,W/16]
    return latent.detach().to(torch.bfloat16).cpu().contiguous()


def encode_clip_latent(vae, frames_uint8):
    """Clip [T,H,W,3] uint8 -> latente [1,24,T',H/16,W/16].

    Misma normalizacion que la imagen ([-1,1]); el VAE convierte a [0,1] e
    ImageNet por dentro. Se manda a CUDA de una vez: son 73x576x576x3 bytes,
    unos 73 MB en uint8, y trocear la copia no ahorraria nada porque el pico
    esta en las activaciones del encoder, que ya van trozo a trozo.

    Same normalization as the still image; the VAE converts to [0,1] and
    ImageNet internally. Sent to CUDA in one go: ~73 MB in uint8, and splitting
    the copy would save nothing because the peak is in the encoder activations,
    which are already chunked.
    """
    vae_dtype = next(vae.parameters()).dtype
    x = frames_uint8.permute(3, 0, 1, 2).unsqueeze(0)          # [1,3,T,H,W]
    x = x.to("cuda", dtype=torch.float32).div_(127.5).sub_(1.0).to(vae_dtype)
    with torch.inference_mode():
        latent = vae.encode(x)
    del x
    torch.cuda.empty_cache()
    return latent.detach().to(torch.bfloat16).cpu().contiguous()


def read_audio_channels(nf4_model_id, default=32):
    """Read the audio VAE latent dim (H3 = 32) / Lee la dim latente del VAE de audio (H3 = 32)."""
    candidates = [
        os.path.join("audio_vae", "config.json"),
        os.path.join("vae", "audio_config.json"),
        os.path.join("transformer", "config.json"),
        os.path.join("transformer", "config_nf4.json"),
        "config.json",
    ]
    for rel in candidates:
        p = os.path.join(nf4_model_id, rel)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    c = json.load(f)
                for key in ("audio_latents_dim", "audio_in_channels", "audio_channels", "latent_channels"):
                    if key in c:
                        val = c[key]
                        if isinstance(val, (list, tuple)):
                            val = val[0]
                        log_dev(L("[AUDIO] Detected {} = {} in {}",
                                  "[AUDIO] Detectado {} = {} en {}").format(key, int(val), rel))
                        return int(val)
            except Exception:
                pass
    log_dev(L("[AUDIO] audio latent dim not found; using default {} (H3 reference = 32).",
              "[AUDIO] no se detecto la dim latente de audio; usando {} por defecto (referencia H3 = 32)."
              ).format(default))
    return default


def make_audio_latent(audio_channels):
    """Zero placeholder. NOTE: for image-only training the reference trainers either omit the
    audio stream entirely or feed noised silence with ZERO loss weight.
    Placeholder de ceros. NOTA: para entrenamiento solo-imagen, los trainers de referencia o
    bien omiten el stream de audio, o bien meten silencio ruidoso con peso de loss CERO."""
    return torch.zeros((1, audio_channels, 1), dtype=torch.bfloat16)


# ============================================================================
# MAIN
# ============================================================================
def preprocess_minimaxh3():
    total_start = time.time()

    log_dev("")
    log_dev("=" * 90)
    log_dev(L(" MINIMAX-H3 PRE-CACHE v3  (H3 REFERENCE VAE + NF4 TEXT ENCODER + DIAGNOSTICS)",
              " PRE-CACHE MINIMAX-H3 v3  (VAE H3 DE REFERENCIA + TEXT ENCODER NF4 + DIAGNOSTICOS)"))
    log_dev("=" * 90)
    log_dev(L("Original model : {}", "Modelo original: {}").format(os.path.abspath(MODEL_ID)))
    log_dev(L("NF4 model      : {}", "Modelo NF4     : {}").format(os.path.abspath(NF4_MODEL_ID)))
    # Sin L(): las dos mitades eran identicas, asi que _Bi las unia con " / " y la
    # ruta salia impresa DOS veces en la misma linea. Una etiqueta que se escribe
    # igual en ingles y en espanol no necesita duplicarse.
    # No L() here: both halves were identical, so _Bi joined them with " / " and the
    # path was printed TWICE on the same line.
    log_dev("Dataset        : {}".format(os.path.abspath(DATASET_PATH)))
    log_dev("Cache          : {}".format(os.path.abspath(CACHE_DIR)))
    log_dev(L("Target area    : {} | Multiple: {} | Max side: {}",
              "Area objetivo  : {} | Multiplo: {} | Lado max: {}").format(TARGET_AREA, MULTIPLE, MAX_SIDE))
    log_dev(L("Max seq len    : {} | Hidden layer: {}",
              "Max seq len    : {} | Capa oculta : {}").format(MAX_SEQ_LEN, TEXT_ENCODER_HIDDEN_LAYER))
    log_dev(L("Strict load    : {} | Require captions: {} | Self-tests: {}",
              "Carga estricta : {} | Exigir captions: {} | Auto-tests: {}")
            .format(STRICT_LOAD, REQUIRE_CAPTIONS, RUN_SELFTESTS))
    log_dev(L("Force GPU      : {} | CPU fallback: {} | Log level: {}",
              "Forzar GPU     : {} | Respaldo CPU: {} | Nivel de log: {}")
            .format(FORCE_GPU, ALLOW_CPU_FALLBACK, LOGS_DEV))

    if SYSTEM_RAM_GB is not None:
        log_dev(L("System RAM     : {:.1f} GB | Low RAM mode: {}",
                  "RAM del sistema: {:.1f} GB | Modo Low RAM: {}").format(SYSTEM_RAM_GB, LOW_RAM_MODE))

    if not torch.cuda.is_available():
        raise RuntimeError(L("CUDA is not available.", "CUDA no esta disponible."))

    import transformers as _tfv
    DIAG["env"] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": getattr(_tfv, "__version__", "?"),
        "gpu": torch.cuda.get_device_name(0),
        "system_ram_gb": SYSTEM_RAM_GB,
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
    DIAG["config"] = {
        "model_id": MODEL_ID, "nf4_model_id": NF4_MODEL_ID, "dataset_path": DATASET_PATH,
        "cache_dir": CACHE_DIR, "target_area": TARGET_AREA, "multiple": MULTIPLE,
        "max_side": MAX_SIDE, "max_seq_len": MAX_SEQ_LEN, "num_frames_config": NUM_FRAMES,
        "frame_rate": FRAME_RATE, "trigger_word": TRIGGER_WORD,
        "text_encoder_hidden_layer": TEXT_ENCODER_HIDDEN_LAYER,
        "low_ram_mode": LOW_RAM_MODE, "strict_load": STRICT_LOAD,
        "require_captions": REQUIRE_CAPTIONS, "write_audio_latent": WRITE_AUDIO_LATENT,
    }
    log_dev("GPU            : {} | CUDA {}".format(DIAG["env"]["gpu"], DIAG["env"]["cuda"]))
    log_dev("transformers   : {} | torch {}".format(
        DIAG["env"]["transformers"], DIAG["env"]["torch"]))
    log_dev("=" * 90)

    ensure_nf4_model_exists(NF4_MODEL_ID, repo_id="AcademiaSD/MiniMax-H3-NF4")

    os.makedirs(DATASET_PATH, exist_ok=True)

    # Stale-cache guard: caches written before format version 3 came from a text encoder whose
    # embeddings/norms/RoPE were zero-filled. Refuse to mix them with new data.
    # Guardia de cache obsoleta: las caches anteriores a la version de formato 3 salieron de un
    # text encoder con embeddings/normas/RoPE a ceros. Nos negamos a mezclarlas con datos nuevos.
    _cache_info_path = os.path.join(CACHE_DIR, "cache_info.json")
    if os.path.isfile(_cache_info_path):
        try:
            with open(_cache_info_path, "r", encoding="utf-8") as f:
                _old_version = int((json.load(f) or {}).get("version", 0))
        except Exception:
            _old_version = 0
        if _old_version < CACHE_FORMAT_VERSION:
            raise RuntimeError(
                L("{} holds a version-{} cache. Versions below {} were produced with a broken "
                  "text encoder (zero embeddings / dead RoPE). Delete the folder and re-run.",
                  "{} contiene una cache de version {}. Las versiones anteriores a la {} se "
                  "generaron con un text encoder roto (embeddings a cero / RoPE muerto). "
                  "Borra la carpeta y vuelve a ejecutar.")
                .format(os.path.abspath(CACHE_DIR), _old_version, CACHE_FORMAT_VERSION))

    os.makedirs(CACHE_DIR, exist_ok=True)

    images = sorted(f for f in os.listdir(DATASET_PATH)
                    if f.lower().endswith(IMAGE_EXTS + VIDEO_EXTS))
    if not images:
        raise RuntimeError(L("No images in {}", "No hay imagenes en {}").format(DATASET_PATH))

    # ---- caption coverage check / comprobacion de cobertura de captions ----
    missing_caption, empty_caption = [], []
    for filename in images:
        base = os.path.splitext(filename)[0]
        prompt, had = read_prompt(base)
        if not had:
            missing_caption.append(filename)
        elif not prompt.strip():
            empty_caption.append(filename)

    DIAG["dataset"] = {
        "num_images": len(images),
        "missing_caption": len(missing_caption),
        "missing_caption_first": missing_caption[:15],
        "empty_caption": len(empty_caption),
        "empty_caption_first": empty_caption[:15],
    }

    log_dev("")
    log_dev(L("[DATASET] Images: {} | without .txt: {} | empty caption: {}",
              "[DATASET] Imagenes: {} | sin .txt: {} | caption vacio: {}")
            .format(len(images), len(missing_caption), len(empty_caption)))

    if missing_caption or empty_caption:
        msg = ("{} images have no caption file and {} have an empty caption. Training on empty "
               "prompts is the most common cause of zero likeness. First missing: {} / "
               "{} imagenes no tienen fichero de caption y {} lo tienen vacio. Entrenar con "
               "prompts vacios es la causa mas comun de parecido nulo. Primeras sin caption: {}"
               ).format(len(missing_caption), len(empty_caption), missing_caption[:8],
                        len(missing_caption), len(empty_caption), missing_caption[:8])
        if REQUIRE_CAPTIONS:
            raise RuntimeError(msg)
        diag_warn(msg)

    audio_channels = read_audio_channels(NF4_MODEL_ID, AUDIO_LATENT_CHANNELS)
    DIAG["config"]["audio_latent_channels"] = audio_channels

    # ==================================================================
    # PHASE 1 / FASE 1: TEXT ENCODER -> EMBEDDINGS
    # ==================================================================
    log_dev("")
    log_dev("=" * 90)
    log_dev(L(" PHASE 1/2: NF4 TEXT ENCODER -> EMBEDDINGS",
              " FASE 1/2: TEXT ENCODER NF4 -> EMBEDDINGS"))
    log_dev("=" * 90)

    text_encoder, processor, te_config = load_text_encoder_from_nf4(NF4_MODEL_ID, MODEL_ID)

    # ---------- THE DIAGNOSTIC / EL DIAGNOSTICO ----------
    report_rope_state(text_encoder, "before_fix")

    # Pass the ROOT config: the helpers descend into .text_config / .vision_config themselves,
    # so each rotary module resolves against the right sub-config.
    # Pasar el config RAIZ: los helpers bajan solos a .text_config / .vision_config, asi cada
    # modulo rotary se resuelve contra el sub-config que le corresponde.
    fallback_cfg = te_config
    rebuilt = rebuild_rope_buffers(text_encoder, fallback_config=fallback_cfg)

    report_rope_state(text_encoder, "after_fix")

    if rebuilt:
        log_error("")
        log_error("*" * 90)
        log_error(L("[ROPE] {} rotary buffer(s) were UNINITIALIZED and have been rebuilt.",
                    "[ROPE] {} buffer(s) rotary estaban SIN INICIALIZAR y se han reconstruido.")
                  .format(len(rebuilt)))
        log_error(L("[ROPE] Any cache produced by a previous run is INVALID - retrain from scratch.",
                    "[ROPE] Cualquier cache de una ejecucion anterior es INVALIDA - reentrenar desde cero."))
        log_error("*" * 90)
        log_error("")
    else:
        log_dev(L("[ROPE] All rotary buffers were already valid.",
                  "[ROPE] Todos los buffers rotary ya eran validos."))

    disable_final_norm(text_encoder)

    # Verify the two tensor groups that silently killed the encoder when zero-filled.
    # Verificar los dos grupos de tensores que mataban el encoder en silencio al ir a cero.
    verify_language_weights(text_encoder)

    force_cuda_or_die(text_encoder, "text_encoder", strict_meta=True)

    try:
        from bitsandbytes.nn import Linear4bit
        total_l4 = verified = 0
        for _, module in text_encoder.named_modules():
            if isinstance(module, Linear4bit):
                total_l4 += 1
                if getattr(module.weight, "bnb_quantized", False) \
                        and getattr(module.weight, "quant_state", None) is not None:
                    verified += 1
        DIAG["text_encoder"]["linear4bit_total"] = total_l4
        DIAG["text_encoder"]["linear4bit_verified"] = verified
        log_dev(L("[NF4-TE] Linear4bit total: {} | verified: {}",
                  "[NF4-TE] Linear4bit totales: {} | verificadas: {}").format(total_l4, verified))
        if total_l4 and verified != total_l4:
            diag_warn("Some Linear4bit layers are not properly quantized / "
                      "Algunas capas Linear4bit no estan bien cuantizadas: {}/{}".format(verified, total_l4))
    except Exception as e:
        diag_warn("Linear4bit check failed / fallo la comprobacion: {}".format(e))

    torch.cuda.reset_peak_memory_stats()

    # ---------- SELF-TESTS ----------
    if RUN_SELFTESTS:
        selftest_rope_permutation(text_encoder, processor)
        selftest_prompt_discrimination(text_encoder, processor)

    # ---------- negative prompt ----------
    log_dev("")
    log_dev(L("[1.1] Encoding negative/empty prompt...",
              "[1.1] Codificando prompt negativo/vacio..."))
    with torch.inference_mode():
        neg_result = encode_prompt_minimax(text_encoder, processor, "", "cuda")
    save_prompt_result(neg_result, "_neg")
    del neg_result
    gc_cuda()

    # ---------- custom preview prompt ----------
    if PREVIEW_CUSTOM_PROMPT:
        custom_prompt = PREVIEW_CUSTOM_PROMPT
        if TRIGGER_WORD and TRIGGER_WORD.lower() not in custom_prompt.lower():
            custom_prompt = "{}, {}".format(TRIGGER_WORD, custom_prompt).strip(", ")
        log_dev(L("[1.2] Encoding custom prompt: {}",
                  "[1.2] Codificando prompt custom: {}").format(custom_prompt))
        with torch.inference_mode():
            custom_result = encode_prompt_minimax(text_encoder, processor, custom_prompt, "cuda")
        save_prompt_result(custom_result, "_custom")
        del custom_result
        gc_cuda()

    # ---------- per-image embeddings ----------
    log_dev("")
    log_dev(L("[1.3] Generating embeddings for {} images...",
              "[1.3] Generando embeddings para {} imagenes...").format(len(images)))

    embed_stats = []
    truncated_count = 0

    for idx, filename in enumerate(images, start=1):
        base = os.path.splitext(filename)[0]
        struct_path = os.path.join(CACHE_DIR, "{}_prompt_structure.json".format(base))
        prompt, _ = read_prompt(base)

        if os.path.exists(struct_path):
            log_dev(L("  [{}/{}] SKIP embeddings {}",
                      "  [{}/{}] OMITIR embeddings {}").format(idx, len(images), filename))
            continue

        log_dev("")
        log_dev("  [{}/{}] {}".format(idx, len(images), filename))
        log_dev("    Prompt: {}".format(
            prompt[:110] + ("..." if len(prompt) > 110 else "")))

        with torch.inference_mode():
            prompt_result = encode_prompt_minimax(text_encoder, processor, prompt, "cuda")

        if prompt_result["truncated"]:
            truncated_count += 1

        pe = prompt_result["prompt_embeds"].float()
        embed_stats.append({
            "file": filename,
            "tokens": prompt_result["num_tokens"],
            "mean": float(pe.mean()),
            "std": float(pe.std()),
            "absmax": float(pe.abs().max()),
        })

        structure = save_prompt_result(prompt_result, "{}_prompt".format(base))

        atomic_json({"filename": filename, "prompt": prompt, "prompt_structure": structure},
                    os.path.join(CACHE_DIR, "{}_info_partial.json".format(base)))

        log_dev(L("    Saved. tokens={} shape={}",
                  "    Guardado. tokens={} shape={}")
                .format(prompt_result["num_tokens"], tuple(prompt_result["prompt_embeds"].shape)))

        del prompt_result, pe
        gc_cuda()

    DIAG["text_encoder"]["truncated_captions"] = truncated_count
    DIAG["text_encoder"]["embed_stats_sample"] = embed_stats[:10]
    if embed_stats:
        DIAG["text_encoder"]["embed_std_mean"] = sum(e["std"] for e in embed_stats) / len(embed_stats)

    log_dev("")
    log_dev(L("[OK] PHASE 1 complete.", "[OK] FASE 1 completada."))

    # ---------- release the text encoder ----------
    log_dev(L("[MEM] Releasing text_encoder...", "[MEM] Liberando text_encoder..."))
    text_encoder = None
    processor = None
    gc_cuda()
    log_dev(L("[MEM] VRAM after release: {:.2f} GB",
              "[MEM] VRAM tras liberar: {:.2f} GB").format(vram_gb()))

    # ==================================================================
    # PHASE 2 / FASE 2: VAE -> LATENTS
    # ==================================================================
    log_dev("")
    log_dev("=" * 90)
    log_dev(L(" PHASE 2/2: VIDEO VAE -> LATENTS", " FASE 2/2: VIDEO VAE -> LATENTES"))
    log_dev("=" * 90)

    vae, _ = load_vaes(NF4_MODEL_ID)
    force_cuda_or_die(vae, "video VAE", strict_meta=True)
    torch.cuda.reset_peak_memory_stats()

    # Running accumulators for the latent statistics self-test.
    # Acumuladores para el auto-test de estadisticas del latente.
    chan_sum = torch.zeros(24, dtype=torch.float64)
    chan_sqsum = torch.zeros(24, dtype=torch.float64)
    chan_count = 0
    latent_shapes = []

    for idx, filename in enumerate(images, start=1):
        base = os.path.splitext(filename)[0]
        video_path = os.path.join(CACHE_DIR, "{}_video_latent.pt".format(base))
        audio_path = os.path.join(CACHE_DIR, "{}_audio_latent.pt".format(base))

        need_audio = WRITE_AUDIO_LATENT and not os.path.exists(audio_path)
        if os.path.exists(video_path) and not need_audio:
            log_dev(L("  [{}/{}] SKIP latents {}",
                      "  [{}/{}] OMITIR latentes {}").format(idx, len(images), filename))
            continue

        log_dev("")
        log_dev("  [{}/{}] {}".format(idx, len(images), filename))
        t0 = time.time()

        ruta_media = os.path.join(DATASET_PATH, filename)

        if is_video(filename):
            info = probe_video(ruta_media)
            if info is None:
                raise RuntimeError(L("Could not read the clip: {}",
                                     "No se pudo leer el clip: {}").format(filename))
            total, vw, vh = info
            keep = h3_valid_frames(total, NUM_FRAMES)
            if keep is None:
                raise RuntimeError(
                    L("{} has {} frames; the minimum is {}.",
                      "{} tiene {} fotogramas; el minimo son {}.")
                    .format(filename, total, H3_BASE_FRAMES))

            bw, bh = bucket_size(vw, vh)
            log_dev("    Bucket: {}x{} | {} de {} fotogramas -> {} latentes".format(
                bw, bh, keep, total, h3_latent_frames(keep)))
            if keep < total:
                log_dev(L("    Trimming {} -> {} frames (H3 needs 17n+5).",
                          "    Recortando {} -> {} fotogramas (H3 exige 17n+5).")
                        .format(total, keep))

            frames_uint8 = read_video_frames(ruta_media, keep, bw, bh)
            video_latent = encode_clip_latent(vae, frames_uint8)
            del frames_uint8
        else:
            image = Image.open(ruta_media).convert("RGB")
            bw, bh = bucket_size(image.width, image.height)
            scale = max(bw / image.width, bh / image.height)
            image = image.resize((math.ceil(image.width * scale), math.ceil(image.height * scale)),
                                 Image.LANCZOS)
            left = (image.width - bw) // 2
            top = (image.height - bh) // 2
            image = image.crop((left, top, left + bw, top + bh))

            log_dev("    Bucket: {}x{}".format(bw, bh))

            with torch.inference_mode():
                video_latent = encode_video_latent(vae, image)

        torch.save(video_latent, video_path)

        lat = video_latent.float()
        latent_shapes.append(list(video_latent.shape))
        # [B,24,1,h,w] -> per-channel accumulation / acumulacion por canal
        flat = lat.reshape(lat.shape[0], lat.shape[1], -1)
        chan_sum += flat.sum(dim=(0, 2)).double()
        chan_sqsum += (flat ** 2).sum(dim=(0, 2)).double()
        chan_count += flat.shape[0] * flat.shape[2]

        log_dev(L("    Video latent: {}  mean={:.4f} std={:.4f}   [expected mean~0 std~1]",
                  "    Latente video: {}  media={:.4f} std={:.4f}   [esperado media~0 std~1]")
                .format(tuple(video_latent.shape), float(lat.mean()), float(lat.std())))

        if WRITE_AUDIO_LATENT:
            audio_latent = make_audio_latent(audio_channels)
            torch.save(audio_latent, audio_path)
            del audio_latent

        prompt, _ = read_prompt(base)
        prompt_structure = None
        struct_path = os.path.join(CACHE_DIR, "{}_prompt_structure.json".format(base))
        if os.path.exists(struct_path):
            with open(struct_path, "r", encoding="utf-8") as f:
                prompt_structure = json.load(f)

        atomic_json(
            {
                "filename": filename,
                "width": bw,
                "height": bh,
                # FIX: the encoded latent has exactly ONE frame. The old value (17) desynced
                # any geometry the trainer derives from it.
                # FIX: el latente codificado tiene exactamente UN frame. El valor antiguo (17)
                # desincronizaba cualquier geometria que el trainer derive de el.
                "num_frames": 1,
                "num_frames_pixels": 1,
                "num_frames_config": NUM_FRAMES,
                "frame_rate": FRAME_RATE,
                "prompt": prompt,
                "video_latent": os.path.basename(video_path),
                "audio_latent": os.path.basename(audio_path) if WRITE_AUDIO_LATENT else None,
                "latent_shape": list(video_latent.shape),
                # Fotogramas de PIXEL que se cachearon: 1 en una imagen, 17n+5 en
                # un clip. El entrenador lo necesita para calcular las filas de
                # audio, que dependen de la duracion y no del numero de latentes.
                # PIXEL frames cached: 1 for an image, 17n+5 for a clip. The
                # trainer needs it to size the audio rows, which depend on the
                # duration and not on the latent count.
                "num_frames": int(video_latent.shape[2] and
                                  (h3_pixel_frames(video_latent.shape[2])
                                   if video_latent.shape[2] > 1 else 1)),
                "kind": "video" if is_video(filename) else "image",
                "prompt_structure": prompt_structure,
            },
            os.path.join(CACHE_DIR, "{}_info.json".format(base)),
        )

        partial = os.path.join(CACHE_DIR, "{}_info_partial.json".format(base))
        if os.path.exists(partial):
            try:
                os.remove(partial)
            except Exception:
                pass

        log_dev(L("    Latents time: {:.2f}s | peak VRAM: {:.2f} GB",
                  "    Tiempo latentes: {:.2f}s | VRAM pico: {:.2f} GB")
                .format(time.time() - t0, vram_peak_gb()))

        del video_latent, lat, flat
        gc_cuda()

    # ---------- latent statistics self-test ----------
    if chan_count > 0:
        mean = (chan_sum / chan_count)
        var = (chan_sqsum / chan_count) - mean ** 2
        std = var.clamp_min(0).sqrt()
        stats = {
            "num_values_per_channel": int(chan_count),
            "per_channel_mean": [round(float(v), 4) for v in mean.tolist()],
            "per_channel_std": [round(float(v), 4) for v in std.tolist()],
            "global_mean": round(float(mean.mean()), 4),
            "global_std": round(float(std.mean()), 4),
            "latent_shapes_sample": latent_shapes[:5],
        }
        # H3-normalized latents should be roughly N(0,1) per channel.
        # Los latentes normalizados H3 deberian ser aproximadamente N(0,1) por canal.
        ok = (abs(stats["global_mean"]) < 0.5) and (0.5 < stats["global_std"] < 2.0)
        stats["verdict"] = "OK" if ok else "SUSPICIOUS"
        DIAG["selftests"]["latent_statistics"] = stats

        log_dev("")
        log_dev("=" * 90)
        log_dev(L("[SELFTEST] Latent statistics (expected mean~0, std~1 per channel)",
                  "[SELFTEST] Estadisticas del latente (esperado media~0, std~1 por canal)"))
        log_dev(L("[SELFTEST] global mean={:.4f}  global std={:.4f}  -> {}",
                  "[SELFTEST] media global={:.4f}  std global={:.4f}  -> {}")
                .format(stats["global_mean"], stats["global_std"], stats["verdict"]))
        log_dev("=" * 90)
        if not ok:
            diag_error(L("Latent statistics are off. Suspect the VAE weights or latents_mean/std.",
                         "Las estadisticas del latente estan mal. Sospechar de los pesos del VAE "
                         "o de latents_mean/std."))

    # ---------- cache info ----------
    # Que hay realmente en la cache. Se cuenta en vez de suponerlo: un dataset
    # puede ser mixto, y el entrenador necesita saber si va a encontrar latentes
    # de 1 fotograma, de 17n+5, o de ambos.
    # What the cache actually holds, counted rather than assumed: a dataset can
    # be mixed, and the trainer needs to know whether it will find 1-frame
    # latents, 17n+5 ones, or both.
    _n_clips = sum(1 for f in images if is_video(f))
    _n_imgs = len(images) - _n_clips
    if _n_clips and _n_imgs:
        _kind, _frames = "mixed", None
    elif _n_clips:
        _kind, _frames = "video", NUM_FRAMES
    else:
        _kind, _frames = "image", 1

    cache_info = {
        "format": "MiniMaxH3-LoRA-Precache",
        "version": CACHE_FORMAT_VERSION,
        "model_id": MODEL_ID,
        "nf4_model_id": NF4_MODEL_ID,
        "dataset_path": DATASET_PATH,
        "cache_dir": CACHE_DIR,
        "target_area": TARGET_AREA,
        "multiple": MULTIPLE,
        "frame_rate": FRAME_RATE,
        # num_frames = fotogramas de PIXEL por muestra. None en un dataset mixto:
        # ahi no hay un solo valor y cada _info.json lleva el suyo.
        # num_frames = PIXEL frames per sample. None on a mixed dataset, where
        # there is no single value and each _info.json carries its own.
        "num_frames": _frames,
        "content": _kind,
        "num_images": _n_imgs,
        "num_clips": _n_clips,
        "max_sequence_length": MAX_SEQ_LEN,
        "trigger_word": TRIGGER_WORD,
        "audio_latent_channels": audio_channels,
        "text_encoder_hidden_layer": TEXT_ENCODER_HIDDEN_LAYER,
        "prompt_encoding": "Qwen3-VL raw layer-50 output, NO final norm, no special tokens",
        "rope_rebuilt_modules": DIAG["rope"].get("rebuilt_modules", []),
        "note": ("Transformer NOT loaded. H3 reference Video VAE: 24 ch, H/16 x W/16. "
                 "Stills encode to T=1; clips encode in 17-frame chunks with 3 trailing "
                 "latents dropped, so 17n+5 pixel frames give 5n+2 latent frames. Audio "
                 "latent is a zero placeholder - the trainer MUST give it zero loss weight "
                 "or skip the audio stream entirely."),
    }
    atomic_json(cache_info, os.path.join(CACHE_DIR, "cache_info.json"))

    del vae
    gc_cuda()

    DIAG["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    DIAG["elapsed_minutes"] = round((time.time() - total_start) / 60.0, 2)
    DIAG["peak_vram_gb"] = round(vram_peak_gb(), 2)
    diag_path = os.path.join(CACHE_DIR, "_diagnostics.json")
    atomic_json(DIAG, diag_path)

    log_dev("")
    log_dev("=" * 90)
    log_dev(L(" MINIMAX-H3 PRE-CACHE COMPLETE", " PRE-CACHE MINIMAX-H3 COMPLETADO"))
    log_dev("=" * 90)
    log_dev("Cache          : {}".format(os.path.abspath(CACHE_DIR)))
    log_dev(L("Images         : {}", "Imagenes       : {}").format(len(images)))
    log_dev(L("Peak VRAM      : {:.2f} GB", "VRAM pico      : {:.2f} GB").format(vram_peak_gb()))
    log_dev(L("Total time     : {:.2f} min", "Tiempo total   : {:.2f} min")
            .format(DIAG["elapsed_minutes"]))
    log_dev(L("Warnings: {} | Errors: {}", "Avisos: {} | Errores: {}")
            .format(len(DIAG["warnings"]), len(DIAG["errors"])))
    log_dev("")
    log_dev("*" * 90)
    log_dev(L(">>> SEND THIS FILE BACK: {}", ">>> DEVUELVE ESTE FICHERO: {}").format(diag_path))
    log_dev("*" * 90)
    log_dev("")


if __name__ == "__main__":
    try:
        preprocess_minimaxh3()
    except Exception:
        log_error("")
        log_error("=" * 90)
        log_error(L("ERROR IN MINIMAX-H3 PRE-CACHE", "ERROR EN EL PRE-CACHE MINIMAX-H3"))
        log_error("=" * 90)
        traceback.print_exc()
        DIAG["errors"].append(traceback.format_exc())
        try:
            DIAG["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            os.makedirs(CACHE_DIR, exist_ok=True)
            p = os.path.join(CACHE_DIR, "_diagnostics.json")
            atomic_json(DIAG, p)
            log_error(L(">>> Partial diagnostics saved: {}",
                        ">>> Diagnostico parcial guardado: {}").format(p))
        except Exception:
            pass
        raise
