# -*- coding: utf-8 -*-
"""
2_train_lora_MiniMaxH3.py

Versión v5: NF4 CPU + presupuesto VRAM manual + single-swap seguro con checkpoint/bitsandbytes.
CORRECCIÓN: Las capas "no-convert" se cargan en CUDA primero, se dequantizan,
y luego se mueven a CPU como bf16.

Estrategia:
- Todo el transformer se carga inicialmente en CPU (NF4 empaquetado).
- Las capas "no-convert" se cargan en CUDA, se dequantizan a bf16, y se mueven a CPU.
- Luego se mueven a GPU:
  * módulos imprescindibles fuera de bloques
  * tantos bloques como quepan dentro de vram_budget_gb
- Se reserva vram_swap_gb para cargar temporalmente bloques desde CPU.
- Los bloques que no residen en GPU se ejecutan con block-swap JIT:
  se mueven a GPU justo antes de forward/backward y se devuelven a CPU.
"""

import atexit
import os
import shutil
import platform



os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("DIFFUSERS_NO_ADVISORY_WARNINGS", "1")

if platform.system() != "Windows":
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,garbage_collection_threshold:0.8",
    )
else:
    # PyTorch >= 2.6 soporta expandable_segments en Windows, y el propio mensaje de
    # OOM de torch lo recomienda explicitamente. Con block swap el pool se fragmenta
    # muchisimo, asi que aqui importa mas que en Linux. La variable se lee en la
    # primera asignacion CUDA; si el build no lo soporta, torch avisa y sigue.
    # Para desactivarlo, exporta PYTORCH_CUDA_ALLOC_CONF antes de lanzar el script.
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,garbage_collection_threshold:0.8",
    )

import gc
import math
import time
import random
import json
import signal
import sys
import inspect
import logging
import traceback
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit, Params4bit
from safetensors import safe_open
from safetensors.torch import save_file, load_file

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    from torch.autograd.graph import save_on_cpu as _save_on_cpu_ctx
    _SAVE_ON_CPU_AVAILABLE = True
except Exception:
    _save_on_cpu_ctx = None
    _SAVE_ON_CPU_AVAILABLE = False

ACTIVATION_OFFLOAD_ACTIVE = False

import psutil

# torch.cuda.OutOfMemoryError solo existe en PyTorch >= 2.0; en versiones
# anteriores un OOM real de CUDA llega como RuntimeError normal.
_CUDA_OOM_EXCEPTION_TYPES = tuple(
    t for t in (getattr(torch.cuda, "OutOfMemoryError", None),) if t is not None
) or (RuntimeError,)


def _is_cuda_oom_error(exc):
    if isinstance(exc, _CUDA_OOM_EXCEPTION_TYPES):
        return True
    return "out of memory" in str(exc).lower()


def ram_stats(label=""):
    """Monitorear RAM del sistema (no solo CUDA)"""
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / 1e9
    log_print(f"[RAM] {label} | RSS: {ram_gb:.2f} GB", flush=True)
# =============================================================================
# CONFIG
# =============================================================================

DEFAULTS = {
    "nf4_cache_dir": "./MiniMax-H3-NF4",
    "cache_dir": "./cached_data_minimaxh3",
    "output_dir": "./minimaxh3_lora_output",
    # --- Presupuesto de pasos -----------------------------------------------
    # total_steps y warmup_steps se cuentan en MICRO-PASOS (1 muestra cada uno).
    # El optimizador solo actualiza cada grad_accum_steps, asi que lo que de
    # verdad entrena son total_steps/grad_accum_steps ACTUALIZACIONES.
    # Antes: 800/4 = 200 actualizaciones -> un LoRA de identidad se queda a medio
    # cocer con eso (de ahi el parecido bajo). Ahora: 2000/2 = 1000.
    # total_steps/warmup_steps are MICRO-steps; real training = total/accum updates.
    # Before: 800/4 = 200 updates, far too few for an identity LoRA.
    # A 20 s/it el tiempo de pared lo fija total_steps, NO grad_accum. Con
    # grad_accum=2 la mitad de los forwards no producen ninguna actualizacion:
    # pagas el doble de segundos por update. Con batch 1 y Adam, acumular no
    # compra casi nada aqui; lo que compra parecido es el NUMERO de updates.
    # grad_accum=1 -> cada micro-paso es un update: 2x updates por el mismo tiempo.
    # At 20 s/it wall time is set by total_steps, not grad_accum. accum=1 doubles
    # the number of updates for the same number of seconds.
    "total_steps": 600,
    "batch_size": 1,
    "grad_accum_steps": 1,

    "lr": 2e-4,

    # "flat" = LR constante tras el warmup. "cosine" = decaimiento a min_lr_ratio.
    # El coseno a 0.1 solo deja pasar un 55% del movimiento total que permitiria un
    # LR plano, y el LoRA de referencia se mueve 5,7x mas que el nuestro. El preset
    # que funciona en otro trainer de esta familia es explicitamente "flat 2e-4 sin
    # warmup". Para un LoRA de identidad que arranca en cero, decaer el LR antes de
    # haber llegado a ningun sitio es tirar la mitad del presupuesto.
    # "flat" keeps LR constant after warmup. Cosine-to-0.1 only delivers 55% of the
    # total movement a flat schedule would, and our LoRA moves 5.7x less than the
    # reference one.
    "lr_schedule": "flat",
    "min_lr_ratio": 0.1,
    # Warmup en ACTUALIZACIONES del optimizador (ver lr_at()).
    "warmup_steps": 50,
    # rank 8 es poco para clavar una identidad en un DiT de 33B: el detalle facial
    # necesita subespacio. 16 es el punto dulce; 32 si sobra VRAM.
    # rank 8 is thin for identity in a 33B DiT; 16 is the sweet spot, 32 if VRAM allows.
    "lora_rank": 16,
    "lora_alpha": 16,
    "weight_decay": 0.0,

    # --- Optimizador --------------------------------------------------------
    # "adamw"     -> torch.optim.AdamW, estado en fp32 (RECOMENDADO en H3).
    # "adamw8bit" -> bnb.optim.PagedAdamW8bit, estado cuantizado a 8 bits.
    #
    # En H3 (33B, 50 bloques) los gradientes por parametro del LoRA son muy
    # pequenos. AdamW8bit cuantiza exp_avg/exp_avg_sq por bloques, y ese error
    # se come justo las componentes de baja magnitud: el LoRA acierta pose,
    # pelo y encuadre pero la CARA se queda blanda. Con AdamW en fp32 el
    # sintoma desaparece. Cuesta ~6 bytes/parametro extra de VRAM.
    # On H3 the LoRA per-parameter gradients are tiny; AdamW8bit's quantized
    # optimizer state destroys the low-magnitude components, which is exactly
    # where facial detail lives. fp32 AdamW fixes it for ~6 extra bytes/param.
    "optimizer_type": "adamw",
    # Perfil de gradiente por bloque cada N pasos (0 = off). Mide si los bloques
    # tempranos reciben gradiente consistente o solo ruido.
    "grad_profile_every": 100,

    # Cada N pasos, evalua el MISMO batch con y sin adaptador y reporta si el LoRA
    # mejora la prediccion. 0 = off. Es la medida que distingue "aprende poco" de
    # "no aprende nada del contenido".
    # Cada 10 pasos cuesta un forward extra el 10% del tiempo. A 50 la senal
    # sigue siendo perfectamente legible y el coste baja al 2%.
    # Every 10 steps costs an extra forward 10% of the time; 50 keeps the signal
    # readable at a 2% cost.
    "overfit_probe_every": 50,

    "max_grad_norm": 1.0,

    # --- Previsualizacion de progreso / progress preview ---------------------
    # Cada N pasos genera <OUTPUT_DIR>/preview_step_N.png con el prompt de la
    # PRIMERA imagen del dataset, para ver como avanza el parecido. 0 = APAGADO
    # (ni se carga el VAE ni se pierde un segundo). Ver el bloque
    # "PREVISUALIZACION DE PROGRESO" para el detalle de como funciona.
    # Every N steps renders <OUTPUT_DIR>/preview_step_N.png using the FIRST
    # dataset caption. 0 = OFF (nothing is loaded, no time is lost).
    "preview_every": 0,
    # De donde sale el prompt de cada preview:
    #   "first"  -> siempre el caption de la primera imagen. Comparable entre
    #               pasos: es el modo para ver progreso.
    #   "random" -> un caption cualquiera del dataset en cada preview. Da
    #               previews variadas y enseña como responde el LoRA a distintas
    #               descripciones, pero dos previews seguidas no son comparables.
    #   "rotate" -> recorre los captions en orden, uno por preview. Variedad
    #               como "random" pero con cobertura garantizada del dataset.
    #   "custom" -> un prompt libre. OJO: el entrenador NO tiene text encoder
    #               (esa es toda la gracia del pre-cache), asi que el prompt hay
    #               que codificarlo ANTES, en el script 1: se escribe en
    #               preview_custom_prompt, se relanza el pre-cache (que salta las
    #               imagenes ya cacheadas y solo re-codifica el prompt, ~2 min) y
    #               deja _custom_structure.json en la cache. Si no esta, se avisa
    #               y se cae a "first".
    # Where each preview's prompt comes from. "custom" needs the prompt encoded
    # by script 1 beforehand: the trainer has no text encoder by design.
    "preview_caption_mode": "first",
    # Solo informativo aqui: lo que se usa es el embedding cacheado. Se guarda
    # para que el JSON del proyecto quede completo y el script 1 lo lea.
    # Informational here: the cached embedding is what gets used.
    "preview_custom_prompt": "",
    # Pasos del sampler. MiniMax-H3 no esta destilado en pasos, asi que por
    # debajo de ~10 la imagen sale sin definir; 12-20 es el rango util para una
    # preview. Cada paso es un forward completo del DiT.
    # The model is not step-distilled: 12-20 is the useful range for a preview.
    "preview_steps": 20,
    # CFG. El checkpoint es guidance-distilled: NO tiene rama incondicional y la
    # receta oficial no usa CFG. <= 1.0 lo desactiva (recomendado). Por encima
    # de 1 se usa el prompt vacio cacheado (_neg) y CUESTA EL DOBLE de forwards.
    # The checkpoint is guidance-distilled; <= 1.0 disables CFG (recommended).
    "preview_cfg": 1.0,
    # Desplazamiento sigma del sampler. MiniMax-H3 solo tiene UN sampler
    # (MiniMaxH3Scheduler: Euler de flow rectificado, eta=0), asi que este es el
    # unico mando real: 12.0 es el del muestreador de video oficial, valores mas
    # bajos concentran los pasos en ruido medio-bajo.
    # H3 ships exactly ONE sampler, so this is the only real knob. 12.0 is the
    # official video sampler's shift.
    "preview_shift": 6.0,
    # Semilla de la preview. <= 0 = usa `seed`. Fijarla hace que todas las
    # previews sean el MISMO ruido inicial, que es lo que permite comparar
    # pasos entre si en vez de mirar imagenes sin relacion.
    # <= 0 = use `seed`. A fixed seed is what makes previews comparable.
    "preview_seed": -1,
    # "cpu" (por defecto) o "cuda". El decoder del VAE son ~2,4 G parametros:
    # 4,8 GB en bf16 en VRAM, o ~9,7 GB de RAM en fp32. En una tarjeta de 16 GB
    # con el DiT residente NO cabe en VRAM, y forzarlo mata el entrenamiento con
    # OOM. En CPU tarda ~1-2 min y no toca ni un byte de VRAM.
    # "cpu" (default) or "cuda". The VAE decoder needs 4.8 GB of VRAM in bf16;
    # it does not fit next to the DiT on a 16 GB card. CPU takes ~1-2 min.
    "preview_vae_device": "cpu",
    # El decoder expande cada frame latente a 4 frames de pixel. Cual de los 4
    # es "la imagen" no esta documentado para un frame suelto; 0 es el que usa
    # decode_cached_latent.py. Si las previews salen raras, prueba 3.
    # The decoder expands each latent frame into 4 pixel frames; 0 is what
    # decode_cached_latent.py uses. Try 3 if previews look wrong.
    "preview_frame_index": 0,
    # --- Fotogramas de la preview ------------------------------------------
    # 1 = modo imagen: UN solo frame latente. Es lo que se entrena y lo mas
    # barato, pero es un regimen que el modelo base no ha visto nunca al
    # GENERAR: H3 es un modelo de video y siempre produce secuencias, donde el
    # detalle fino queda sujeto a la coherencia temporal. La sospecha, aun sin
    # confirmar, es que por eso las previews salen blandas mientras el mismo
    # latente REAL del dataset decodifica nitido.
    #
    # >1 genera un clip corto y guarda un fotograma. El valor se ajusta a la
    # rejilla 17n+5 del VAE, que da 5n+2 frames latentes:
    #    5 frames  -> 2 frames latentes  (~2x el coste, prueba rapida)
    #   22 frames  -> 7 frames latentes  (~8x el coste, regimen nativo)
    # Es un EXPERIMENTO: si a 22 la imagen sale nitida, el problema es generar
    # un fotograma aislado y habra que decidir si compensa el coste.
    #
    # 1 = image mode, one latent frame: cheapest and what training uses, but a
    # regime the base model never sees when GENERATING. >1 renders a short clip
    # and saves one frame, snapped to the VAE's 17n+5 grid (5n+2 latent frames).
    # An experiment: if 22 comes out sharp, isolated single-frame generation is
    # the problem.
    "preview_num_frames": 1,
    # Decodifica el latente REAL de la primera imagen del dataset al arrancar y
    # lo guarda como preview_step_0.png. Sirve de doble control: valida que toda
    # la ruta del decoder funciona ANTES de gastar horas, y deja una referencia
    # visual contra la que comparar el progreso.
    # Decodes the first image's GROUND-TRUTH latent at startup as
    # preview_step_0.png: validates the decoder path before wasting hours and
    # leaves a visual reference to compare progress against.

    "save_every": 100,
    "seed": 314156,
    "frame_rate": 24.0,
    "project_name": "",
    "trigger_word": "",
    "max_text_tokens": 100,

    # Guarda un SEGUNDO .safetensors con las keys tal cual las produce PEFT (nombres
    # diffusers, sin renombrar, sin fusionar QKV y sin el swap SwiGLU de mlp.fc1).
    #
    # Existia para una pregunta concreta: si el fichero en crudo SI hacia efecto en
    # el inferenciador y el convertido no, el bug estaba en la conversion. La
    # respuesta resulto ser que SI: a la conversion le faltaba el intercambio de
    # mitades SwiGLU en mlp.fc1. Arreglado eso y verificada la traduccion de keys
    # contra diffusers, esta red de seguridad ya no hace falta.
    #
    # En ComfyUI el fichero en crudo NO hace nada: sus keys (transformer_blocks,
    # to_q, ff.net.0.proj) no existen en el checkpoint que carga, asi que el
    # cargador las ignora en silencio. Solo sirve para un pipeline diffusers+PEFT.
    # Cuesta 173 MB por cada guardado.
    #
    # Ponlo en true solo si vuelves a sospechar de la conversion de keys.
    #
    # Saves a SECOND file with the raw PEFT/diffusers key names. It existed to tell
    # a training bug apart from a key-conversion bug; the conversion turned out to
    # be the bug (the missing SwiGLU swap on mlp.fc1) and is now verified, so the
    # net is no longer needed. In ComfyUI the raw file does nothing: its key names
    # do not exist in the checkpoint it loads. Costs 173 MB per save.
    "lora_save_raw_copy": False,

    # --- Caption dropout ------------------------------------------------------
    # Con probabilidad p el paso se entrena SIN conditioning de texto. Obliga a que
    # la identidad viva en los pesos y no solo en la correlacion con la frase exacta
    # del caption; sin esto el LoRA puede aprender "cuando veas ESTE caption, pinta
    # esto" y desmoronarse con cualquier otro prompt. En esta familia de modelos otro
    # trainer lo tenia fijo en 0.05 y reporta que hace trabajo real.
    # 0.0 = off | 0.05 = por defecto | 0.10 = mas agresivo (multi-sujeto)
    # With probability p the step trains WITHOUT text conditioning, forcing identity
    # into the weights instead of into caption correlation.
    "caption_dropout": 0.05,
    "lora_only_attn": False,

    # --- Precision de los pesos ENTRENABLES (LoRA) ----------------------------
    # ESTE ERA EL FALLO GORDO. Los pesos LoRA se casteaban a bf16 y torch.optim.AdamW
    # crea exp_avg/exp_avg_sq con torch.zeros_like(p): si el parametro es bf16, el
    # ESTADO DEL OPTIMIZADOR TAMBIEN ES bf16. bf16 tiene 8 bits de mantisa (~0,4% de
    # resolucion relativa), asi que:
    #   - exp_avg_sq (que son cuadrados de gradientes ~1e-6) pierde casi toda la
    #     informacion y el denominador de Adam sale con escalones,
    #   - cada update de lr*~1e-4 sobre un peso de ~1e-2 cae por debajo del ULP y
    #     se REDONDEA A CERO en cuanto el gradiente no es grande.
    # Resultado exacto: el LoRA aprende lo grueso (pose, encuadre, color) y no
    # termina de fijar la cara. Es el mismo sintoma que los comentarios de abajo
    # atribuian a adamw8bit... pero pasaba igual con "adamw" por culpa del bf16.
    # fp32 aqui = master weights fp32 + estado Adam fp32; el computo sigue en bf16
    # por autocast, asi que el coste es solo memoria (~4x sobre los pesos LoRA).
    #
    # THIS WAS THE MAIN BUG. LoRA weights were cast to bf16, and torch.optim.AdamW
    # allocates its state with zeros_like(p) -> bf16 optimizer state. bf16 has ~0.4%
    # relative resolution, so exp_avg_sq (squared grads ~1e-6) is destroyed and small
    # updates round to zero. The LoRA learns coarse structure and never locks the face.
    "lora_dtype": "fp32",

    # CORRECCION: esto estuvo en True y fue un error mio. El LoRA de referencia que
    # funciona en ComfyUI (Lain_MiniMax.safetensors) SI incluye
    # diffusion_model.token_refiner.blocks.0 y .1 con las cuatro familias
    # (attn.qkv_proj, attn.out_proj, mlp.fc1, mlp.fc2). O sea que el cargador de
    # ComfyUI SI aplica esas keys, al contrario de lo que supuse. Excluirlas quitaba
    # 16 keys que el LoRA de referencia entrena.
    # CORRECTION: this was True and it was my mistake. The reference LoRA that works
    # in ComfyUI DOES include token_refiner.blocks.0/.1, so ComfyUI does apply those
    # keys, contrary to what I assumed.
    "lora_exclude_refiner": False,

    # "shuffle_epoch": permutacion barajada por epoca -> todas las imagenes se ven
    # el mismo numero de veces. "random": random.choice con reemplazo (lo de antes),
    # que con datasets pequenos deja imagenes vistas 3x y otras 1x.
    "dataset_sampler": "shuffle_epoch",

    # torch acumula los matmul bf16 en bf16 por defecto. Ponerlo en False obliga a
    # acumular en fp32: gradientes bastante mas limpios por ~2-4% de velocidad.
    # False forces fp32 accumulation for bf16 matmuls: cleaner grads, ~2-4% slower.
    "bf16_reduced_precision_reduction": False,

    "cast_frozen_bf16": True,
    "use_audio_loss": False,
    "lora_key_prefix": "diffusion_model.",
    "low_vram_12gb": True,
    "activation_offload": False,
    # Techo de RAM del proceso, en GB. 0 = sin limite (comportamiento de
    # siempre: todos los bloques aparcados van a RAM). Con un valor > 0, en
    # cuanto el proceso alcanza ese techo los bloques restantes se aparcan en
    # un fichero mapeado en memoria en vez de en RAM.
    # Process RAM ceiling, in GB. 0 = no limit (usual behaviour: every parked
    # block goes to RAM). With a value > 0, once the process reaches that
    # ceiling the remaining blocks are parked in a memory-mapped file instead.
    # Reutilizar la copia CPU de los bloques swapeados en vez de rehacerla con
    # una copia D2H cada vez. Es una mejora grande (medido 16,58 -> 9,61 s/it) y
    # es segura porque los pesos NF4 estan congelados. Ponlo a False solo para
    # descartar que sea la causa de algun problema.
    # Reuse the CPU copy of swapped blocks instead of remaking it with a D2H copy
    # every time. Big win (measured 16.58 -> 9.61 s/it) and safe because the NF4
    # weights are frozen. Set to False only to rule it out as a cause of trouble.
    # SageAttention: atencion cuantizada a int8/fp8. Menos memoria y mas rapida
    # que la nativa, a cambio de cambiar la NUMERICA de la atencion. En
    # inferencia es gratis; entrenando no lo es, asi que viene apagada.
    # SageAttention: int8/fp8 quantized attention. Less memory and faster than
    # native, in exchange for changing the attention NUMERICS. Free in inference;
    # not free when training, so it ships off.
    # Bloques residentes forzados. 0 = automatico (lo calcula el plan de VRAM).
    #
    # El plan solo sabe contar PESOS: base + N*bloque + overhead. Ese overhead de
    # 2,5 GB se calibro con secuencias de imagen de 300-700 tokens, donde las
    # activaciones son pequenas. Con un clip de 3.400 tokens las activaciones
    # dominan y el plan se pasa de largo: pide 30 residentes y satura la tarjeta.
    #
    # Modelar las activaciones de video en condiciones exigiria medir mucho, y la
    # atencion escala O(n^2), asi que una formula lineal mas seria mentir con mas
    # decimales. Mejor un numero a mano.
    #
    # Forced resident block count. 0 = automatic. The plan can only count WEIGHTS
    # (base + N*block + overhead), and that 2.5 GB overhead was calibrated on
    # 300-700 token image sequences where activations are small. On a 3,400 token
    # clip the activations dominate and the plan overshoots. Modelling video
    # activations properly would take a lot of measuring, and attention is O(n^2),
    # so another linear formula would just be a more precise lie.
    "resident_blocks": 0,

    "use_sage_attention": False,

    "nf4_cpu_home": True,
    # Donde viven los bloques que no caben en VRAM:
    #   "ram"   siempre en RAM (mas rapido si hay RAM de sobra)
    #   "disk"  siempre en un fichero mapeado (para maquinas con poca RAM)
    #   "auto"  RAM, y solo pasan a disco los que no quepan bajo ram_limit_gb
    # Where the blocks that do not fit in VRAM live:
    #   "ram"   always in RAM (fastest when RAM is plentiful)
    #   "disk"  always in a mapped file (for machines short on RAM)
    #   "auto"  RAM, spilling to disk only what does not fit under ram_limit_gb
    "park_mode": "auto",
    "ram_limit_gb": 0.0,
    "park_disk_dir": "",
    "loss_chunk_elements": 125000,
    # Rank 16 + pesos LoRA fp32 + estado de AdamW fp32 ocupan ~1,5 GB mas que
    # rank 8 en bf16. Se le quitan al presupuesto residente para no acabar en OOM.
    # Si te sobra VRAM, subelo otra vez a 16.0 (ira mas rapido: menos block swap).
    # Si te falta, baja a 13.0 antes de tocar el rank.
    # rank 16 + fp32 LoRA + fp32 Adam state costs ~1.5 GB more than rank 8 in bf16.
    "vram_budget_gb": 14.0,
    # MINIMO REAL: 1.34. Medido, no estimado.
    #
    # El tope no lo pone la tarjeta, lo pone el guard _enforce_manual_swap_budget,
    # que exige `_nf4_swap_required_bytes = get_block_nf4_bytes * 4`. Un bloque
    # NF4 de este modelo mide EXACTAMENTE 333.204.880 bytes (0,333205 GB), asi
    # que el guard pide 4 x 0,333205 = 1,3328 GB. Con 1.33 lanza RuntimeError;
    # con 1.34 pasa. Comprobado a mano: 1.34 es el valor mas bajo que arranca.
    #
    # De donde sale el 4x: hay que cubrir el bloque residente MAS el peso bf16
    # que bitsandbytes materializa entero en cada matmul, porque cae en
    # `_dequant_linear_fallback` en vez de usar un kernel fusionado. El peor
    # tensor del bloque NO es el proj del FF (294 MiB, el del traceback) sino
    # adaln_proj.linear [96768, 2688], que son 496 MiB. Pico real por bloque:
    # 0,333 residente + 0,520 del dequant = 0,853 GB. El resto del margen hasta
    # 1,333 absorbe la fragmentacion del allocator, que con el swap es real.
    #
    # No bajar de 1.34 sin cambiar tambien el factor 4 de
    # _nf4_swap_required_bytes, y sin medirlo: entre 0,853 y 1,333 es terreno
    # sin explorar, y equivocarse ahi es un OOM a mitad de una corrida de horas.
    #
    # REAL MINIMUM: 1.34, measured. The limit is the _enforce_manual_swap_budget
    # guard, not the card: it requires block_nf4_bytes * 4, and one NF4 block is
    # exactly 333,204,880 bytes, so the guard asks for 1.3328 GB. 1.33 raises,
    # 1.34 passes. The 4x covers the resident block plus the full bf16 weight
    # bitsandbytes materializes per matmul; the worst tensor is adaln_proj.linear
    # at 496 MiB, not the FF proj at 294 MiB. Real peak 0.853 GB; the rest is
    # allocator-fragmentation margin. Do not go below 1.34 without also changing
    # the factor, and without measuring.
    "vram_swap_gb": 1.34,
    # El headroom tiene que cubrir TODO lo que no son pesos residentes: activaciones,
    # el grafo de autograd, los gradientes, el estado de AdamW, los pesos bf16 que
    # bitsandbytes materializa al hacer backward de cada Linear4bit, y los workspaces
    # de cuBLAS/SDPA. 0.5 GB (y no digamos 0.1) es fisicamente imposible: por eso el
    # consumo real se iba siempre muy por encima del presupuesto.
    # Headroom must cover activations, autograd graph, grads, AdamW state, the bf16
    # weights bnb materializes during backward, and cuBLAS/SDPA workspaces.
    "vram_headroom_gb": 0.1,

    # --- Contabilidad REAL de VRAM -------------------------------------------
    # vram_budget_gb solo contabiliza PESOS RESIDENTES medidos con memory_allocated()
    # en el momento del plan. Todo lo demas quedaba fuera de la cuenta. Esto reserva
    # explicitamente el coste de entrenar (gradientes + estado de AdamW + activaciones
    # + picos de dequantizacion) ANTES de decidir cuantos bloques caben.
    # Sube este numero si sigues viendo picos por encima del presupuesto.
    # vram_budget_gb only accounted for RESIDENT WEIGHTS. This reserves the training
    # cost (grads + AdamW state + activations + dequant peaks) before block placement.
    # 83 M parametros entrenables en fp32 = 0,33 GB de pesos + 0,33 de gradientes
    # + 0,66 de estado de AdamW = 1,33 GB fijos, mas activaciones y workspaces.
    "vram_training_overhead_gb": 2.5,

    # TOPE DURO. Le dice a PyTorch que no puede pasar de
    # (budget + swap + headroom + overhead) y que si lo intenta debe lanzar OOM.
    # Sin esto NADA impedia al proceso comerse la tarjeta entera, y "simular una GPU
    # de 8/12 GB" era imposible por construccion: el plan colocaba pocos bloques y
    # luego el entrenamiento usaba toda la VRAM disponible igualmente.
    # HARD CAP via set_per_process_memory_fraction. Without it nothing stopped the
    # process from using the whole card, so simulating a smaller GPU never worked.
    "vram_hard_cap_enabled": True,

    # empty_cache() cada N pasos. El block swap pide y suelta los mismos bloques
    # cientos de veces por paso; el caching allocator NO devuelve esa memoria al
    # driver, asi que `reserved` (lo que ve nvidia-smi) crece hasta llenar la
    # tarjeta aunque `alloc` sea 5 GB. En tu log: Alloc 5.48 / Reserved 15.98.
    # 1 = cada paso (a 20 s/it el coste es despreciable). 0 = nunca.
    # The block swap churns the allocator; reserved grows to fill the card even when
    # alloc stays low. Your log: Alloc 5.48 / Reserved 15.98.
    "vram_empty_cache_every": 1,
    "nf4_load_chunk_layers": 100,
    "cpu_offload_reserve_gb": 2.0,
    "cpu_offload_blocks_enabled": True,
    "explicit_checkpointing_enabled": False,

    # --- CAUSA RAIZ DEL DESBORDE DE VRAM -------------------------------------
    # bitsandbytes NO usa save_for_backward para el peso cuantizado: hace
    # `ctx.tensors = (None, B)` en MatMul4Bit.forward, o sea un atributo normal del
    # ctx. torch.utils.checkpoint(use_reentrant=False) descarta los tensores
    # guardados interceptando save_for_backward... que bnb no llama. Resultado: CADA
    # Linear4bit ejecutado deja su peso NF4 de CUDA CLAVADO en el grafo hasta el
    # backward. Mover `module.weight` a CPU reasigna el atributo pero NO libera esa
    # memoria, porque el grafo sigue apuntando al tensor viejo. Por eso el swap
    # "devolvia" los 50 bloques a CPU y la VRAM subia igual hasta los 16 GB: al
    # final del forward tenias el modelo entero (~16 GB en NF4) vivo en VRAM.
    #
    # use_reentrant=True ejecuta el forward original bajo torch.no_grad(), asi que
    # bnb no crea ctx ni clava nada. Solo durante el backward, al recomputar un
    # bloque, se clava ESE bloque, y se suelta al terminar su backward. Es la
    # unica forma de que el block swap funcione con Linear4bit.
    #
    # bnb stashes the quantized weight as `ctx.tensors`, not via save_for_backward,
    # so non-reentrant checkpointing does NOT discard it: every executed Linear4bit
    # pins its CUDA NF4 weight in the graph until backward. Moving module.weight to
    # CPU rebinds the attribute but frees nothing. Reentrant checkpointing runs the
    # forward under no_grad, so nothing is pinned.
    "checkpoint_use_reentrant": True,

    # PyTorch >= 2.6 ya soporta expandable_segments en Windows y el propio mensaje de
    # OOM lo recomienda. Reduce muchisimo la fragmentacion del pool con block swap.
    # Ponlo en false si tu build de torch se queja.
    "expandable_segments_windows": True,
    "cpu_offload_dtype": "fp32",
    "cpu_offload_threads": 8,
    "cpu_offload_cache_kwargs": True,
    "offload_debug_sync": False,
    "audio_cpu_offload_enabled": True,
    "checkpoint_gpu_blocks_only": True,
    "lora_skip_first_n_blocks": 0,
    "timestep_scale_multiplier": 1.0,

    "flow_target_sign": -1,
    "debug_training": True,
    # None (default) = auto logit-normal + resolution-shift schedule, recomendado para
    # LoRAs de IMAGEN (concentra el muestreo en la zona de sigma donde vive el detalle
    # de identidad). Un float explícito activa el mapeo uniforme-u + shift legacy.
    # sigma_shift=1 (lo que habia) = muestreo UNIFORME de sigma en (0,1): solo un 10%
    # del entrenamiento cae por debajo de sigma=0.1, y ademas se gasta un 50% de los
    # pasos por encima de 0.5, donde solo se aprende composicion. Para PARECIDO hay
    # que concentrar en la banda media-baja, que es donde el modelo decide rasgos.
    # None = logit-normal (densidad de SD3/Flux), controlada por logit_normal_mu/std.
    # sigma_shift=1 was UNIFORM sigma sampling. None = logit-normal (SD3/Flux density).
    "sigma_shift": None,
    # mu<0 desplaza la densidad hacia sigma BAJA (mas detalle fino / identidad).
    # mu=0 -> mediana 0.50 | mu=-0.4 -> mediana 0.40 | mu=-0.8 -> mediana 0.31
    # -0.4 es el compromiso: sigue viendo ruido alto (composicion) pero dedica
    # bastante mas presupuesto a la banda donde vive la cara.
    # CORRECCION: puse -0.4 razonando que sigma baja = detalle facial. ERROR. La
    # implementacion de referencia que SI consigue parecido usa mu=0 + shift de
    # resolucion (mediana 0.62), y su comentario documenta que un schedule de sigma
    # equivocado da "structurally poor likeness at every checkpoint" en run real.
    # A sigma baja el target esta dominado por el ruido, que es inestimable: son
    # pasos que aportan gradiente casi puro ruido: medido, el coseno pred/target
    # cae a ~0.74 en sigma 0.05, el peor punto de toda la curva.
    # CORRECTION: -0.4 was my error. The reference implementation that DOES get
    # likeness uses mu=0 + resolution shift (median 0.62).
    "logit_normal_mu": 0.0,
    "logit_normal_std": 1.0,
    # CORRECCION: lo desactive por el mismo razonamiento erroneo. Es parte de la
    # receta validada: shift = exp(mu), mu lineal en el numero de tokens
    # (256 tokens -> 0.5, 6400 -> 1.15). A tus 247 tokens da shift 1.647.
    # CORRECTION: this is part of the validated recipe, not something to disable.
    "sigma_resolution_shift": True,
    # Aborta si algún peso entrenado o buffer del DiT sigue en META tras la carga NF4.
    # Rellenarlos en silencio (ceros / unos) es lo que produce un entrenamiento que
    # corre sin errores y no aprende nada.
    # --- Convención del timestep que consume el transformer -------------------
    # "one_minus_sigma": t = (1 - sigma) * escala   (convenio ComfyUI/ai-toolkit, t=1 limpio)
    # "sigma":           t = sigma * escala          (convenio diffusers, t crece con el ruido)
    # La escala es timestep_scale_multiplier: 1.0 para [0,1], 1000.0 para [0,1000].
    # Si el coseno pred/target del paso 1 es ~0, esta es la primera sospechosa.
    "timestep_convention": "one_minus_sigma",
    # Primeros N pasos en los que se imprime la linea [FLOW-CONV] con el coseno
    # pred/target. Es lo que queda del banco de sondas PROBE0..PROBE8 (eliminado):
    # una sola linea por paso que responde a la misma pregunta que costaba ~900
    # lineas y varios forwards extra. Un coseno claramente POSITIVO en el paso 1
    # significa que la interfaz (timestep, sigma, target, posiciones) esta bien.
    # First N steps that print the [FLOW-CONV] line with the pred/target cosine.
    # This is all that remains of the PROBE0..PROBE8 bench (removed): one line per
    # step answering the same question that used to cost ~900 lines and several
    # extra forwards. A clearly POSITIVE cosine on step 1 means the interface
    # (timestep, sigma, target, positions) is wired correctly.
    "flow_convention_debug_steps": 6,
    # Carpeta del transformer ORIGINAL (bf16, ~163 GB). OPCIONAL y VACIA por defecto.
    # NO hace falta para entrenar. Solo se usa como ULTIMO recurso si el repo NF4 no
    # trae la seccion `precision_critical`.
    # Con "" (vacio) ambas cosas se saltan limpiamente y el entrenamiento usa
    # exclusivamente la cache NF4, que es justo para lo que se comprimio.
    # NO la comentes: dejala en "". Comentarla borra la clave del dict y el codigo
    # que la lee revienta con KeyError.
    #
    # ORIGINAL (bf16, ~163 GB) transformer folder. OPTIONAL, EMPTY by default.
    # NOT needed for training. Only a LAST-RESORT fallback when the NF4 repo has no
    # `precision_critical` section. With "" it is skipped cleanly. Do NOT comment
    # this line out: removing the key makes the code that reads it raise KeyError.
    # Leave it as "".
    "original_transformer_dir": "",
    # Recarga sin NF4 los modulos que el modelo declara de precision critica
    # (_keep_in_fp32_modules) mas la ruta de modulacion (norm_out, context_embedder,
    # token_refiner). El error de NF4 en esos modulos rompe la cancelacion del AdaLN final.
    # Reload without NF4 the modules the model declares precision-critical, plus the
    # modulation path. NF4 error there breaks the final AdaLN cancellation.
    "fp32_repair_enabled": True,
    "strict_meta_load": True,
    # Además, aborta si son las NORMAS las que faltan (se rellenarían con 1.0, que no son
    # los valores entrenados). Ponlo a false solo si sabes lo que haces.
    "strict_meta_load_norms": True,
    # Versión mínima de formato de caché aceptada (el script 1 v3 escribe version=3).
    "min_cache_version": 3,

    # --- Diagnostico de presion de memoria -----------------------------------
    # Imprime num_alloc_retries / fragmentacion / RSS cada 10 pasos. Distingue
    # "presion de VRAM" (retries sube) de "thrashing de RAM" (rss cerca del limite
    # fisico) sin depender del resto del pipeline. Coste: una linea de texto.
    # Prints num_alloc_retries / fragmentation / RSS every 10 steps. Tells VRAM
    # pressure (retries climbing) apart from host-RAM thrashing (rss near the
    # physical limit) without depending on the rest of the pipeline.
    "alloc_probe": False,

    # --- Precision --------------------------------------------------------------
    # cast_frozen_bf16 ya NO toca los modulos que el modelo declara en
    # _keep_in_fp32_modules ni los buffers inv_freq del RoPE. Esta opcion permite
    # volver al comportamiento antiguo (castear absolutamente todo) para comparar.
    # cast_frozen_bf16 no longer touches the modules the model declares in
    # _keep_in_fp32_modules nor the RoPE inv_freq buffers. This option restores the
    # old behaviour (cast absolutely everything) for A/B comparisons.
    "cast_frozen_respect_fp32_modules": True,
    # El forward de MiniMax-H3 gestiona sus propios dtypes con casts explicitos
    # (.to(self.proj_in.weight.dtype), etc.). torch.autocast los anula y fuerza bf16
    # en todo nn.Linear. Ponlo a false para ejecutar el forward tal y como lo
    # disenaron los autores. MIDE antes de dejarlo fijo.
    # The MiniMax-H3 forward manages its own dtypes with explicit casts. torch.autocast
    # overrides them and forces bf16 on every nn.Linear. Set to false to run the forward
    # the way its authors designed it. MEASURE before making it permanent.
    "use_autocast": True,
    # Construye noisy/target en fp32 y castea solo al final. El tensor son ~25k
    # elementos: el coste es nulo y evita ~0.4% de error de cuantizacion en el
    # objetivo de regresion.
    # Builds noisy/target in fp32 and casts only at the end. The tensor is ~25k
    # elements: the cost is nil and it removes ~0.4% quantization error from the
    # regression target.
    "fp32_noise_construction": True,

    # --- Texto ------------------------------------------------------------------
    # Recorta el padding del prompt usando la attention_mask de la cache antes de
    # construir los indices empaquetados. El forward de H3 NO acepta attention_mask:
    # el unico canal para expresar padding es token_tags=-1, y ademas el eje t del
    # RoPE del video arranca en origin=text_len, asi que el padding desplaza la
    # geometria posicional del video segun la longitud de cada prompt.
    # Trims prompt padding using the cache's attention_mask before building the packed
    # indices. The H3 forward takes NO attention_mask: the only channel for padding is
    # token_tags=-1, and the video RoPE t axis starts at origin=text_len, so padding
    # shifts the video's positional geometry per prompt length.
    "trim_text_padding": True,

    # --- Audio ------------------------------------------------------------------
    # Solo afecta al ENTRENAMIENTO. Con use_audio_loss=False la fila de audio no
    # recibe gradiente, asi que mandarla solo cuesta atencion en los 50 bloques;
    # a true se elimina del todo (el forward soporta indices vacios).
    #
    # OJO: la PREVIEW no usa este ajuste y manda SIEMPRE las filas de audio. Son
    # dos situaciones distintas: entrenando se parte de un latente real y sobra
    # una fila muda, pero muestreando desde ruido puro el modelo tiene que ver el
    # layout completo con el que se entreno, que siempre lleva audio. Es lo que
    # hace tambien la implementacion de referencia.
    #
    # Training only. With use_audio_loss=False the audio row gets no gradient, so
    # sending it only costs attention across the 50 blocks. The PREVIEW ignores
    # this flag and always sends audio rows: sampling from pure noise has to show
    # the model the full layout it was trained on.
    "drop_audio_rows_when_unused": True,
}

CONFIG_PATH = "train_settings.json"

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)

    if isinstance(raw_cfg, dict):
        cfg = {str(k).strip(): v for k, v in raw_cfg.items()}
    else:
        cfg = {}

    print("[OK] Configuration loaded / Configuracion cargada: {}".format(CONFIG_PATH),
          flush=True)
else:
    cfg = {}
    print("[!] {} not found; using defaults / {} no existe; usando valores por "
          "defecto.".format(CONFIG_PATH, CONFIG_PATH), flush=True)


def cfg_get(key, default):
    if not isinstance(cfg, dict):
        return default
    return cfg.get(key, default)


def _cfg_bool(key, default):
    value = cfg_get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


DEBUG_TRAINING = _cfg_bool("debug_training", DEFAULTS["debug_training"])


# Estado de la barra de progreso: True mientras su linea esta abierta (se
# escribe con end="" para poder sobreescribirla en el paso siguiente).
# Progress-bar state: True while its line is left open (written with end="").
_PROGRESS_LINE = {"open": False}


def log_print(*args, **kwargs):
    """Imprime solo logs esenciales cuando debug_training=False.

    Los mensajes de diagnóstico/configuración/seguimiento pasan a depender de
    DEBUG_TRAINING. Fallos, avisos importantes, checkpoints, señales, inicio
    y final del entrenamiento se mantienen siempre visibles.
    """
    try:
        text = " ".join(str(x) for x in args).upper()
    except Exception:
        text = ""

    essential_tokens = (
        "ERROR",
        "EXCEPTION",
        "TRACEBACK",
        "[WARN]",
        "WARNING",
        "FALLO",
        "FALLÓ",
        "FAILED",
        "NO SE PUDO",
        "NO SE ENCONTRÓ",
        "NO EXISTE",
        "SIGNAL RECEIVED",
        "PAUSE",
        "PAUSA",
        "CHECKPOINT DETECTED",
        "CHECKPOINT SAVED",
        "SAVING CHECKPOINT",
        "RESUMING FROM STEP",
        "STARTING TRAINING",
        "TRAINING COMPLETED",
        "FINAL LORA SAVED",
        "ERROR EN TRAINER",
        "ADALN-FIX",
        # Estos tres tienen que verse SIEMPRE, tambien con debug_training=False:
        # el tiempo total es el resumen de la corrida, [LIVE] confirma que un
        # cambio de ajustes en caliente se ha aplicado (sin eco, el usuario no
        # sabe si su "Save JSON" ha surtido efecto) y [PREVIEW] avisa de que se
        # esta generando una imagen, que tarda minutos y si no parece un cuelgue.
        # These three must always be visible, debug off included: the run's total
        # time, the confirmation that a live settings change landed, and the
        # notice that a preview is rendering (it takes minutes and would
        # otherwise look like a freeze).
        "TOTAL TRAINING TIME",
        "[LIVE]",
        "[PREVIEW]",
        # El plan de VRAM es lo unico que dice cuantos bloques quedan residentes
        # y cuanto ocupa la base fuera de bloques. Sin el no hay forma de ajustar
        # vram_budget_gb con criterio, y estaba oculto con debug_training=False.
        # The VRAM plan is the only thing that reports how many blocks stay
        # resident and how big the non-block base is; without it vram_budget_gb
        # cannot be tuned. It was hidden with debug_training=False.
        "[VRAM-PLAN]",
        "[SPILL]",
        "[ATTN]",
        # El resumen del dataset son 3-4 lineas una sola vez por corrida y es
        # justo lo que hay que tener delante al decidir el siguiente experimento.
        # Ya no lleva [WARN] porque no es un aviso: es informacion.
        # The dataset summary is 3-4 lines once per run and is exactly what you
        # want in front of you when planning the next experiment. It no longer
        # carries [WARN] because it is not a warning: it is information.
        "[DATASET]",
    )

    if DEBUG_TRAINING or any(token in text for token in essential_tokens):
        # ----------------------------------------------------------------
        # CERRAR LA LINEA DE PROGRESO ANTES DE ESCRIBIR.
        #
        # La barra de progreso se reescribe con `print("\r...", end="")`, o sea
        # que deja la linea ABIERTA a proposito para poder sobreescribirla en el
        # paso siguiente. Cualquier mensaje posterior se pegaba a su derecha:
        #   ... | ETA 00:28:14 | gnorm 0.0996[PREVIEW] Generando preview...
        # Las llamadas `log_print("")` que habia para separar no servian: con
        # debug_training=False el texto vacio no contiene ninguna palabra clave
        # esencial y quedaba filtrado, justo cuando mas falta hacia.
        #
        # Resolverlo aqui lo arregla de una vez para TODO lo que pasa por
        # log_print: previews, checkpoints, [LIVE], vram_stats y los avisos.
        #
        # CLOSE THE PROGRESS LINE FIRST. The progress bar deliberately leaves its
        # line open (`print("\r...", end="")`) so it can overwrite itself, so any
        # later message was glued to its right. The `log_print("")` separators
        # did not help: with debug off, empty text matches no essential token and
        # was filtered out. Fixing it here covers everything that goes through
        # log_print at once.
        # ----------------------------------------------------------------
        if _PROGRESS_LINE["open"]:
            print("", flush=True)
            _PROGRESS_LINE["open"] = False
        print(*args, **kwargs)


# =============================================================================
# LOG A FICHERO / FILE LOG
#
# server.py solo streamea stdout al terminal del navegador: no guarda NADA en
# disco. Al cerrar la pestana o reiniciar el servidor se pierde el historial
# entero de la corrida, incluida la linea de progreso (loss / lr / s-it / gnorm),
# que se imprime con `print` directo y por tanto sobrevive a debug_training=False.
# Esto duplica stdout a <OUTPUT_DIR>/train_log.txt sin tocar lo que se ve.
#
# La linea de progreso se reescribe con "\r" y end="": en el fichero se convierte
# a salto de linea para que quede un registro por paso en vez de una unica linea
# kilometrica. Se hace flush en cada escritura, asi que un cuelgue o un OOM dejan
# el log completo hasta el ultimo instante.
#
# server.py only streams stdout to the browser terminal and persists nothing.
# This mirrors stdout into <OUTPUT_DIR>/train_log.txt without changing what is
# displayed. The "\r" progress line becomes one line per step in the file, and
# every write is flushed so a crash still leaves a complete log.
# =============================================================================

class _TeeStream(object):
    """Escribe en el stream original y ademas en un fichero.
    Writes to the original stream and additionally to a file."""

    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle

    def write(self, text):
        self._stream.write(text)
        try:
            self._handle.write(text.replace("\r", "\n") if "\r" in text else text)
            self._handle.flush()
        except Exception:
            pass
        return len(text)

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        try:
            self._handle.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        # isatty, encoding, fileno... los delega al stream real.
        return getattr(self._stream, name)


_TRAIN_LOG = {"handle": None, "stdout": None, "stderr": None, "path": None}


def install_train_log(output_dir):
    """Duplica stdout/stderr a <output_dir>/train_log.txt. Idempotente: si ya
    habia un log abierto (la GUI reutiliza el proceso) lo cierra antes.
    Mirrors stdout/stderr into <output_dir>/train_log.txt. Idempotent."""
    close_train_log()
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "train_log.txt")
        handle = open(path, "a", encoding="utf-8", errors="replace")
        handle.write("\n{}\n[TRAIN-LOG] Session started / Sesion iniciada: {}\n{}\n"
                     .format("=" * 78,
                             time.strftime("%Y-%m-%d %H:%M:%S"),
                             "=" * 78))
        handle.flush()
        _TRAIN_LOG["handle"] = handle
        _TRAIN_LOG["path"] = path
        _TRAIN_LOG["stdout"] = sys.stdout
        _TRAIN_LOG["stderr"] = sys.stderr
        sys.stdout = _TeeStream(sys.stdout, handle)
        sys.stderr = _TeeStream(sys.stderr, handle)
        print("[TRAIN-LOG] Full log written to / Guardando el log completo en: {}"
              .format(os.path.abspath(path)), flush=True)
        return path
    except Exception as e:
        # Un log que falla nunca debe tumbar un entrenamiento de horas.
        # A failing log must never take down an hours-long training run.
        print("[TRAIN-LOG][WARN] Could not open the file log: {} / No se pudo abrir el "
              "log de fichero: {}".format(e, e), flush=True)
        _TRAIN_LOG["handle"] = None
        return None


def format_duration_bilingual(seconds):
    """Segundos -> "2 Hours 26 Minutes / 2 Horas 26 Minutos".

    Se queda en la unidad util: por debajo de un minuto solo segundos, por
    debajo de una hora minutos y segundos, y a partir de ahi horas y minutos
    (los segundos sobran cuando la cifra son horas). Singular y plural correctos
    en los dos idiomas.
    Seconds -> a bilingual duration string, using only the units that matter.
    """
    try:
        total = int(max(0, round(float(seconds))))
    except Exception:
        return "? / ?"

    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)

    def en(value, unit):
        return "{} {}{}".format(value, unit, "" if value == 1 else "s")

    def es(value, unit_sg, unit_pl):
        return "{} {}".format(value, unit_sg if value == 1 else unit_pl)

    if hours > 0:
        return "{} {} / {} {}".format(
            en(hours, "Hour"), en(minutes, "Minute"),
            es(hours, "Hora", "Horas"), es(minutes, "Minuto", "Minutos"))
    if minutes > 0:
        return "{} {} / {} {}".format(
            en(minutes, "Minute"), en(secs, "Second"),
            es(minutes, "Minuto", "Minutos"), es(secs, "Segundo", "Segundos"))
    return "{} / {}".format(en(secs, "Second"), es(secs, "Segundo", "Segundos"))


def close_train_log():
    """Restaura stdout/stderr y cierra el fichero. / Restores the streams."""
    if _TRAIN_LOG["stdout"] is not None:
        sys.stdout = _TRAIN_LOG["stdout"]
    if _TRAIN_LOG["stderr"] is not None:
        sys.stderr = _TRAIN_LOG["stderr"]
    handle = _TRAIN_LOG["handle"]
    if handle is not None:
        try:
            handle.write("[TRAIN-LOG] Session closed / Sesion cerrada: {}\n"
                         .format(time.strftime("%Y-%m-%d %H:%M:%S")))
            handle.flush()
            handle.close()
        except Exception:
            pass
    _TRAIN_LOG["handle"] = None
    _TRAIN_LOG["stdout"] = None
    _TRAIN_LOG["stderr"] = None


TIMESTEP_CONVENTION = str(
    cfg_get("timestep_convention", DEFAULTS["timestep_convention"])
).strip().lower()
if TIMESTEP_CONVENTION not in ("one_minus_sigma", "sigma"):
    TIMESTEP_CONVENTION = "one_minus_sigma"
FLOW_CONV_DEBUG_STEPS = int(
    cfg_get("flow_convention_debug_steps", DEFAULTS["flow_convention_debug_steps"]) or 0
)

FP32_REPAIR_ENABLED = _cfg_bool("fp32_repair_enabled", DEFAULTS["fp32_repair_enabled"])
STRICT_META_LOAD = _cfg_bool("strict_meta_load", DEFAULTS["strict_meta_load"])
STRICT_META_LOAD_NORMS = _cfg_bool("strict_meta_load_norms", DEFAULTS["strict_meta_load_norms"])
MIN_CACHE_VERSION = int(cfg_get("min_cache_version", DEFAULTS["min_cache_version"]) or 0)

ALLOC_PROBE = _cfg_bool("alloc_probe", DEFAULTS["alloc_probe"])
CAST_FROZEN_RESPECT_FP32 = _cfg_bool(
    "cast_frozen_respect_fp32_modules", DEFAULTS["cast_frozen_respect_fp32_modules"])
USE_AUTOCAST = _cfg_bool("use_autocast", DEFAULTS["use_autocast"])
FP32_NOISE_CONSTRUCTION = _cfg_bool(
    "fp32_noise_construction", DEFAULTS["fp32_noise_construction"])
TRIM_TEXT_PADDING = _cfg_bool("trim_text_padding", DEFAULTS["trim_text_padding"])
DROP_AUDIO_ROWS_WHEN_UNUSED = _cfg_bool(
    "drop_audio_rows_when_unused", DEFAULTS["drop_audio_rows_when_unused"])

FLOW_TARGET_SIGN = int(cfg_get("flow_target_sign", DEFAULTS["flow_target_sign"]) or -1)
_sigma_shift_raw = cfg_get("sigma_shift", DEFAULTS["sigma_shift"])
SIGMA_SHIFT = float(_sigma_shift_raw) if _sigma_shift_raw is not None else None
LOGIT_NORMAL_MU = float(cfg_get("logit_normal_mu", DEFAULTS["logit_normal_mu"]))
LOGIT_NORMAL_STD = float(cfg_get("logit_normal_std", DEFAULTS["logit_normal_std"]))
SIGMA_RESOLUTION_SHIFT = _cfg_bool(
    "sigma_resolution_shift", DEFAULTS["sigma_resolution_shift"])
BF16_REDUCED_PRECISION = _cfg_bool(
    "bf16_reduced_precision_reduction", DEFAULTS["bf16_reduced_precision_reduction"])
DATASET_SAMPLER = str(
    cfg_get("dataset_sampler", DEFAULTS["dataset_sampler"])).strip().lower()

NF4_CACHE_DIR = str(cfg_get("nf4_cache_dir", DEFAULTS["nf4_cache_dir"])).strip()
TOTAL_STEPS = int(cfg_get("total_steps", DEFAULTS["total_steps"]))
BATCH_SIZE = int(cfg_get("batch_size", DEFAULTS["batch_size"]))
GRAD_ACCUM_STEPS = int(cfg_get("grad_accum_steps", DEFAULTS["grad_accum_steps"]))
LR = float(cfg_get("lr", DEFAULTS["lr"]))
MIN_LR_RATIO = float(cfg_get("min_lr_ratio", DEFAULTS["min_lr_ratio"]))
LR_SCHEDULE = str(cfg_get("lr_schedule", DEFAULTS["lr_schedule"])).strip().lower()
GRAD_PROFILE_EVERY = int(cfg_get("grad_profile_every", DEFAULTS["grad_profile_every"]) or 0)
OVERFIT_PROBE_EVERY = int(cfg_get("overfit_probe_every", DEFAULTS["overfit_probe_every"]) or 0)
WARMUP_STEPS = int(cfg_get("warmup_steps", DEFAULTS["warmup_steps"]))
LORA_RANK = int(cfg_get("lora_rank", DEFAULTS["lora_rank"]))
LORA_ALPHA = int(cfg_get("lora_alpha", DEFAULTS["lora_alpha"]))
WEIGHT_DECAY = float(cfg_get("weight_decay", DEFAULTS["weight_decay"]))


def _normalize_optimizer_type(value):
    text = str(value or "").strip().lower().replace("-", "").replace("_", "")
    if text in ("adamw8bit", "adamw8", "8bit", "pagedadamw8bit"):
        return "adamw8bit"
    return "adamw"


OPTIMIZER_TYPE = _normalize_optimizer_type(
    cfg_get("optimizer_type", DEFAULTS["optimizer_type"]))

MAX_GRAD_NORM = float(cfg_get("max_grad_norm", DEFAULTS["max_grad_norm"]))

PREVIEW_CAPTION_MODES = ("first", "random", "rotate", "custom")
PREVIEW_CAPTION_MODE = str(
    cfg_get("preview_caption_mode", DEFAULTS["preview_caption_mode"])).strip().lower()
if PREVIEW_CAPTION_MODE not in PREVIEW_CAPTION_MODES:
    PREVIEW_CAPTION_MODE = "first"
PREVIEW_EVERY = int(cfg_get("preview_every", DEFAULTS["preview_every"]) or 0)
PREVIEW_STEPS = max(2, int(cfg_get("preview_steps", DEFAULTS["preview_steps"]) or 12))
PREVIEW_CFG = float(cfg_get("preview_cfg", DEFAULTS["preview_cfg"]) or 0.0)
PREVIEW_SHIFT = float(cfg_get("preview_shift", DEFAULTS["preview_shift"]) or 12.0)
PREVIEW_SEED = int(cfg_get("preview_seed", DEFAULTS["preview_seed"]) or -1)
PREVIEW_VAE_DEVICE = str(
    cfg_get("preview_vae_device", DEFAULTS["preview_vae_device"])).strip().lower()
if PREVIEW_VAE_DEVICE not in ("cpu", "cuda"):
    PREVIEW_VAE_DEVICE = "cpu"
PREVIEW_FRAME_INDEX = int(cfg_get("preview_frame_index", DEFAULTS["preview_frame_index"]) or 0)
PREVIEW_NUM_FRAMES = max(1, int(cfg_get("preview_num_frames", DEFAULTS["preview_num_frames"]) or 1))

SAVE_EVERY = int(cfg_get("save_every", DEFAULTS["save_every"]))
SEED = int(cfg_get("seed", DEFAULTS["seed"]))
FRAME_RATE = float(cfg_get("frame_rate", DEFAULTS["frame_rate"]))
TRIGGER_WORD = str(cfg_get("trigger_word", DEFAULTS["trigger_word"])).strip()
PROJECT_NAME = str(cfg_get("project_name", DEFAULTS["project_name"])).strip()
MAX_TEXT_TOKENS = int(cfg_get("max_text_tokens", DEFAULTS["max_text_tokens"]) or 0)
CAPTION_DROPOUT = float(cfg_get("caption_dropout", DEFAULTS["caption_dropout"]) or 0.0)
LORA_SAVE_RAW_COPY = _cfg_bool("lora_save_raw_copy", DEFAULTS["lora_save_raw_copy"])

LORA_ONLY_ATTN = _cfg_bool("lora_only_attn", DEFAULTS["lora_only_attn"])
LORA_DTYPE_STR = str(cfg_get("lora_dtype", DEFAULTS["lora_dtype"])).strip().lower()
LORA_DTYPE = torch.bfloat16 if LORA_DTYPE_STR in ("bf16", "bfloat16") else torch.float32
LORA_EXCLUDE_REFINER = _cfg_bool("lora_exclude_refiner", DEFAULTS["lora_exclude_refiner"])
CAST_FROZEN_BF16 = _cfg_bool("cast_frozen_bf16", DEFAULTS["cast_frozen_bf16"])
USE_AUDIO_LOSS = _cfg_bool("use_audio_loss", DEFAULTS["use_audio_loss"])
LORA_KEY_PREFIX = str(cfg_get("lora_key_prefix", DEFAULTS["lora_key_prefix"]))

LOW_VRAM_12GB = _cfg_bool("low_vram_12gb", DEFAULTS["low_vram_12gb"])
ACTIVATION_OFFLOAD = _cfg_bool("activation_offload", DEFAULTS["activation_offload"])
RESIDENT_BLOCKS = int(cfg_get("resident_blocks", DEFAULTS["resident_blocks"]) or 0)
USE_SAGE_ATTENTION = _cfg_bool("use_sage_attention", DEFAULTS["use_sage_attention"])
NF4_CPU_HOME = _cfg_bool("nf4_cpu_home", DEFAULTS["nf4_cpu_home"])
PARK_MODE = str(cfg_get("park_mode", DEFAULTS["park_mode"])).strip().lower()
if PARK_MODE not in ("auto", "ram", "disk"):
    PARK_MODE = "auto"
RAM_LIMIT_GB = float(cfg_get("ram_limit_gb", DEFAULTS["ram_limit_gb"]))
PARK_DISK_DIR = str(cfg_get("park_disk_dir", DEFAULTS["park_disk_dir"]))
LOSS_CHUNK_ELEMENTS = int(cfg_get("loss_chunk_elements", DEFAULTS["loss_chunk_elements"]))

VRAM_BUDGET_GB = float(cfg_get("vram_budget_gb", DEFAULTS["vram_budget_gb"]))
VRAM_SWAP_GB = float(cfg_get("vram_swap_gb", DEFAULTS["vram_swap_gb"]))
VRAM_HEADROOM_GB = float(cfg_get("vram_headroom_gb", DEFAULTS["vram_headroom_gb"]))
VRAM_TRAINING_OVERHEAD_GB = float(
    cfg_get("vram_training_overhead_gb", DEFAULTS["vram_training_overhead_gb"]))
VRAM_HARD_CAP_ENABLED = _cfg_bool(
    "vram_hard_cap_enabled", DEFAULTS["vram_hard_cap_enabled"])
VRAM_EMPTY_CACHE_EVERY = int(
    cfg_get("vram_empty_cache_every", DEFAULTS["vram_empty_cache_every"]) or 0)
NF4_LOAD_CHUNK_LAYERS = int(cfg_get("nf4_load_chunk_layers", DEFAULTS["nf4_load_chunk_layers"]))

CPU_OFFLOAD_RESERVE_GB = float(cfg_get("cpu_offload_reserve_gb", DEFAULTS["cpu_offload_reserve_gb"]))

# Presupuesto VRAM MANUAL: no se reduce automáticamente según la VRAM física.
# El límite operativo es budget + swap + headroom.
VRAM_RESIDENT_MAX_BYTES = max(0.0, VRAM_BUDGET_GB) * 1e9
VRAM_SWAP_MAX_BYTES = max(0.0, VRAM_SWAP_GB) * 1e9
VRAM_HEADROOM_MAX_BYTES = max(0.0, VRAM_HEADROOM_GB) * 1e9
VRAM_RUNTIME_MAX_BYTES = VRAM_RESIDENT_MAX_BYTES + VRAM_SWAP_MAX_BYTES + VRAM_HEADROOM_MAX_BYTES
CPU_OFFLOAD_BLOCKS_ENABLED = _cfg_bool("cpu_offload_blocks_enabled", DEFAULTS["cpu_offload_blocks_enabled"])
EXPLICIT_CHECKPOINTING_ENABLED = _cfg_bool("explicit_checkpointing_enabled", DEFAULTS["explicit_checkpointing_enabled"])
CHECKPOINT_USE_REENTRANT = _cfg_bool(
    "checkpoint_use_reentrant", DEFAULTS["checkpoint_use_reentrant"])

CPU_OFFLOAD_DTYPE_STR = str(cfg_get("cpu_offload_dtype", DEFAULTS["cpu_offload_dtype"])).strip().lower()


def _resolve_cpu_dtype(s):
    s = (s or "fp32").lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


CPU_OFFLOAD_DTYPE = _resolve_cpu_dtype(CPU_OFFLOAD_DTYPE_STR)
CPU_OFFLOAD_THREADS = int(cfg_get("cpu_offload_threads", DEFAULTS["cpu_offload_threads"]))
CPU_OFFLOAD_CACHE_KWARGS = _cfg_bool("cpu_offload_cache_kwargs", DEFAULTS["cpu_offload_cache_kwargs"])
OFFLOAD_DEBUG_SYNC = _cfg_bool("offload_debug_sync", DEFAULTS["offload_debug_sync"])
AUDIO_CPU_OFFLOAD_ENABLED = _cfg_bool("audio_cpu_offload_enabled", DEFAULTS["audio_cpu_offload_enabled"])
CHECKPOINT_GPU_BLOCKS_ONLY = _cfg_bool("checkpoint_gpu_blocks_only", DEFAULTS["checkpoint_gpu_blocks_only"])
LORA_SKIP_FIRST_N_BLOCKS = int(cfg_get("lora_skip_first_n_blocks", DEFAULTS["lora_skip_first_n_blocks"]) or 0)

if LOW_VRAM_12GB:
    LOSS_CHUNK_ELEMENTS = min(LOSS_CHUNK_ELEMENTS, 125000)

if PROJECT_NAME:
    CACHE_DIR = "./cached_data_minimaxh3_{}".format(PROJECT_NAME)
    OUTPUT_DIR = "./minimaxh3_lora_output_{}".format(PROJECT_NAME)
else:
    CACHE_DIR = str(cfg_get("cache_dir", DEFAULTS["cache_dir"])).strip()
    OUTPUT_DIR = str(cfg_get("output_dir", DEFAULTS["output_dir"])).strip()

os.makedirs(OUTPUT_DIR, exist_ok=True)
RESUME_DIR = os.path.join(OUTPUT_DIR, "resume_checkpoint")
OPT_FILE = os.path.join(OUTPUT_DIR, "optimizer.pt")
STEP_FILE = os.path.join(OUTPUT_DIR, "current_step.txt")
LOSS_STATE_FILE = os.path.join(OUTPUT_DIR, "loss_state.json")


# =============================================================================
# BANNER
# =============================================================================

log_print()
log_print("=" * 80)
log_print("  MINIMAX-H3 LoRA TRAINER - NF4 CPU LOAD + VRAM PLAN + BLOCK SWAP JIT")
log_print("=" * 80)
log_print("  NF4 cache dir       : {}".format(NF4_CACHE_DIR))
log_print("  Project             : {}".format(PROJECT_NAME if PROJECT_NAME else "(Default)"))
log_print("  Trigger Word        : {}".format(TRIGGER_WORD))
log_print("  Cache Dir           : {}".format(CACHE_DIR))
log_print("  Output Dir          : {}".format(OUTPUT_DIR))
log_print("  Total Steps         : {}".format(TOTAL_STEPS))
log_print("  Learning Rate       : {}".format(LR))
log_print("  LoRA Rank/Alpha     : {}/{}".format(LORA_RANK, LORA_ALPHA))
log_print("  Optimizer           : {}".format(
    "AdamW (fp32)" if OPTIMIZER_TYPE == "adamw" else "PagedAdamW8bit (8-bit)"))
log_print("  Batch / Grad Accum  : {}/{}".format(BATCH_SIZE, GRAD_ACCUM_STEPS))
log_print("  Max Text Tokens     : {}".format(MAX_TEXT_TOKENS))
log_print("  LoRA Only Attention : {}".format("ON" if LORA_ONLY_ATTN else "OFF"))
log_print("  Use Audio Loss      : {}".format("ON" if USE_AUDIO_LOSS else "OFF"))
log_print("  Seed                : {} ({})".format(SEED, "RANDOM" if SEED <= 0 else "FIXED"))
log_print("  Low VRAM 12GB mode  : {}".format("ON" if LOW_VRAM_12GB else "OFF"))
log_print("  Activation Offload  : {}".format("ON" if ACTIVATION_OFFLOAD else "OFF"))
log_print("  Resident Blocks     : {}".format(
    RESIDENT_BLOCKS if RESIDENT_BLOCKS > 0 else "AUTO (plan de VRAM / VRAM plan)"))
log_print("  Sage Attention      : {}".format("ON" if USE_SAGE_ATTENTION else "OFF"))
log_print("  NF4 CPU Home        : {}".format("ON" if NF4_CPU_HOME else "OFF"))
log_print("  Block Park Mode     : {}".format(PARK_MODE.upper()))
log_print("  RAM Limit GB        : {}".format(
    RAM_LIMIT_GB if RAM_LIMIT_GB > 0 else "OFF (sin limite / no limit)"))
log_print("  Loss Chunk Elements : {}".format(LOSS_CHUNK_ELEMENTS))
log_print("  CPU Offload Bloques : {}".format("ON" if CPU_OFFLOAD_BLOCKS_ENABLED else "OFF"))
log_print("  VRAM Budget GB      : {}".format(VRAM_BUDGET_GB))
log_print("  VRAM Swap GB        : {}".format(VRAM_SWAP_GB))
log_print("  VRAM Headroom GB    : {}".format(VRAM_HEADROOM_GB))
log_print("  NF4 load chunk      : {}".format(NF4_LOAD_CHUNK_LAYERS))
log_print("  CPU Offload dtype   : {}".format(CPU_OFFLOAD_DTYPE_STR))
log_print("  CPU Offload threads : {}".format(CPU_OFFLOAD_THREADS))
log_print("  Audio CPU offload   : {}".format("ON" if AUDIO_CPU_OFFLOAD_ENABLED else "OFF"))
log_print("  Flow target sign    : {} ({})".format(
    FLOW_TARGET_SIGN,
    "x0-noise" if FLOW_TARGET_SIGN < 0 else "noise-x0"
))
log_print("  Debug training      : {}".format("ON" if DEBUG_TRAINING else "OFF"))
log_print("  Sigma schedule      : {}".format(
    "logit-normal mu={:g} std={:g} res_shift={}".format(
        LOGIT_NORMAL_MU, LOGIT_NORMAL_STD, SIGMA_RESOLUTION_SHIFT)
    if SIGMA_SHIFT is None
    else "legacy uniform-u, shift={}".format(SIGMA_SHIFT)
))
log_print("  LoRA weights dtype  : {} (el estado de AdamW hereda este dtype)".format(
    LORA_DTYPE_STR))
log_print("  Updates optimizador : {} ({} micro-pasos / grad_accum {})".format(
    int(TOTAL_STEPS / max(1, GRAD_ACCUM_STEPS)), TOTAL_STEPS, GRAD_ACCUM_STEPS))

if int(TOTAL_STEPS / max(1, GRAD_ACCUM_STEPS)) < 400:
    print("[CONFIG][WARN] Solo {} actualizaciones del optimizador ({} pasos / accum {}). "
          "Un LoRA de identidad necesita >=600-1000: por debajo de eso el parecido se "
          "queda a medias hagas lo que hagas con el resto de ajustes. Sube total_steps "
          "o baja grad_accum_steps a 1. / Only {} optimizer updates; identity LoRAs need "
          ">=600-1000.".format(
              int(TOTAL_STEPS / max(1, GRAD_ACCUM_STEPS)), TOTAL_STEPS, GRAD_ACCUM_STEPS,
              int(TOTAL_STEPS / max(1, GRAD_ACCUM_STEPS))), flush=True)

log_print("  Autocast bf16       : {}".format(
    "ON" if USE_AUTOCAST else "OFF (model handles dtypes / el modelo gestiona dtypes)"))
log_print("  fp32 noise/target   : {}".format("ON" if FP32_NOISE_CONSTRUCTION else "OFF"))
log_print("  Trim text padding   : {}".format("ON" if TRIM_TEXT_PADDING else "OFF"))
log_print("  Keep fp32 modules   : {}".format(
    "RESPECTED / RESPETADO" if CAST_FROZEN_RESPECT_FP32 else "IGNORED / IGNORADO"))
log_print("  Alloc probe         : {}".format("ON" if ALLOC_PROBE else "OFF"))

if LORA_SKIP_FIRST_N_BLOCKS > 0:
    log_print("  LoRA salta bloques  : 0-{} sin LoRA ({} bloques)".format(
        LORA_SKIP_FIRST_N_BLOCKS - 1, LORA_SKIP_FIRST_N_BLOCKS
    ))
else:
    log_print("  LoRA salta bloques  : ninguno")

if BATCH_SIZE != 1:
    print("[CONFIG][WARN] batch_size={} is NOT used: the loop takes one cached entry per "
          "step and B comes from the latent itself. The effective batch is 1 x "
          "grad_accum_steps={}. / batch_size={} NO se usa: el bucle toma una entrada de "
          "cache por paso y B sale del propio latente. El lote efectivo es 1 x "
          "grad_accum_steps={}.".format(
              BATCH_SIZE, GRAD_ACCUM_STEPS, BATCH_SIZE, GRAD_ACCUM_STEPS), flush=True)

log_print("=" * 80)


# =============================================================================
# UTILIDADES
# =============================================================================

def free_vram(*objects, clear_cache=True, collect=True):
    """collect=False evita el gc.collect() completo.

    Una coleccion completa sobre el heap de un modelo de 33B (millones de objetos:
    parametros, quant_state, dicts de modulos) cuesta entre 0,5 y 2 s. Llamarla una
    vez por paso de entrenamiento era una de las mayores fuentes de s/it.
    collect=False skips the full gc.collect(). A full collection over a 33B model's
    heap costs 0.5-2 s; calling it once per training step was a major s/it sink.
    """
    for obj in objects:
        try:
            del obj
        except Exception:
            pass

    if collect:
        gc.collect()

    if torch.cuda.is_available() and clear_cache:
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def vram_stats(label=""):
    if not torch.cuda.is_available():
        return

    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak_alloc = torch.cuda.max_memory_allocated() / 1e9
    peak_reserved = torch.cuda.max_memory_reserved() / 1e9

    # Lo que satura la tarjeta y lo que ve nvidia-smi NO es `alloc`, es `reserved`
    # (el pool del caching allocator) mas el contexto de CUDA (~0,5-1 GB). Un
    # reserved muy por encima de alloc = fragmentacion del asignador, que es
    # exactamente lo que produce el block swap al pedir y soltar los mismos
    # bloques miles de veces por paso. Por eso hay que imprimir PICO, no instante.
    # What saturates the card (and what nvidia-smi shows) is `reserved` + the CUDA
    # context, not `alloc`. reserved >> alloc means allocator fragmentation.
    log_print(
        "[VRAM] {} | Alloc: {:.2f} GB (pico {:.2f}) | Reserved: {:.2f} GB (pico {:.2f})"
        .format(label, alloc, peak_alloc, reserved, peak_reserved),
        flush=True,
    )


def config_get(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    if hasattr(config, "get"):
        try:
            return config.get(key, default)
        except Exception:
            pass
    return getattr(config, key, default)


def safe_int(value, default=1):
    try:
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return default
            value = value[0]
        return int(value)
    except Exception:
        return default


def safe_float(value, default=1.0):
    try:
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return default
            value = value[0]
        return float(value)
    except Exception:
        return default


def get_parent_module(root, name):
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def get_patch_sizes(config):
    raw = config_get(config, "patch_size", 1)
    raw_t = config_get(config, "patch_size_t", None)
    if raw_t is None:
        raw_t = config_get(config, "temporal_patch_size", None)

    if isinstance(raw, dict):
        pt = safe_int(raw.get("t", raw.get("temporal", raw_t)), 1)
        ph = safe_int(raw.get("h", raw.get("height", raw.get("spatial", 1))), 1)
        pw = safe_int(raw.get("w", raw.get("width", ph)), ph)
        return max(1, pt), max(1, ph), max(1, pw)

    if isinstance(raw, (list, tuple)):
        vals = [safe_int(v, 1) for v in raw]
        if len(vals) >= 3:
            pt, ph, pw = vals[0], vals[1], vals[2]
        elif len(vals) == 2:
            if raw_t is not None:
                ph, pw = vals[0], vals[1]
                pt = safe_int(raw_t, 1)
            elif vals[0] == 1 and vals[1] != 1:
                pt = vals[0]
                ph = pw = vals[1]
            else:
                ph, pw = vals[0], vals[1]
                pt = 1
        elif len(vals) == 1:
            ph = pw = vals[0]
            pt = safe_int(raw_t, 1)
        else:
            pt = ph = pw = 1
    else:
        ph = pw = safe_int(raw, 1)
        pt = safe_int(raw_t, 1)

    return max(1, pt), max(1, ph), max(1, pw)


def filter_forward_kwargs(kwargs, forward_fn):
    try:
        signature = inspect.signature(forward_fn)
    except Exception:
        return kwargs

    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    )
    if has_var_keyword:
        return kwargs

    return {k: v for k, v in kwargs.items() if k in signature.parameters}


def configure_cpu_backend():
    try:
        if CPU_OFFLOAD_THREADS > 0:
            torch.set_num_threads(int(CPU_OFFLOAD_THREADS))
    except Exception as e:
        log_print("[CPU] No se pudo fijar torch.set_num_threads: {}".format(e), flush=True)

    try:
        log_print("[CPU] torch.get_num_threads() = {}".format(torch.get_num_threads()), flush=True)
    except Exception:
        pass


def _clear_offload_cpu_cache():
    return


def _debug_tensor_stats(name, t):
    try:
        tf = t.detach().float()
        log_print(
            "[DEBUG-TENSOR] {} | shape={} | mean={:.6g} | std={:.6g} | min={:.6g} | max={:.6g}".format(
                name,
                tuple(tf.shape),
                tf.mean().item(),
                tf.std().item(),
                tf.min().item(),
                tf.max().item(),
            ),
            flush=True,
        )
    except Exception as e:
        log_print("[DEBUG-TENSOR] {} stats failed: {}".format(name, e), flush=True)


def ensure_trainable_parameters_on_cuda(module):
    for p in module.parameters():
        if getattr(p, "requires_grad", False) and p.device.type != "cuda":
            p.data = p.data.to("cuda")


# =============================================================================
# MÓDULOS QUE NO DEBEN ESTAR CUANTIZADOS
# =============================================================================

NON_CONVERT_PATTERNS = (
    "proj_in",
    "audio_proj_in",
    "context_embedder",
    "time_embedder",
    "time_proj",
    "token_refiner",
    "norm_out",
    "proj_out",
    "audio_proj_out",
)


def should_restore_non_convert_module(name):
    for pattern in NON_CONVERT_PATTERNS:
        if name == pattern or name.startswith(pattern + "."):
            return True
    return False


class CastingLinear(torch.nn.Linear):
    def forward(self, input):
        if input.dtype != self.weight.dtype:
            input = input.to(self.weight.dtype)
        return torch.nn.functional.linear(input, self.weight, self.bias)


_ADALN_DTYPE_PATCHED = False


def _nf4_compute_dtype(lin):
    """
    Devuelve el dtype de COMPUTO de una capa 4-bit, o None si la capa no es 4-bit.
    En bitsandbytes `weight.dtype` es el del empaquetado (torch.uint8), no el del
    computo; diffusers hace `x.to(self.linear.weight.dtype)` dando por hecho lo
    contrario, y eso convierte las activaciones a enteros.
    Returns the COMPUTE dtype of a 4-bit layer, or None if the layer is not 4-bit.
    In bitsandbytes `weight.dtype` is the packing dtype (torch.uint8), not the
    compute dtype; diffusers does `x.to(self.linear.weight.dtype)` assuming the
    opposite, which casts the activations to integers.
    """
    if lin is None:
        return None
    base = getattr(lin, "base_layer", None) or lin
    w = getattr(base, "weight", None)
    if not isinstance(w, Params4bit):
        return None
    dt = getattr(base, "compute_dtype", None)
    if dt is None:
        qs = getattr(w, "quant_state", None)
        dt = getattr(qs, "dtype", None)
    if dt is None or dt == torch.uint8:
        dt = torch.bfloat16
    return dt


def install_adaln_dtype_fix():
    """
    Arregla el choque diffusers <-> bitsandbytes en las proyecciones AdaLN.

    `MiniMaxH3AdaLayerNormModulation.forward` hace:
        temb = self.linear(silu(temb).to(self.linear.weight.dtype))
    Con `self.linear` cuantizado a NF4, `weight.dtype` es `torch.uint8`, asi que
    las activaciones (~0.01) se truncan a enteros: los positivos pequenos a 0 y
    los negativos envuelven a 254/255. Ademas `Linear4bit.forward` termina con
    `out.to(inp_dtype)`, y como la entrada era uint8 la SALIDA vuelve a uint8.
    Resultado: shift/scale/gate enteros, y `norm1(x)*(1+scale)+shift` explota ya
    en el bloque 0. Aqui se reimplementan los dos `forward` usando el dtype de
    computo real de la capa. Si la capa no es 4-bit se llama al original.

    Fixes the diffusers <-> bitsandbytes clash in the AdaLN projections. With an
    NF4 `linear`, `weight.dtype` is `torch.uint8`, so the activations are
    truncated to integers and `Linear4bit.forward`'s trailing `out.to(inp_dtype)`
    casts the OUTPUT back to uint8 too. shift/scale/gate become integers and
    `norm1(x)*(1+scale)+shift` explodes from block 0 on. Both forwards are
    reimplemented using the layer's real compute dtype; non-4-bit layers fall
    back to the original code.
    """
    global _ADALN_DTYPE_PATCHED
    if _ADALN_DTYPE_PATCHED:
        return True

    try:
        from diffusers.models.transformers import transformer_minimax_h3 as _tm
    except Exception as _e:
        log_print("[ADALN-FIX] could not import transformer_minimax_h3: {} / no se pudo "
                  "importar transformer_minimax_h3: {}".format(_e, _e), flush=True)
        return False

    _touched = []

    _mod_cls = getattr(_tm, "MiniMaxH3AdaLayerNormModulation", None)
    if _mod_cls is not None and not getattr(_mod_cls, "_dtype_patched", False):
        _orig_mod = _mod_cls.forward

        def _mod_forward(self, temb, __orig=_orig_mod):
            _dt = _nf4_compute_dtype(getattr(self, "linear", None))
            if _dt is None:
                return __orig(self, temb)
            out = self.linear(torch.nn.functional.silu(temb).to(_dt))
            out = out.view(-1, 6 * self.hidden_size)
            return out.chunk(6, dim=-1)

        _mod_cls.forward = _mod_forward
        _mod_cls._dtype_patched = True
        _touched.append("MiniMaxH3AdaLayerNormModulation")

    _out_cls = getattr(_tm, "MiniMaxH3AdaLayerNormOut", None)
    if _out_cls is not None and not getattr(_out_cls, "_dtype_patched", False):
        _orig_out = _out_cls.forward

        def _out_forward(self, hidden_states, temb, timestep_indices, __orig=_orig_out):
            _dt = _nf4_compute_dtype(getattr(self, "linear", None))
            if _dt is None:
                return __orig(self, hidden_states, temb, timestep_indices)
            shift, scale = self.linear(
                torch.nn.functional.silu(temb).to(_dt)).chunk(2, dim=-1)
            hidden_states = self.norm(hidden_states)
            return hidden_states * (
                1.0 + scale.index_select(0, timestep_indices)
            ) + shift.index_select(0, timestep_indices)

        _out_cls.forward = _out_forward
        _out_cls._dtype_patched = True
        _touched.append("MiniMaxH3AdaLayerNormOut")

    _ADALN_DTYPE_PATCHED = True
    log_print("[ADALN-FIX] uint8 activation-cast fix installed on: {} / parche del "
              "cast a uint8 de las activaciones instalado en: {}".format(
                  ", ".join(_touched) or "-", ", ".join(_touched) or "-"), flush=True)
    return True


def dequantize_linear4bit_module(root, name, target_device="cuda", target_dtype=torch.bfloat16):
    parent, child_name = get_parent_module(root, name)

    if child_name.isdigit():
        module = parent[int(child_name)]
    else:
        module = getattr(parent, child_name)

    if not isinstance(module, Linear4bit):
        return False

    weight = module.weight
    quant_state = getattr(weight, "quant_state", None)
    if quant_state is None:
        return False

    try:
        import bitsandbytes.functional as bnbF

        source_weight = weight.data
        if source_weight.device.type != "cuda":
            source_weight = source_weight.to("cuda")

        # IMPORTANTE: quant_state (absmax/code/nested state) tiene que estar
        # en el MISMO device que source_weight. Antes esto lo garantizaba (por
        # accidente) el transformer.to("cuda") global que se hacía tras la
        # carga NF4; al quitarlo, quant_state se queda en CPU (fue aparcado
        # ahí durante la carga por lotes) mientras source_weight ya está en
        # CUDA, y bitsandbytes lanza "illegal memory access" al mezclar
        # punteros CPU/CUDA dentro del kernel de dequantización.
        quant_state = _move_quant_state_to_device(quant_state, "cuda")
        weight.quant_state = quant_state

        dequantized_weight = bnbF.dequantize_4bit(source_weight, quant_state)

    except Exception as e:
        log_print("[WARN] No se pudo dequantizar {}: {}".format(name, e), flush=True)
        return False

    dequantized_weight = dequantized_weight.to(target_device, dtype=target_dtype)

    bias = None
    if module.bias is not None:
        bias_src = module.bias.detach()
        if bias_src.device.type != "cuda":
            bias_src = bias_src.to("cuda")
        bias = bias_src.to(target_device, dtype=target_dtype)

    new_layer = CastingLinear(
        module.in_features,
        module.out_features,
        bias=bias is not None,
        device=target_device,
        dtype=target_dtype,
    )

    new_layer.weight = torch.nn.Parameter(
        dequantized_weight.contiguous(),
        requires_grad=False,
    )

    if bias is not None:
        new_layer.bias = torch.nn.Parameter(
            bias.contiguous(),
            requires_grad=False,
        )

    if child_name.isdigit():
        parent[int(child_name)] = new_layer
    else:
        setattr(parent, child_name, new_layer)

    del dequantized_weight
    if target_device == "cpu":
        free_vram()

    return True

def restore_non_convert_modules(root):
    restored = []
    for name, module in list(root.named_modules()):
        if isinstance(module, Linear4bit) and should_restore_non_convert_module(name):
            if dequantize_linear4bit_module(root, name, target_device="cpu"):
                restored.append(name)

    log_print("[NF4] Módulos dequantizados (no-convert): {}".format(len(restored)), flush=True)
    for n in restored[:20]:
        log_print("  RESTORED: {}".format(n), flush=True)
    if len(restored) > 20:
        log_print("  ... y {} más".format(len(restored) - 20), flush=True)

    return restored


# =============================================================================
# CARGA DE TRANSFORMER DESDE CACHÉ NF4
# =============================================================================

def _find_transformer_cache_dir(nf4_root):
    candidates = [
        os.path.join(nf4_root, "transformer"),
        os.path.join(nf4_root, "transformers", "transformer"),
        os.path.join(nf4_root, "transformers"),
    ]

    index_candidates = (
        "index.json",
        "config_nf4.json",
        os.path.join("weights", "index.json"),
        os.path.join("weights", "config_nf4.json"),
    )

    for c in candidates:
        if not os.path.isdir(c):
            continue
        for idx in index_candidates:
            if os.path.exists(os.path.join(c, idx)):
                return c

    return None


def _find_nf4_index_path(cache_dir):
    candidates_dirs = [
        cache_dir,
        os.path.join(cache_dir, "weights"),
    ]

    candidate_names = (
        "index.json",
        "config_nf4.json",
    )

    for d in candidates_dirs:
        if not os.path.isdir(d):
            continue
        for name in candidate_names:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p

    raise FileNotFoundError(
        "No se encontró index.json ni config_nf4.json en {}".format(cache_dir)
    )


def resolve_transformer_class(cls_name):
    import importlib
    import diffusers

    cls = getattr(diffusers, cls_name, None)
    if cls is not None:
        return cls

    modules_to_try = (
        "models",
        "models.transformers",
        "pipelines",
    )

    for mod_name in modules_to_try:
        try:
            mod = importlib.import_module("diffusers.{}".format(mod_name))
            cls = getattr(mod, cls_name, None)
            if cls is not None:
                return cls
        except Exception:
            pass

    raise RuntimeError("No se pudo resolver la clase transformer: {}".format(cls_name))


def _instantiate_transformer_on_meta(transformer_cls, config_dict):
    errors = []

    try:
        from accelerate import init_empty_weights
        with init_empty_weights():
            model = transformer_cls.from_config(config_dict)
        log_print("[NF4] Modelo instanciado con accelerate.init_empty_weights().", flush=True)
        return model
    except Exception as e:
        errors.append("accelerate.init_empty_weights: {}".format(e))

    try:
        if not hasattr(torch, "set_default_device"):
            raise RuntimeError("torch.set_default_device no disponible")

        torch.set_default_device("meta")
        torch.set_default_dtype(torch.bfloat16)

        try:
            model = transformer_cls.from_config(config_dict)
            log_print("[NF4] Modelo instanciado con torch.set_default_device('meta').", flush=True)
            return model
        finally:
            try:
                torch.set_default_device("cpu")
            except Exception:
                pass
            torch.set_default_dtype(torch.bfloat16)

    except Exception as e:
        errors.append("torch.set_default_device meta + from_config: {}".format(e))

    try:
        if not hasattr(torch, "set_default_device"):
            raise RuntimeError("torch.set_default_device no disponible")

        init_kwargs = {
            k: v
            for k, v in config_dict.items()
            if not k.startswith("_")
        }

        torch.set_default_device("meta")
        torch.set_default_dtype(torch.bfloat16)

        try:
            model = transformer_cls(**init_kwargs)
            log_print("[NF4] Modelo instanciado con constructor directo en meta.", flush=True)
            return model
        finally:
            try:
                torch.set_default_device("cpu")
            except Exception:
                pass
            torch.set_default_dtype(torch.bfloat16)

    except Exception as e:
        errors.append("constructor directo meta: {}".format(e))

    raise RuntimeError(
        "No se pudo instanciar el transformer en META device.\n" + "\n".join(errors)
    )


def audit_meta_tensors(module):
    """Clasifica lo que sigue en META antes de tocarlo / Classify what is still on META.

    Devuelve (params_norm, params_otros, buffers). Esta separación importa: rellenar una
    norm con 1.0 es un fallback tolerable (escala neutra), pero rellenar un peso entrenado
    o un buffer calculado (rope inv_freq) con ceros DESTRUYE el modelo en silencio.
    """
    p_norm, p_other, bufs = [], [], []

    def _is_norm_weight(n):
        low = n.lower()
        return low.endswith(".weight") and any(
            k in low for k in ("norm", "ln", "layernorm", "rmsnorm")
        )

    for name, param in module.named_parameters():
        if isinstance(param, Params4bit):
            continue
        if param.device.type == "meta":
            (p_norm if _is_norm_weight(name) else p_other).append(name)

    for name, buf in module.named_buffers():
        if buf.device.type == "meta":
            bufs.append(name)

    return p_norm, p_other, bufs


def rebuild_dit_rope_buffers(module):
    """Reconstruye cualquier inv_freq del DiT que haya nacido en META.

    Igual que en el script 1: inv_freq es un buffer CALCULADO, no siempre viaja en el
    checkpoint. Si se rellena con ceros, cos=1 y sin=0 -> el RoPE se vuelve la identidad
    y el transformer pierde toda la geometría posicional sin dar ningún error.
    """
    fixed, failed = [], []

    for name, mod in module.named_modules():
        buf = getattr(mod, "inv_freq", None)
        if buf is None or not torch.is_tensor(buf):
            continue
        broken = bool(buf.is_meta) or (
            not buf.is_meta and float(buf.detach().abs().sum()) == 0.0
        )
        if not broken:
            continue

        n = int(buf.shape[-1])
        theta = None
        for src in (getattr(mod, "config", None), getattr(module, "config", None)):
            if src is None:
                continue
            rp = getattr(src, "rope_parameters", None)
            if isinstance(rp, dict) and rp.get("rope_theta"):
                theta = float(rp["rope_theta"])
                break
            v = getattr(src, "rope_theta", None)
            if isinstance(v, (int, float)) and v:
                theta = float(v)
                break
        if theta is None:
            theta = float(getattr(mod, "theta", 10000.0) or 10000.0)

        try:
            idx = torch.arange(n, dtype=torch.float32)
            new_buf = 1.0 / (theta ** (idx / float(n)))
            if float(new_buf.abs().sum()) == 0.0 or n < 1:
                raise RuntimeError("tabla RoPE inválida")
            mod.register_buffer("inv_freq", new_buf, persistent=False)
            fixed.append((name, n, theta))
            log_print(
                "[ROPE-DiT] Reconstruido {} n={} theta={:g}".format(name, n, theta),
                flush=True,
            )
        except Exception as e:
            failed.append((name, str(e)))
            log_print("[ROPE-DiT] FALLO al reconstruir {}: {}".format(name, e), flush=True)

    if not fixed and not failed:
        log_print("[ROPE-DiT] Ningún inv_freq pendiente (o el modelo calcula RoPE al vuelo).",
                  flush=True)
    return fixed, failed


def materialize_meta_tensors(module, device="cpu"):
    materialized = []

    norm_keywords = (
        "norm",
        "ln",
        "layernorm",
        "rmsnorm",
    )

    def is_norm_weight(full_name):
        low = full_name.lower()
        return low.endswith(".weight") and any(k in low for k in norm_keywords)

    for name, param in list(module.named_parameters()):
        if isinstance(param, Params4bit):
            continue

        if param.device.type == "meta":
            parent, child_name = get_parent_module(module, name)

            if param.is_floating_point() and is_norm_weight(name):
                data = torch.ones(
                    param.shape,
                    dtype=param.dtype,
                    device=device,
                )
            else:
                data = torch.zeros(
                    param.shape,
                    dtype=param.dtype,
                    device=device,
                )

            new_param = torch.nn.Parameter(
                data,
                requires_grad=param.requires_grad,
            )

            if child_name.isdigit():
                parent[int(child_name)] = new_param
            else:
                setattr(parent, child_name, new_param)

            materialized.append(name)

    for name, buf in list(module.named_buffers()):
        if buf.device.type == "meta":
            parent, child_name = get_parent_module(module, name)

            data = torch.zeros(
                buf.shape,
                dtype=buf.dtype,
                device=device,
            )

            if child_name.isdigit():
                parent[int(child_name)] = data
            else:
                if child_name in getattr(parent, "_buffers", {}):
                    parent.register_buffer(child_name, data)
                else:
                    setattr(parent, child_name, data)

            materialized.append(name)

    return materialized


_DTYPE_BY_NAME = {"float32": torch.float32, "fp32": torch.float32,
                  "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                  "float16": torch.float16, "fp16": torch.float16}


def _repair_from_nf4_index(transformer):
    """Carga la seccion `precision_critical` del propio repo NF4.
    Load the `precision_critical` section from the NF4 repo itself.

    Cada entrada es una clave de tensor con ruta completa (p.ej. `proj_in.weight`) guardada
    SIN cuantizar y con su dtype exigido. Se reconstruye el nn.Linear entero cuando estan
    presentes weight y bias.
    """
    cache_dir = _find_transformer_cache_dir(NF4_CACHE_DIR)
    if not cache_dir:
        return 0
    index_path = _find_nf4_index_path(cache_dir)
    if not index_path or not os.path.isfile(index_path):
        return 0
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            pc = (json.load(f) or {}).get("precision_critical") or {}
    except Exception:
        return 0
    if not pc:
        return 0

    weights_dir = os.path.join(cache_dir, "weights")
    # agrupar por modulo / group by module
    by_module = {}
    for key, info in pc.items():
        leaf = key.rsplit(".", 1)[-1]
        if leaf not in ("weight", "bias"):
            continue
        by_module.setdefault(key[: -(len(leaf) + 1)], {})[leaf] = info

    fixed, failed = [], []
    skipped_in_block = []
    mods = dict(transformer.named_modules())
    for mname, parts in sorted(by_module.items()):
        if mname not in mods or "weight" not in parts:
            continue

        # BLINDAJE: nunca sustituir un Linear4bit que viva DENTRO de los bloques.
        # El swap JIT solo mueve Linear4bit; park_nf4_block() deja en CUDA de forma
        # permanente todo lo demas. Un modulo de bloque convertido a nn.Linear se
        # queda fijo en VRAM para los 50 bloques y reduce los bloques residentes.
        # GUARD: never replace a Linear4bit living INSIDE the blocks. The JIT swap only
        # moves Linear4bit; park_nf4_block() pins everything else on CUDA permanently, so
        # a block module turned into nn.Linear stays in VRAM for all 50 blocks.
        if mname.startswith(("transformer_blocks.", "blocks.")):
            skipped_in_block.append(mname)
            continue

        try:
            tensors = {}
            for leaf, info in parts.items():
                fp = os.path.join(weights_dir, info["file"])
                if not os.path.isfile(fp):
                    raise FileNotFoundError(info["file"])
                with safe_open(fp, framework="pt", device="cpu") as f:
                    key = mname + "." + leaf
                    src = key if key in f.keys() else list(f.keys())[0]
                    t = f.get_tensor(src)
                dt = _DTYPE_BY_NAME.get(str(info.get("dtype", "")).lower())
                tensors[leaf] = t.to(dt) if dt is not None else t

            w = tensors["weight"]
            b = tensors.get("bias")

            # ----------------------------------------------------------------
            # PESO DE 1 DIMENSION = ES UNA NORMA, NO UNA LINEAR.
            #
            # La seccion precision_critical incluye norm_out.norm, que es una
            # RMSNorm: su peso es un vector [5376]. Construir un nn.Linear con
            # w.shape[1] lanzaba "tuple index out of range" y la norma acababa en
            # la lista de fallos, imprimiendo un aviso que asustaba sin que
            # hubiera nada roto. Una norma nunca pasa por NF4, asi que lo unico
            # que hay que hacer es reemplazar su parametro por el tensor cargado,
            # que es justo la restauracion de precision que se buscaba.
            #
            # A 1-D weight means this is a NORM, not a Linear. Building an
            # nn.Linear from w.shape[1] raised "tuple index out of range" and the
            # norm was reported as a failure when nothing was actually wrong.
            # ----------------------------------------------------------------
            if w.ndim == 1:
                target = mods[mname]
                applied = []
                for leaf, t in tensors.items():
                    current = getattr(target, leaf, None)
                    if isinstance(current, torch.nn.Parameter):
                        setattr(target, leaf,
                                torch.nn.Parameter(t, requires_grad=False))
                        applied.append(leaf)
                if not applied:
                    raise RuntimeError(
                        "peso 1-D pero {} no tiene parametros que restaurar / "
                        "1-D weight but {} has no parameters to restore".format(
                            mname, mname))
                fixed.append((mname, str(w.dtype).replace("torch.", "")))
                continue

            new = torch.nn.Linear(w.shape[1], w.shape[0], bias=b is not None)
            new.weight = torch.nn.Parameter(w, requires_grad=False)
            if b is not None:
                new.bias = torch.nn.Parameter(b, requires_grad=False)
            new.requires_grad_(False)

            parent, child = get_parent_module(transformer, mname)
            if child.isdigit():
                parent[int(child)] = new
            else:
                setattr(parent, child, new)
            fixed.append((mname, str(w.dtype).replace("torch.", "")))
        except Exception as e:
            failed.append((mname, str(e)))

    if fixed:
        log_print("[FP32-FIX] Desde el repo NF4 (seccion precision_critical): {} modulos "
                  "/ From the NF4 repo: {} modules".format(len(fixed), len(fixed)), flush=True)
        for n, d in fixed:
            log_print("  [FP32-FIX] {:<44} -> {}".format(n, d), flush=True)
    if skipped_in_block:
        log_print("[FP32-FIX][WARN] {} precision_critical tensors live INSIDE the transformer "
                  "blocks and were SKIPPED: dequantizing them would pin them in VRAM for all "
                  "blocks and shrink the resident block count. Re-export without them. / {} "
                  "tensores de precision_critical estan DENTRO de los bloques y se han "
                  "OMITIDO: dequantizarlos los fijaria en VRAM para todos los bloques y "
                  "reduciria los bloques residentes. Re-exporta sin ellos.".format(
                      len(skipped_in_block), len(skipped_in_block)), flush=True)
        for n in skipped_in_block[:10]:
            log_print("  [FP32-FIX][SKIP] {}".format(n), flush=True)
    for n, e in failed[:10]:
        log_print("  [FP32-FIX][WARN] {}: {}".format(n, e), flush=True)
    return len(fixed)


def repair_precision_critical_modules(transformer, orig_dir):
    """Recarga desde el checkpoint ORIGINAL los modulos que nunca debieron pasar por NF4.
    Reload from the ORIGINAL checkpoint the modules that should never have gone through NF4.

    El modelo declara `_keep_in_fp32_modules`; ademas `norm_out`, `context_embedder` y
    `token_refiner` forman la ruta de modulacion. En `MiniMaxH3AdaLayerNormOut` la salida
    nace de una cancelacion entre `norm(x)*(1+scale)` y `shift`: el error de NF4 la rompe y
    deja el `shift` en crudo, produciendo una salida enorme e independiente de la entrada.
    The model declares `_keep_in_fp32_modules`; `norm_out`, `context_embedder` and
    `token_refiner` also sit on the modulation path. In `MiniMaxH3AdaLayerNormOut` the output
    comes from a cancellation between `norm(x)*(1+scale)` and `shift`: NF4 error breaks it and
    leaves the raw `shift`, giving a huge, input-independent output.
    """
    # ------------------------------------------------------------------
    # FUENTE 1 (preferida): la seccion `precision_critical` DENTRO del repo NF4.
    # El repo NF4 debe ser autosuficiente: nadie deberia necesitar el checkpoint
    # original de 163 GB para entrenar.
    # SOURCE 1 (preferred): the `precision_critical` section INSIDE the NF4 repo.
    # The NF4 repo must be self-contained: nobody should need the 163 GB original.
    # ------------------------------------------------------------------
    n = _repair_from_nf4_index(transformer)
    if n > 0:
        log_print("[FP32-FIX] {} modulos reparados desde la seccion 'precision_critical' del "
                  "repo NF4. NO se toca el checkpoint original. / {} modules repaired from the "
                  "NF4 repo's 'precision_critical' section. The original checkpoint is not "
                  "touched.".format(n, n), flush=True)
        return n

    # ------------------------------------------------------------------
    # Sin carpeta original configurada no hay respaldo posible: se sale en silencio
    # en vez de soltar el aviso de 163 GB, que solo aplica si el usuario la ha puesto.
    # With no original folder configured there is no fallback: return quietly instead
    # of printing the 163 GB warning, which only applies if the user set the path.
    # ------------------------------------------------------------------
    if not orig_dir:
        log_print("=" * 78, flush=True)
        log_print("[FP32-FIX][WARN] El repo NF4 NO trae seccion 'precision_critical' y "
                  "original_transformer_dir esta vacio: NO se repara nada. / The NF4 repo has "
                  "no 'precision_critical' section and original_transformer_dir is empty: "
                  "nothing is repaired.", flush=True)
        log_print("[FP32-FIX][WARN] ARREGLO: ejecuta una vez "
                  "`5b_export_nonlinear_NF4.py --precision_critical` para meter esa seccion en "
                  "el repo NF4. NO hace falta el checkpoint de 163 GB para entrenar. / FIX: run "
                  "`5b_export_nonlinear_NF4.py --precision_critical` once to add that section "
                  "to the NF4 repo. The 163 GB checkpoint is NOT needed for training.",
                  flush=True)
        log_print("=" * 78, flush=True)
        return 0

    # ------------------------------------------------------------------
    # AVISO FUERTE: el respaldo abre los shards del checkpoint ORIGINAL (163 GB) con
    # safe_open, que los mapea en memoria. Recorrerlos mete decenas de GB en la cache
    # de paginas del SO y DESALOJA los ~18 GB de pesos NF4 aparcados en RAM, que son
    # justo los que el block swap tiene que leer decenas de veces por paso. A partir
    # de ahi cada "carga de bloque desde CPU" pasa a ser una lectura de disco y el
    # entrenamiento se vuelve entre 10x y 20x mas lento.
    # STRONG WARNING: the fallback memory-maps the ORIGINAL 163 GB checkpoint shards.
    # Walking them floods the OS page cache and EVICTS the ~18 GB of parked NF4 weights
    # that the block swap must read dozens of times per step, turning every block load
    # into a disk read and making training 10-20x slower.
    # ------------------------------------------------------------------
    log_print("=" * 78, flush=True)
    log_print("[FP32-FIX][WARN] The NF4 repo has NO 'precision_critical' section. / El repo "
              "NF4 NO trae seccion 'precision_critical'.", flush=True)
    log_print("[FP32-FIX][WARN] Falling back to the ORIGINAL 163 GB checkpoint. This memory-maps "
              "its shards, floods the OS page cache and can evict the parked NF4 weights the "
              "block swap reads every step -> training can become 10-20x slower. / Cayendo al "
              "checkpoint ORIGINAL de 163 GB. Esto mapea sus shards, satura la cache de paginas "
              "y puede desalojar los pesos NF4 aparcados que el block swap lee cada paso -> el "
              "entrenamiento puede volverse 10-20x mas lento.", flush=True)
    log_print("[FP32-FIX][WARN] FIX: run `5b_export_nonlinear_NF4.py --precision_critical` once, "
              "or set fp32_repair_enabled=false. / ARREGLO: ejecuta una vez "
              "`5b_export_nonlinear_NF4.py --precision_critical`, o pon "
              "fp32_repair_enabled=false.", flush=True)
    log_print("=" * 78, flush=True)
    ram_stats("antes del respaldo fp32 / before fp32 fallback")

    if not os.path.isdir(orig_dir):
        log_print("[FP32-FIX] Carpeta original no encontrada / original dir not found: {}"
                  .format(os.path.abspath(orig_dir)), flush=True)
        return 0

    # dtype exigido por modulo / dtype required per module
    _fp32 = tuple(getattr(type(transformer), "_keep_in_fp32_modules", None)
                  or ["proj_in", "audio_proj_in", "time_embedder",
                      "proj_out", "audio_proj_out", "rope"])
    _bf16 = ("norm_out", "context_embedder", "token_refiner")

    _wmap = {}
    for _ix in ("diffusion_pytorch_model.safetensors.index.json",
                "model.safetensors.index.json"):
        _p = os.path.join(orig_dir, _ix)
        if os.path.isfile(_p):
            with open(_p, "r", encoding="utf-8") as _f:
                _wmap = (json.load(_f) or {}).get("weight_map") or {}
            break
    if not _wmap:
        log_print("[FP32-FIX] Sin weight_map en {} / no weight_map".format(orig_dir), flush=True)
        return 0

    def _target_dtype(name):
        if any(name == p or name.startswith(p + ".") for p in _fp32):
            return torch.float32
        if any(name == p or name.startswith(p + ".") for p in _bf16):
            return torch.bfloat16
        return None

    _open = {}
    fixed, failed = [], []
    try:
        for name, mod in list(transformer.named_modules()):
            if not isinstance(mod, torch.nn.Linear) and mod.__class__.__name__ not in (
                    "Linear4bit", "CastingLinear"):
                continue
            # Mismo blindaje que en _repair_from_nf4_index: nada de dentro de los bloques.
            # Same guard as in _repair_from_nf4_index: nothing inside the blocks.
            if name.startswith(("transformer_blocks.", "blocks.")):
                continue
            _dt = _target_dtype(name)
            if _dt is None:
                continue
            wkey = name + ".weight"
            if wkey not in _wmap:
                continue
            try:
                _sh = _wmap[wkey]
                _sp = os.path.join(orig_dir, _sh)
                if _sp not in _open:
                    _open[_sp] = safe_open(_sp, framework="pt", device="cpu").__enter__()
                _f = _open[_sp]
                _w = _f.get_tensor(wkey).to(_dt)
                _b = None
                bkey = name + ".bias"
                if bkey in _wmap:
                    _sp2 = os.path.join(orig_dir, _wmap[bkey])
                    if _sp2 not in _open:
                        _open[_sp2] = safe_open(_sp2, framework="pt", device="cpu").__enter__()
                    if bkey in _open[_sp2].keys():
                        _b = _open[_sp2].get_tensor(bkey).to(_dt)

                new = torch.nn.Linear(_w.shape[1], _w.shape[0], bias=_b is not None)
                new.weight = torch.nn.Parameter(_w, requires_grad=False)
                if _b is not None:
                    new.bias = torch.nn.Parameter(_b, requires_grad=False)
                new.requires_grad_(False)

                parent, child = get_parent_module(transformer, name)
                if child.isdigit():
                    parent[int(child)] = new
                else:
                    setattr(parent, child, new)
                fixed.append((name, str(_dt).replace("torch.", "")))
            except Exception as e:
                failed.append((name, str(e)))
    finally:
        for _h in _open.values():
            try:
                _h.__exit__(None, None, None)
            except Exception:
                pass

    ram_stats("despues del respaldo fp32 / after fp32 fallback")
    log_print("[FP32-FIX] Modulos recargados sin NF4 / modules reloaded without NF4: {}"
              .format(len(fixed)), flush=True)
    for n, d in fixed:
        log_print("  [FP32-FIX] {:<44} -> {}".format(n, d), flush=True)
    for n, e in failed[:10]:
        log_print("  [FP32-FIX][WARN] {}: {}".format(n, e), flush=True)
    return len(fixed)


def load_transformer_from_nf4(nf4_cache_dir):
    _mem_bytes = {"quantized": 0, "unquantized": 0, "other": 0}

    cache_dir = _find_transformer_cache_dir(nf4_cache_dir)
    if cache_dir is None:
        raise FileNotFoundError(
            "No se encontró la caché NF4 del transformer en: {}".format(nf4_cache_dir)
        )

    log_print("[NF4] Cache dir detectado: {}".format(cache_dir), flush=True)

    config_path = os.path.join(cache_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError("Falta config.json en {}".format(cache_dir))

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    cls_name = config_dict.get("_class_name", "MiniMaxH3Transformer3DModel")
    log_print("[NF4] Clase transformer: {}".format(cls_name), flush=True)

    transformer_cls = resolve_transformer_class(cls_name)

    log_print("[NF4] Instanciando transformer vacío en META device...", flush=True)
    transformer = _instantiate_transformer_on_meta(transformer_cls, config_dict)
    transformer.requires_grad_(False)

    index_path = _find_nf4_index_path(cache_dir)
    log_print("[NF4] Index NF4: {}".format(index_path), flush=True)

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    quantized = index.get("quantized", {})
    unquantized = index.get("unquantized", {})
    other = index.get("other", {})
    aliases = index.get("aliases", {})

    weights_dir = os.path.join(cache_dir, "weights")
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError("No existe la carpeta weights en {}".format(cache_dir))

    log_print("[NF4] Capas NF4: {}".format(len(quantized)), flush=True)
    log_print("[NF4] Capas BF16: {}".format(len(unquantized)), flush=True)
    log_print("[NF4] Tensores no-Lineales: {}".format(len(other)), flush=True)
    log_print("[NF4] Aliases: {}".format(len(aliases)), flush=True)

    skip_other_names = set()
    for name in quantized.keys():
        skip_other_names.add(name + ".weight")
        skip_other_names.add(name + ".bias")

    for name in unquantized.keys():
        skip_other_names.add(name + ".weight")
        skip_other_names.add(name + ".bias")

    replaced = 0

    # ------------------------------------------------------------------
    # CAPAS NF4: CARGA DIRECTA A CPU
    # ------------------------------------------------------------------
    log_print(
        "[NF4-LOAD] Cargando capas NF4 directamente a CPU (sin CUDA intermedio).",
        flush=True,
    )

    _nf4_total = len(quantized)
    _nf4_done = 0
    _nf4_t0 = time.time()

    for name, info in quantized.items():
        _nf4_done += 1
        # Aviso de progreso cada 50 capas. Va por print() directo, no por log_print(),
        # para que se vea SIEMPRE aunque debug_log esté apagado: esta carga tarda
        # minutos y sin señal parece que el proceso se ha colgado.
        # Progress notice every 50 layers. Uses print() rather than log_print() so it is
        # ALWAYS visible even with debug logging off: this load takes minutes and without
        # feedback it looks like the process has hung.
        if _nf4_done == 1 or _nf4_done % 50 == 0 or _nf4_done == _nf4_total:
            _el = time.time() - _nf4_t0
            _eta = (_el / max(1, _nf4_done)) * (_nf4_total - _nf4_done)
            print(
                "[NF4-LOAD] Loading layer {}/{} ({:.0f}%) | elapsed {:.0f}s | ETA {:.0f}s"
                " / Cargando capa {}/{} ({:.0f}%) | transcurrido {:.0f}s | ETA {:.0f}s".format(
                    _nf4_done, _nf4_total, 100.0 * _nf4_done / max(1, _nf4_total), _el, _eta,
                    _nf4_done, _nf4_total, 100.0 * _nf4_done / max(1, _nf4_total), _el, _eta,
                ),
                flush=True,
            )

        filepath = os.path.join(weights_dir, info["file"])
        if not os.path.exists(filepath):
            raise FileNotFoundError("No existe peso NF4: {}".format(filepath))

        parent, child_name = get_parent_module(transformer, name)

        with safe_open(filepath, framework="pt", device="cpu") as f:
            weight_data = f.get_tensor("weight")

            bias_data = None
            if info.get("bias", False) and "bias" in f.keys():
                bias_data = f.get_tensor("bias")

            qs_dict = {}
            for key in f.keys():
                if key.startswith("quant_state."):
                    qs_dict[key[len("quant_state."):]] = f.get_tensor(key)

            try:
                meta = f.metadata() or {}
            except Exception:
                meta = {}

            for mkey, mval in meta.items():
                if mkey.startswith("qs_"):
                    clean_key = mkey[3:]
                    try:
                        qs_dict[clean_key] = json.loads(mval)
                    except Exception:
                        qs_dict[clean_key] = mval

        _mem_bytes["quantized"] += weight_data.numel() * weight_data.element_size()
        for _qt in qs_dict.values():
            if torch.is_tensor(_qt):
                _mem_bytes["quantized"] += _qt.numel() * _qt.element_size()

        if bias_data is not None:
            _mem_bytes["quantized"] += bias_data.numel() * bias_data.element_size()

        compress = any(str(k).startswith("nested_") for k in qs_dict.keys())

        try:
            new_layer = _new_linear4bit_empty(
                int(info["in_features"]),
                int(info["out_features"]),
                bias=(bias_data is not None),
                quant_type=info.get("quant_type", "nf4"),
                compress_statistics=compress,
            )
        except TypeError:
            new_layer = _new_linear4bit_empty(
                int(info["in_features"]),
                int(info["out_features"]),
                bias=(bias_data is not None),
                quant_type=info.get("quant_type", "nf4"),
            )

        # CARGAR DIRECTAMENTE EN CPU.
        try:
            new_weight = Params4bit.from_prequantized(
                data=weight_data,
                quantized_stats=qs_dict,
                requires_grad=False,
                device="cpu",
                module=new_layer,
            )
        except TypeError:
            try:
                new_weight = Params4bit.from_prequantized(
                    data=weight_data,
                    quantized_stats=qs_dict,
                    requires_grad=False,
                    device="cpu",
                )
            except TypeError:
                new_weight = Params4bit.from_prequantized(
                    data=weight_data,
                    quantized_stats=qs_dict,
                    requires_grad=False,
                )

        new_layer.weight = new_weight

        if bias_data is not None:
            new_layer.bias = torch.nn.Parameter(
                bias_data.to(dtype=torch.bfloat16),
                requires_grad=False,
            )

        if child_name.isdigit():
            parent[int(child_name)] = new_layer
        else:
            setattr(parent, child_name, new_layer)

        replaced += 1

    gc.collect()

    log_print("[NF4] Capas NF4 reconstruidas: {}".format(replaced), flush=True)

    # ------------------------------------------------------------------
    # CAPAS BF16 NO CUANTIZADAS: A CPU
    # ------------------------------------------------------------------
    for name, info in unquantized.items():
        filepath = os.path.join(weights_dir, info["file"])
        if not os.path.exists(filepath):
            continue

        parent, child_name = get_parent_module(transformer, name)

        with safe_open(filepath, framework="pt", device="cpu") as f:
            weight = f.get_tensor("weight")

            bias = None
            if info.get("bias", False) and "bias" in f.keys():
                bias = f.get_tensor("bias")

        layer = CastingLinear(
            int(info["in_features"]),
            int(info["out_features"]),
            bias=(bias is not None),
            device="meta",
            dtype=torch.bfloat16,
        )

        layer.weight = torch.nn.Parameter(
            weight.to(dtype=torch.bfloat16),
            requires_grad=False,
        )

        _mem_bytes["unquantized"] += layer.weight.numel() * layer.weight.element_size()

        if bias is not None:
            layer.bias = torch.nn.Parameter(
                bias.to(dtype=torch.bfloat16),
                requires_grad=False,
            )
            _mem_bytes["unquantized"] += layer.bias.numel() * layer.bias.element_size()

        if child_name.isdigit():
            parent[int(child_name)] = layer
        else:
            setattr(parent, child_name, layer)

    needed_real_for_aliases = set(aliases.values())
    kept_tensors_for_aliases = {}

    # ------------------------------------------------------------------
    # OTROS TENSORES: A CPU
    # ------------------------------------------------------------------
    if other:
        log_print(
            "[NF4] Cargando {} tensores no-Lineales desde index...".format(len(other)),
            flush=True,
        )

        file_map = {}
        for tensor_name, tinfo in other.items():
            fname = tinfo.get("file", None)
            if not fname:
                continue
            if fname not in file_map:
                file_map[fname] = []
            file_map[fname].append(tensor_name)

        for fname, tensor_names in file_map.items():
            fpath = os.path.join(weights_dir, fname)
            if not os.path.exists(fpath):
                continue

            with safe_open(fpath, framework="pt", device="cpu") as f:
                available_keys = set(f.keys())

                for tensor_name in tensor_names:
                    if tensor_name not in available_keys:
                        continue

                    if tensor_name in skip_other_names:
                        continue

                    if tensor_name in aliases:
                        continue

                    tensor = f.get_tensor(tensor_name)

                    if tensor.is_floating_point():
                        tensor = tensor.to(torch.bfloat16)

                    _mem_bytes["other"] += tensor.numel() * tensor.element_size()

                    try:
                        parent, child_name = get_parent_module(transformer, tensor_name)

                        if child_name in getattr(parent, "_buffers", {}):
                            parent.register_buffer(child_name, tensor)
                        elif child_name in getattr(parent, "_parameters", {}):
                            parent.register_parameter(
                                child_name,
                                torch.nn.Parameter(tensor, requires_grad=False),
                            )
                        else:
                            setattr(parent, child_name, tensor)
                    except Exception as e:
                        log_print(
                            "[WARN] No se pudo asignar tensor {}: {}".format(tensor_name, e),
                            flush=True,
                        )

                    if tensor_name in needed_real_for_aliases:
                        kept_tensors_for_aliases[tensor_name] = tensor

                    del tensor

    else:
        if os.path.isdir(weights_dir):
            other_files = sorted(
                f for f in os.listdir(weights_dir)
                if f.lower().startswith("other-") and f.lower().endswith(".safetensors")
            )

            if other_files:
                log_print(
                    "[NF4] Index sin 'other', pero se encontraron {} archivos other-*.safetensors.".format(
                        len(other_files)
                    ),
                    flush=True,
                )

                for fname in other_files:
                    fpath = os.path.join(weights_dir, fname)

                    with safe_open(fpath, framework="pt", device="cpu") as f:
                        for tensor_name in f.keys():
                            if tensor_name in skip_other_names:
                                continue

                            if tensor_name in aliases:
                                continue

                            tensor = f.get_tensor(tensor_name)

                            if tensor.is_floating_point():
                                tensor = tensor.to(torch.bfloat16)

                            _mem_bytes["other"] += tensor.numel() * tensor.element_size()

                            try:
                                parent, child_name = get_parent_module(transformer, tensor_name)

                                if child_name in getattr(parent, "_buffers", {}):
                                    parent.register_buffer(child_name, tensor)
                                elif child_name in getattr(parent, "_parameters", {}):
                                    parent.register_parameter(
                                        child_name,
                                        torch.nn.Parameter(tensor, requires_grad=False),
                                    )
                                else:
                                    setattr(parent, child_name, tensor)
                            except Exception as e:
                                log_print(
                                    "[WARN] No se pudo asignar tensor {}: {}".format(tensor_name, e),
                                    flush=True,
                                )

                            if tensor_name in needed_real_for_aliases:
                                kept_tensors_for_aliases[tensor_name] = tensor

                            del tensor

    # ------------------------------------------------------------------
    # RoPE: reconstruir inv_freq ANTES de materializar nada.
    # Un inv_freq relleno de ceros mata el RoPE del DiT sin dar error.
    # ------------------------------------------------------------------
    rope_fixed, rope_failed = rebuild_dit_rope_buffers(transformer)
    if rope_failed and STRICT_META_LOAD:
        raise RuntimeError(
            "[ROPE-DiT] No se pudo reconstruir {} buffer(s) inv_freq: {}. "
            "Continuar los rellenaría con ceros y el RoPE del transformer quedaría "
            "desactivado.".format(len(rope_failed), rope_failed[:5])
        )

    # ------------------------------------------------------------------
    # Auditoría ANTES de rellenar. Rellenar en silencio es exactamente el fallo
    # que dejó el text encoder con embed_tokens a ceros durante semanas.
    # ------------------------------------------------------------------
    p_norm, p_other, bufs = audit_meta_tensors(transformer)

    log_print("[NF4] META pendientes -> normas: {} | otros pesos: {} | buffers: {}".format(
        len(p_norm), len(p_other), len(bufs)), flush=True)

    for n in (p_other[:10] + bufs[:10]):
        log_print("  META (crítico): {}".format(n), flush=True)
    for n in p_norm[:10]:
        log_print("  META (norma, se rellenaría con 1.0): {}".format(n), flush=True)

    if (p_other or bufs) and STRICT_META_LOAD:
        raise RuntimeError(
            "[NF4] {} peso(s) entrenados y {} buffer(s) siguen en META. Rellenarlos con "
            "ceros produciría un modelo roto que entrena sin errores y no aprende nada. "
            "Primeros: {} | Ejecuta 5b_export_nonlinear_NF4.py sobre la carpeta del "
            "transformer, o pon strict_meta_load=false para forzar (NO recomendado).".format(
                len(p_other), len(bufs), (p_other + bufs)[:8]
            )
        )

    if p_norm and STRICT_META_LOAD_NORMS:
        raise RuntimeError(
            "[NF4] {} peso(s) de normalización siguen en META y se rellenarían con 1.0, "
            "que NO son los valores entrenados. Primeros: {} | Ejecuta "
            "5b_export_nonlinear_NF4.py sobre la carpeta del transformer.".format(
                len(p_norm), p_norm[:8]
            )
        )

    log_print("[NF4] Materializando tensores META restantes...", flush=True)
    materialized = materialize_meta_tensors(transformer, device="cpu")

    if materialized:
        log_print("[WARN] Se materializaron {} tensores META.".format(len(materialized)), flush=True)
        for n in materialized[:20]:
            log_print("  META materialized: {}".format(n), flush=True)
        if len(materialized) > 20:
            log_print("  ... y {} más".format(len(materialized) - 20), flush=True)
    else:
        log_print("[NF4] No quedaban tensores META.", flush=True)

    if aliases:
        log_print("[NF4] Resolviendo aliases...", flush=True)

        for alias_name, real_name in aliases.items():
            target = kept_tensors_for_aliases.get(real_name, None)

            if target is None:
                try:
                    real_param = None

                    for pname, p in transformer.named_parameters():
                        if pname == real_name:
                            real_param = p
                            break

                    if real_param is None:
                        for bname, b in transformer.named_buffers():
                            if bname == real_name:
                                real_param = b
                                break

                    target = real_param
                except Exception:
                    target = None

            if target is None:
                log_print(
                    "[WARN] Alias {} no pudo resolverse porque falta {}".format(
                        alias_name,
                        real_name,
                    ),
                    flush=True,
                )
                continue

            try:
                parent, child_name = get_parent_module(transformer, alias_name)

                if child_name in getattr(parent, "_buffers", {}):
                    parent.register_buffer(child_name, target)
                elif child_name in getattr(parent, "_parameters", {}):
                    parent.register_parameter(
                        child_name,
                        torch.nn.Parameter(target, requires_grad=False),
                    )
                else:
                    setattr(parent, child_name, target)
            except Exception as e:
                log_print("[WARN] No se pudo asignar alias {}: {}".format(alias_name, e), flush=True)

    del kept_tensors_for_aliases

    # ------------------------------------------------------------------
    # RESTAURAR NO-CONVERT: DEQUANTIZAR Y DEJAR EN CPU
    # ------------------------------------------------------------------
    restored = []
    for name, module in list(transformer.named_modules()):
        if isinstance(module, Linear4bit) and should_restore_non_convert_module(name):
            if dequantize_linear4bit_module(transformer, name, target_device="cpu"):
                restored.append(name)

    log_print("[NF4] Módulos dequantizados (no-convert): {}".format(len(restored)), flush=True)
    for n in restored[:20]:
        log_print("  RESTORED: {}".format(n), flush=True)
    if len(restored) > 20:
        log_print("  ... y {} más".format(len(restored) - 20), flush=True)

    # Recargar sin NF4 los módulos de precisión crítica (ver _keep_in_fp32_modules).
    # Reload the precision-critical modules without NF4 (see _keep_in_fp32_modules).
    if FP32_REPAIR_ENABLED:
        # .get() y no [] : si alguien comenta la clave en DEFAULTS esto ya no revienta.
        # .get() not [] : commenting the key out of DEFAULTS no longer raises KeyError.
        repair_precision_critical_modules(
            transformer,
            str(cfg_get("original_transformer_dir",
                        DEFAULTS.get("original_transformer_dir", "")) or "").strip(),
        )

    verified = 0
    for _, module in transformer.named_modules():
        if isinstance(module, Linear4bit):
            if (
                getattr(module.weight, "bnb_quantized", False)
                and getattr(module.weight, "quant_state", None) is not None
            ):
                verified += 1

    # ------------------------------------------------------------------
    # CONTRA QUE SE COMPARA `verified`.
    #
    # Antes se comparaba con `replaced`, y eso era un FALSO POSITIVO garantizado:
    # `replaced` se cuenta al reconstruir las capas NF4, pero DESPUES corren
    # restore_non_convert_modules() y repair_precision_critical_modules(), que
    # dequantizan a proposito unos cuantos Linear4bit y los sustituyen por
    # nn.Linear. Al llegar aqui ya no existen como Linear4bit, asi que
    # verified < replaced SIEMPRE, y saltaba un aviso alarmante sobre algo que
    # es el funcionamiento correcto.
    #
    # Lo que de verdad hay que comprobar es que TODA capa Linear4bit que siga
    # existiendo este bien cuantizada. La diferencia con `replaced` es
    # informativa: son las que se dequantizaron aposta.
    #
    # `verified` used to be compared against `replaced`, which was a guaranteed
    # false positive: the fp32 repair intentionally dequantizes some Linear4bit
    # layers afterwards, so verified < replaced ALWAYS. What matters is that
    # every Linear4bit still present is properly quantized.
    # ------------------------------------------------------------------
    present = sum(1 for _, m in transformer.named_modules() if isinstance(m, Linear4bit))
    dequantized_on_purpose = replaced - present

    log_print("[NF4] Rebuilt NF4 layers / Capas NF4 reconstruidas: {}".format(replaced),
              flush=True)
    log_print("[NF4] Still 4-bit / Siguen en 4-bit: {} | verified / verificadas: {} | "
              "intentionally dequantized / dequantizadas a proposito: {}".format(
                  present, verified, dequantized_on_purpose), flush=True)

    if replaced == 0:
        raise RuntimeError(
            "No NF4 layer was rebuilt. Check index.json / config_nf4.json and the "
            "weights folder. / No se reconstruyo ninguna capa NF4. Revisa "
            "index.json / config_nf4.json y la carpeta weights."
        )

    if present and verified != present:
        # Esto SI es un problema real: hay Linear4bit sin quant_state.
        # This IS a real problem: there are Linear4bit layers with no quant_state.
        log_print(
            "[NF4][WARN] {} of {} Linear4bit layers have no valid quant_state. / {} de "
            "{} capas Linear4bit no tienen un quant_state valido.".format(
                present - verified, present, present - verified, present),
            flush=True,
        )

    log_print("[OK] Transformer cargado en CPU desde caché NF4.", flush=True)

    total_mem_gb = sum(_mem_bytes.values()) / 1e9

    log_print("=" * 80, flush=True)
    log_print("[VRAM] Desglose ESTIMADO de los pesos del transformer base:", flush=True)
    log_print("  - Capas NF4 (4-bit)       : {:.2f} GB".format(_mem_bytes["quantized"] / 1e9), flush=True)
    log_print("  - Capas BF16 sin cuantizar: {:.2f} GB".format(_mem_bytes["unquantized"] / 1e9), flush=True)
    log_print("  - Tensores no-lineales    : {:.2f} GB".format(_mem_bytes["other"] / 1e9), flush=True)
    log_print("  - TOTAL estimado          : {:.2f} GB".format(total_mem_gb), flush=True)
    log_print("=" * 80, flush=True)

    return transformer


# =============================================================================
# LoRA TARGETS
# =============================================================================

def inspect_transformer_for_lora(transformer):
    """
    Inspección diagnóstica del transformer cargado.
    Detecta automáticamente TODOS los grupos de bloques (blocks, transformer_blocks,
    token_refiner.blocks, etc.) y muestra los módulos lineales encontrados.
    """
    log_print("=" * 80, flush=True)
    log_print("[LORA DISCOVERY] INSPECCIÓN DE ESTRUCTURA REAL DEL TRANSFORMER", flush=True)
    log_print("=" * 80, flush=True)

    all_names = [name for name, _ in transformer.named_modules()]

    # Detectar todos los grupos de bloques numéricos
    # Un grupo es un prefijo que termina en .N. donde N es un número
    import re
    block_pattern = re.compile(r'^(.*?\.)(\d+)\.(.+)$')

    groups = {}  # prefix -> {block_indices, suffixes}
    for name in all_names:
        m = block_pattern.match(name)
        if m:
            prefix = m.group(1)  # e.g., "blocks.", "token_refiner.blocks."
            idx = int(m.group(2))
            suffix = m.group(3)
            if prefix not in groups:
                groups[prefix] = {"indices": set(), "suffixes": {}}
            groups[prefix]["indices"].add(idx)
            groups[prefix]["suffixes"][suffix] = groups[prefix]["suffixes"].get(suffix, 0) + 1

    if not groups:
        log_print("[LORA DISCOVERY] ¡ALERTA! No se detectaron bloques numéricos.", flush=True)
        log_print("[LORA DISCOVERY] Muestra de named_modules():", flush=True)
        for name in sorted(all_names)[:30]:
            log_print("  {}".format(name), flush=True)
        log_print("=" * 80, flush=True)
        return

    for prefix, data in sorted(groups.items()):
        indices = sorted(data["indices"])
        n = len(indices)
        log_print("[LORA DISCOVERY] Grupo: '{}' | Bloques: {} | Rango: {}-{}".format(
            prefix, n, indices[0], indices[-1]
        ), flush=True)

        # Mostrar sufijos que aparecen en >= n/2 bloques (probablemente módulos lineales)
        common_suffixes = {s: c for s, c in data["suffixes"].items() if c >= n // 2}
        for suffix, count in sorted(common_suffixes.items(), key=lambda x: x[1], reverse=True)[:12]:
            log_print("  {:<45} | {:>3} instancias".format(suffix, count), flush=True)

        # Ejemplos de rutas completas del primer bloque
        example_idx = indices[0]
        examples = [n for n in all_names if n.startswith("{}{}.".format(prefix, example_idx))]
        log_print("  Ejemplos bloque {}:".format(example_idx), flush=True)
        for ex in sorted(examples)[:8]:
            log_print("    {}".format(ex), flush=True)

    log_print("=" * 80, flush=True)


def discover_lora_targets(transformer):
    """
    Descubre los módulos LoRA de forma AUTO-ADAPTATIVA.
    Detecta automáticamente la convención de nombres REAL del transformer cargado,
    soportando cualquier prefijo de bloques (blocks, transformer_blocks,
    token_refiner.blocks, etc.) y cualquier esquema de atención/MLP.
    """
    targets = []
    skip_first_n = int(LORA_SKIP_FIRST_N_BLOCKS)

    import re
    all_names = [name for name, _ in transformer.named_modules()]
    module_map = dict(transformer.named_modules())

    # ------------------------------------------------------------------
    # 1) Detectar todos los grupos de bloques numéricos
    # ------------------------------------------------------------------
    block_pattern = re.compile(r'^(.*?\.)(\d+)\.(.+)$')
    groups = {}  # prefix -> {indices, suffixes}

    for name in all_names:
        m = block_pattern.match(name)
        if m:
            prefix = m.group(1)
            idx = int(m.group(2))
            suffix = m.group(3)
            if prefix not in groups:
                groups[prefix] = {"indices": set(), "suffixes": {}}
            groups[prefix]["indices"].add(idx)
            groups[prefix]["suffixes"][suffix] = groups[prefix]["suffixes"].get(suffix, 0) + 1

    if not groups:
        raise RuntimeError(
            "LoRA: no se detectaron bloques numéricos en named_modules(). "
            "Revisa la estructura del transformer cargado."
        )

    log_print("[LoRA] Grupos de bloques detectados: {}".format(len(groups)), flush=True)
    for prefix, data in sorted(groups.items()):
        indices = sorted(data["indices"])
        log_print("  '{}' -> {} bloques ({}-{})".format(
            prefix, len(indices), indices[0], indices[-1]
        ), flush=True)

    # ------------------------------------------------------------------
    # 2) Para cada grupo, detectar esquema de atención y MLP
    # ------------------------------------------------------------------
    for prefix, data in sorted(groups.items()):
        # El token_refiner es la ruta de TEXTO. Entrenar LoRA ahi consume rango y VRAM
        # y luego el cargador de inferencia normalmente ignora esas keys: capacidad
        # entrenada y tirada. Se entrena solo el backbone.
        # token_refiner is the TEXT path; inference loaders usually ignore those keys.
        _has_backbone = any("refiner" not in p.lower() for p in groups)
        if LORA_EXCLUDE_REFINER and _has_backbone and "refiner" in prefix.lower():
            log_print("[LoRA] Grupo '{}' OMITIDO (token_refiner; lora_exclude_refiner=true)"
                      .format(prefix), flush=True)
            continue

        indices = sorted(data["indices"])
        n_blocks = len(indices)
        example_idx = indices[0]
        example_prefix = "{}{}.".format(prefix, example_idx)
        example_names = [n for n in all_names if n.startswith(example_prefix)]

        # Detectar esquema de atención
        has_qkv_proj = any(".attn.qkv_proj" in n for n in example_names)
        has_to_qkv   = any(".attn.to_qkv"   in n for n in example_names)
        has_to_q     = any(".attn.to_q"     in n for n in example_names)
        has_to_k     = any(".attn.to_k"     in n for n in example_names)
        has_to_v     = any(".attn.to_v"     in n for n in example_names)

        has_out_proj = any(".attn.out_proj" in n for n in example_names)
        has_to_out_0 = any(".attn.to_out.0" in n for n in example_names)

        # Detectar esquema MLP/FF
        has_mlp_fc1 = any(".mlp.fc1"       in n for n in example_names)
        has_mlp_fc2 = any(".mlp.fc2"       in n for n in example_names)
        has_ff_0    = any(".ff.net.0.proj" in n for n in example_names)
        has_ff_2    = any(".ff.net.2"      in n for n in example_names)

        # Construir sufijos para este grupo
        group_suffixes = []

        # QKV
        if has_to_q and has_to_k and has_to_v:
            group_suffixes.extend([".attn.to_q", ".attn.to_k", ".attn.to_v"])
            attn_scheme = "diffusers separado (to_q/to_k/to_v)"
        elif has_qkv_proj:
            group_suffixes.append(".attn.qkv_proj")
            attn_scheme = "original/qkv_proj"
        elif has_to_qkv:
            group_suffixes.append(".attn.to_qkv")
            attn_scheme = "diffusers fused (to_qkv)"
        else:
            log_print("[LoRA] Grupo '{}': esquema de atención NO reconocido.".format(prefix), flush=True)
            continue  # Saltar este grupo

        # Out
        if has_out_proj:
            group_suffixes.append(".attn.out_proj")
            out_scheme = "original (out_proj)"
        elif has_to_out_0:
            group_suffixes.append(".attn.to_out.0")
            out_scheme = "diffusers (to_out.0)"
        else:
            log_print("[LoRA] Grupo '{}': proyección de salida NO detectada.".format(prefix), flush=True)
            continue

        # MLP (solo si no es LORA_ONLY_ATTN)
        if LORA_ONLY_ATTN:
            mlp_scheme = "omitido (LORA_ONLY_ATTN)"
        else:
            if has_mlp_fc1 and has_mlp_fc2:
                group_suffixes.extend([".mlp.fc1", ".mlp.fc2"])
                mlp_scheme = "original (mlp.fc1/mlp.fc2)"
            elif has_ff_0 and has_ff_2:
                group_suffixes.extend([".ff.net.0.proj", ".ff.net.2"])
                mlp_scheme = "diffusers FeedForward"
            else:
                log_print("[LoRA] Grupo '{}': MLP/FF NO detectado.".format(prefix), flush=True)
                mlp_scheme = "no detectado"

        log_print(
            "[LoRA] Grupo '{}': attn={}, out={}, mlp={}".format(
                prefix, attn_scheme, out_scheme, mlp_scheme
            ),
            flush=True,
        )

        # Recolectar targets de este grupo
        for name in all_names:
            if not name.startswith(prefix):
                continue
            m = block_pattern.match(name)
            if not m or m.group(1) != prefix:
                continue
            block_idx = int(m.group(2))
            if skip_first_n > 0 and block_idx < skip_first_n:
                continue
            if any(name.endswith(suffix) for suffix in group_suffixes):
                targets.append(name)

    targets = list(dict.fromkeys(targets))

    if not targets:
        raise RuntimeError(
            "LoRA: no se encontraron targets válidos. Revisa la estructura del transformer."
        )

    # Verificación de existencia
    missing = [name for name in targets if name not in module_map]
    if missing:
        raise RuntimeError(
            "LoRA: {} targets no existen en named_modules(). Primeros: {}".format(
                len(missing), missing[:10]
            )
        )

    # Resumen
    log_print("[LoRA] Total targets: {} módulos".format(len(targets)), flush=True)
    for t in targets[:16]:
        log_print("  [LoRA-TARGET] {}".format(t), flush=True)
    if len(targets) > 16:
        log_print("  ... y {} más".format(len(targets) - 16), flush=True)

    return targets


# =============================================================================
# PROMPT / TEXT CONDITIONING (desde caché)
# =============================================================================

def load_prompt_structure(cache_dir, prefix):
    path = os.path.join(cache_dir, "{}_structure.json".format(prefix))
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        structure = json.load(f)

    def recurse(node):
        if node["type"] == "tensor":
            return torch.load(
                os.path.join(cache_dir, node["file"]),
                map_location="cpu",
                weights_only=True,
            )
        if node["type"] == "dict":
            return {k: recurse(v) for k, v in node["items"].items()}
        if node["type"] == "tuple":
            return tuple(recurse(v) for v in node["items"])
        if node["type"] == "list":
            return [recurse(v) for v in node["items"]]
        return node.get("value")

    return recurse(structure)


def flatten_tensors(obj, prefix="root"):
    result = []

    if torch.is_tensor(obj):
        result.append((prefix, obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            result.extend(flatten_tensors(v, "{}.{}".format(prefix, k)))
    elif isinstance(obj, (tuple, list)):
        for i, v in enumerate(obj):
            result.extend(flatten_tensors(v, "{}.{}".format(prefix, i)))

    return result


def get_prompt_pair(result):
    tensors = flatten_tensors(result)

    if not tensors:
        raise RuntimeError("La caché de prompt no contiene tensores.")

    if (
        isinstance(result, (tuple, list))
        and len(result) >= 2
        and torch.is_tensor(result[0])
    ):
        return (
            result[0],
            result[1] if torch.is_tensor(result[1]) else None,
        )

    return (
        tensors[0][1],
        tensors[1][1] if len(tensors) > 1 else None,
    )


# Ultimo cache_info.json leido. Lo rellena verify_cache_compatibility().
# Last cache_info.json read, filled in by verify_cache_compatibility().
_CACHE_INFO = {}


def verify_cache_compatibility(cache_dir):
    """Rechaza cachés generadas por un pre-cache roto / Reject caches from a broken pre-cache.

    Las versiones < 3 salieron de un text encoder con embed_tokens y RMSNorms a ceros y el
    RoPE desactivado: los embeddings guardados son una constante, no conditioning.
    """
    info_path = os.path.join(cache_dir, "cache_info.json")
    if not os.path.exists(info_path):
        log_print("[CACHE] No hay cache_info.json; no se puede verificar la versión. "
                  "Si esta caché es anterior al pre-cache v3, REGENÉRALA.", flush=True)
        return {}

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f) or {}

    # Se memoriza para la cabecera del LoRA: la resolucion y el numero de frames
    # los decidio el pre-cache, el entrenador no los conoce de ninguna otra forma
    # y son justo lo que hace falta para reproducir un entrenamiento.
    # Memorised for the LoRA header: the resolution and frame count were decided
    # by the pre-cache, the trainer has no other way of knowing them, and they are
    # exactly what is needed to reproduce a run.
    _CACHE_INFO.clear()
    _CACHE_INFO.update(info)

    version = int(info.get("version", 0) or 0)
    log_print("[CACHE] formato={} version={} | encoding='{}'".format(
        info.get("format", "?"), version, info.get("prompt_encoding", "?")), flush=True)

    if version < MIN_CACHE_VERSION:
        raise RuntimeError(
            "[CACHE] {} es de version {} y se exige >= {}. Las cachés anteriores se "
            "generaron con un text encoder roto (embed_tokens a ceros, RoPE desactivado): "
            "los embeddings no son conditioning real. Bórrala y re-ejecuta el script 1."
            .format(os.path.abspath(cache_dir), version, MIN_CACHE_VERSION)
        )

    # Que hay en la cache. Antes esto era un WARN fijo en cuanto num_frames no
    # fuera 1, heredado de cuando solo existian imagenes: con una cache de clips
    # saltaba siempre diciendo algo falso, y un aviso que salta siempre deja de
    # leerse. Ahora informa, y solo avisa cuando la geometria es imposible.
    #
    # What the cache holds. This used to be a fixed WARN whenever num_frames was
    # not 1, left over from when only images existed: with a clip cache it fired
    # every time saying something untrue, and a warning that always fires stops
    # being read. It now reports, and only warns when the geometry is impossible.
    contenido = str(info.get("content", "image"))
    nf = info.get("num_frames", None)

    if contenido == "video" and nf:
        # H3 exige 17n+5 fotogramas de pixel -> 5n+2 latentes. Cualquier otro
        # numero no lo pudo producir el VAE, asi que la cache viene de otro sitio
        # o se genero con una version anterior del script 1.
        # H3 requires 17n+5 pixel frames -> 5n+2 latents. Any other count could
        # not have come out of the VAE, so the cache is from elsewhere or from an
        # older version of script 1.
        valido = int(nf) >= 5 and (int(nf) - 5) % 17 == 0
        log_print("[CACHE] Clips: {} | {} pixel frames -> {} latent frames{} / "
                  "Clips: {} | {} fotogramas de pixel -> {} latentes{}".format(
                      info.get("num_clips", "?"), nf, 5 * ((int(nf) - 5) // 17) + 2,
                      "" if valido else "  [!]",
                      info.get("num_clips", "?"), nf, 5 * ((int(nf) - 5) // 17) + 2,
                      "" if valido else "  [!]"), flush=True)
        if not valido:
            log_print("[CACHE][WARN] num_frames={} is not 17n+5, which is the only geometry "
                      "the H3 VAE can produce. Regenerate the pre-cache. / num_frames={} no "
                      "es 17n+5, la unica geometria que puede producir el VAE de H3. "
                      "Regenera la pre-cache.".format(nf, nf), flush=True)
    elif contenido == "mixed":
        log_print("[CACHE] Mixed dataset: {} images + {} clips. / Dataset mixto: {} imagenes "
                  "+ {} clips.".format(info.get("num_images", "?"), info.get("num_clips", "?"),
                                       info.get("num_images", "?"), info.get("num_clips", "?")),
                  flush=True)

    ac = info.get("audio_latent_channels", None)
    if ac is not None:
        log_print("[CACHE] audio_latent_channels={} (referencia H3 = 32)".format(ac), flush=True)

    return info


def load_cached_entries(cache_dir, audio_channels):
    verify_cache_compatibility(cache_dir)
    entries = []

    for filename in sorted(os.listdir(cache_dir)):
        if not filename.endswith("_video_latent.pt"):
            continue

        base = filename[:-len("_video_latent.pt")]
        video_path = os.path.join(cache_dir, filename)
        audio_path = os.path.join(cache_dir, "{}_audio_latent.pt".format(base))
        prompt_structure_path = os.path.join(cache_dir, "{}_prompt_structure.json".format(base))

        if not os.path.exists(audio_path):
            continue
        if not os.path.exists(prompt_structure_path):
            continue

        video_latent = torch.load(video_path, map_location="cpu", weights_only=True)
        if video_latent is None:
            continue

        audio_latent = torch.load(audio_path, map_location="cpu", weights_only=True)
        if audio_latent is None:
            bsz = video_latent.shape[0] if video_latent.ndim == 5 else 1
            audio_latent = torch.zeros((bsz, audio_channels, 1), dtype=torch.bfloat16)

        prompt_result = load_prompt_structure(cache_dir, "{}_prompt".format(base))
        if prompt_result is None:
            continue

        # kind viene del _info.json de la muestra. Interesa uno solo: "audio",
        # que marca una toma sin imagen, donde el latente de video es un unico
        # fotograma negro puesto ahi para no tener que soportar V=0 en el
        # empaquetado, el desempaquetado y las previews. Esa fila NO debe
        # aprenderse: si entra en la loss, el LoRA aprende a generar negro.
        # kind comes from the sample's _info.json. Only one value matters:
        # "audio", marking a take with no picture, where the video latent is a
        # single black frame put there to avoid supporting V=0 across packing,
        # unpacking and previews. That row must NOT be learned: if it enters the
        # loss, the LoRA learns to generate black.
        _kind = "video"
        try:
            with open(os.path.join(cache_dir, "{}_info.json".format(base)),
                      "r", encoding="utf-8") as fh:
                _kind = str(json.load(fh).get("kind", "video") or "video")
        except Exception:
            pass

        entries.append({
            "name": base,
            "kind": _kind,
            "video": video_latent.to(torch.bfloat16),
            "audio": audio_latent.to(torch.bfloat16),
            "prompt": prompt_result,
        })

    if not entries:
        raise RuntimeError("No se encontraron entradas válidas en la caché.")

    return entries


# =============================================================================
# PATCHIFY / TIMESTEP / LOSS
# =============================================================================

def align_video_latent_to_patch(latent, patch_h=1, patch_w=1, patch_t=1):
    if latent.ndim != 5:
        return latent

    B, C, F, H, W = latent.shape

    pt = max(1, int(patch_t))
    ph = max(1, int(patch_h))
    pw = max(1, int(patch_w))

    F = (F // pt) * pt
    H = (H // ph) * ph
    W = (W // pw) * pw

    return latent[:, :, :F, :H, :W].contiguous()


def patch_video_latent(latent, patch_h=1, patch_w=1, patch_t=1):
    if latent.ndim != 5:
        raise RuntimeError("Video latent esperado [B,C,F,H,W].")

    B, C, F, H, W = latent.shape

    pt = max(1, int(patch_t))
    ph = max(1, int(patch_h))
    pw = max(1, int(patch_w))

    F = (F // pt) * pt
    H = (H // ph) * ph
    W = (W // pw) * pw

    latent = latent[:, :, :F, :H, :W]

    x = latent.view(
        B,
        C,
        F // pt,
        pt,
        H // ph,
        ph,
        W // pw,
        pw,
    )

    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(B, -1, C * pt * ph * pw)


def patch_audio_latent(latent):
    if latent.ndim == 2:
        latent = latent.unsqueeze(0)

    if latent.ndim != 3:
        raise RuntimeError("Audio latent esperado [B,C,T] o [C,T].")

    return latent.transpose(1, 2).contiguous()


def unpack_video_latent(tokens, latent_shape, patch_h=1, patch_w=1, patch_t=1):
    B, C, F, H, W = tuple(latent_shape)

    pt = max(1, int(patch_t))
    ph = max(1, int(patch_h))
    pw = max(1, int(patch_w))

    Fp = F // pt
    Hp = H // ph
    Wp = W // pw

    expected_seq = Fp * Hp * Wp
    tokens = tokens[:, :expected_seq, :]

    x = tokens.view(B, Fp, Hp, Wp, C, pt, ph, pw)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)

    return x.reshape(B, C, Fp * pt, Hp * ph, Wp * pw)


def sample_sigmas(batch, device, shift=None, image_tokens=None, generator=None):
    """Niveles de ruido en (0,1) para entrenamiento.

    shift=None (por defecto): logit-normal (sigmoid(randn), la densidad de
    SD3/Flux) con un shift dependiente de la resolución (shift = exp(mu), mu
    lineal en el nº de tokens de imagen: 256 tokens -> 0.5, 6400 -> 1.15).

    OJO CON LA DIRECCIÓN. El comentario anterior decía que esto concentraba el
    muestreo en sigma BAJA; es al revés. Con shift > 1 la transformación
    sigma = s*u/(1+(s-1)*u) empuja sigma HACIA ARRIBA, o sea hacia MÁS ruido.
    Medido sobre 200.000 muestras:

        tokens=256   shift=1.649   sigma media=0.602  mediana=0.623
                     p10=0.314  p90=0.856  frac(sigma<0.1)=0.4%
        tokens=1024  shift=1.788   sigma media=0.618  mediana=0.641
                     p10=0.332  p90=0.866  frac(sigma<0.1)=0.3%

    Es el comportamiento estándar de Flux/SD3 y es probablemente lo que quieres,
    pero solo el ~0,3% del entrenamiento cae por debajo de sigma=0.1. Para sesgar
    de verdad hacia el detalle fino haría falta mu < 0, es decir shift < 1.

    MIND THE DIRECTION. The previous comment claimed this concentrated sampling at
    LOW sigma; it does the opposite. shift > 1 pushes sigma UP, i.e. towards MORE
    noise. This matches Flux/SD3, but only ~0.3% of training lands below sigma=0.1;
    biasing towards fine detail would need mu < 0, i.e. shift < 1.

    shift=<float>: mapeo legacy uniforme-u + shift (sigma = s*u/(1+(s-1)*u)).
    OJO: usar aquí el shift de vídeo del sampler (p.ej. 12.0) entrena casi
    todo el rato a ruido muy alto y da mal parecido — solo para experimentos.
    """
    if shift is None:
        # Logit-normal: sigma = sigmoid(mu + std * N(0,1)).
        #   mu = 0.0  -> mediana 0.50, P(sigma<0.2) = 8%
        #   mu = -0.4 -> mediana 0.40, P(sigma<0.2) = 15%   <- por defecto
        #   mu = -0.8 -> mediana 0.31, P(sigma<0.2) = 24%
        # Mas masa en sigma media-baja = mas presupuesto de entrenamiento en la banda
        # que decide los rasgos de la cara, que es justo lo que un LoRA de identidad
        # necesita. El shift por resolucion (que empuja sigma hacia ARRIBA) queda
        # detras de sigma_resolution_shift porque va en contra de ese objetivo.
        z = torch.randn(batch, device=device, generator=generator)
        base = torch.sigmoid(LOGIT_NORMAL_MU + LOGIT_NORMAL_STD * z)

        if not SIGMA_RESOLUTION_SHIFT:
            return base

        tokens = float(image_tokens or 225)
        mu = 0.5 + (tokens - 256.0) * (1.15 - 0.5) / (6400.0 - 256.0)
        eff_shift = math.exp(mu)
    else:
        eff_shift = float(shift)
        base = torch.rand(batch, device=device, generator=generator)

    if abs(eff_shift - 1.0) < 1e-6:
        return base

    return (eff_shift * base) / (1.0 + (eff_shift - 1.0) * base)


def mse_loss_chunked(pred, target, chunk_elements=None):
    if chunk_elements is None:
        chunk_elements = int(LOSS_CHUNK_ELEMENTS)

    chunk_elements = max(64, int(chunk_elements))

    if pred.numel() == 0:
        return pred.new_zeros((), dtype=torch.float32)

    if pred.numel() <= chunk_elements:
        return F.mse_loss(pred.float(), target.float())

    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    n = pred_flat.numel()

    loss_sum = torch.zeros((), device=pred.device, dtype=torch.float32)

    for start in range(0, n, chunk_elements):
        end = min(start + chunk_elements, n)
        p = pred_flat[start:end].float()
        t = target_flat[start:end].float()
        loss_sum = loss_sum + F.mse_loss(p, t, reduction="sum")
        del p, t

    return loss_sum / float(n)


# =============================================================================
# ÍNDICES EMPAQUETADOS PARA MINIMAX-H3
# =============================================================================
#
# Los `position_ids` (t,h,w) que se pasan al modelo alimentan directamente el RoPE del
# backbone congelado. El checkpoint preentrenado fue entrenado con una convención MUY
# concreta (PackedLayout oficial): coordenadas CONTINUAS normalizadas por área y
# escaladas x32 en (h,w), y en t el vídeo continúa la secuencia justo donde termina el
# texto (origen = text_len), no arranca en 0. Si se le dan al backbone unas posiciones
# distintas (p.ej. índices enteros crudos de la rejilla de parches, o t=0 tanto para
# texto como para vídeo) el RoPE rota los tokens de forma incompatible con lo que el
# modelo aprendió: la atención sigue funcionando y la loss baja, pero la correspondencia
# espacial fina entre el texto (identidad) y cada región de la imagen queda distorsionada
# — el LoRA "hace algo" pero no puede recuperar parecido, porque en inferencia (ComfyUI)
# se usan las posiciones correctas y el LoRA nunca vio ese régimen de atención.
#
# Lo de abajo reproduce esa convención (idéntica en espíritu a la que usa el otro
# port pure-PyTorch verificado: _axis_from_sqrt_area / _frame_grid / _video_t_grid).

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
# FRAME_RESCALE viene del "faithful port" no oficial que nos pasaste al principio;
# no está confirmado contra el código real de diffusers. Para Fp==1 (tu caso, imagen
# suelta) es irrelevante — solo hay un t = origen = text_len, sin ninguna constante
# de por medio. Se deja como fallback aproximado solo por si algún día entrenas video
# multi-frame.
FRAME_RESCALE = 5.0 / 3.0


def _axis_from_sqrt_area(dim, patch, sqrt_area, device):
    ratio = dim / sqrt_area
    n = max(1, dim // patch)
    return (torch.arange(n, dtype=torch.float64, device=device) * (ratio / n)
            + (1.0 - ratio) / 2.0) * 32.0


def _frame_grid(h, w, patch_h, patch_w, device):
    """Coordenadas (h, w) normalizadas por área de UN frame latente, en unidades de
    parche patch_h x patch_w: [(h//patch_h)*(w//patch_w), 2]."""
    area = math.sqrt(h * w)
    hh, ww = torch.meshgrid(
        _axis_from_sqrt_area(h, patch_h, area, device),
        _axis_from_sqrt_area(w, patch_w, area, device),
        indexing="ij",
    )
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)


def _video_t_grid(n, origin, device):
    """n valores de t, arrancando en `origin` (= text_len). Para n==1 (imagen suelta,
    tu caso) es exactamente [origin], sin constantes extra."""
    if n <= 1:
        return torch.tensor([float(origin)], dtype=torch.float64, device=device)
    spans = torch.tensor([FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(n)],
                         dtype=torch.float64, device=device)
    cum = torch.cat([torch.zeros(1, dtype=torch.float64, device=device), spans[:-1].cumsum(0)])
    return float(origin) + cum


def build_minimax_packed_indices(
    B,
    text_len,
    video_len,
    audio_len,
    video_latent_shape,
    patch_t,
    patch_h,
    patch_w,
    device,
    audio_channels=2,
):
    """audio_channels: el audio se empaqueta channel-major (estereo = 2), asi que
    A filas son a_lat = A // audio_channels latentes por canal. Solo se usa para
    colocar las posiciones del audio.
    audio_channels: audio rows are channel-major (stereo = 2)."""
    T = int(text_len)
    V = int(video_len)
    A = int(audio_len)
    S = T + V + A
    width_axis = None

    idx_dtype = torch.long
    pos_dtype = torch.float32   # posiciones continuas, no índices — ver nota arriba

    if T > 0:
        text_indices = torch.arange(0, T, device=device, dtype=idx_dtype)
    else:
        text_indices = torch.empty(0, device=device, dtype=idx_dtype)

    # ------------------------------------------------------------------
    # ORDEN DEL LAYOUT: [texto | audio | video].
    #
    # Es el orden oficial (MiniMaxH3PrepareLayoutStep de diffusers y el
    # pipeline de ai-toolkit): el audio va ENTRE el texto y el video, no
    # detras. Antes estaba como [texto | video | audio].
    #
    # Con A == 0 los dos ordenes son identicos, asi que este cambio NO altera
    # el entrenamiento (que envia cero filas de audio); solo importa para la
    # preview, que ahora si las manda.
    #
    # LAYOUT ORDER: [text | audio | video], the official one. With A == 0 both
    # orders are identical, so this does not change training.
    # ------------------------------------------------------------------
    if A > 0:
        audio_indices = torch.arange(T, T + A, device=device, dtype=idx_dtype)
    else:
        audio_indices = torch.empty(0, device=device, dtype=idx_dtype)

    if V > 0:
        video_indices = torch.arange(T + A, S, device=device, dtype=idx_dtype)
    else:
        video_indices = torch.empty(0, device=device, dtype=idx_dtype)

    text_tags = torch.ones(T, dtype=idx_dtype, device=device)
    audio_tags = torch.full((A,), 2, dtype=idx_dtype, device=device)
    video_tags = torch.zeros(V, dtype=idx_dtype, device=device)

    token_tags = torch.cat([text_tags, audio_tags, video_tags], dim=0).contiguous()
    timestep_indices = torch.zeros(S, dtype=idx_dtype, device=device)

    # texto: eje t = 0..T-1 (cada token en su propia posición), h=w=0
    text_pos = torch.zeros(T, 3, dtype=torch.float64, device=device)
    if T > 0:
        text_pos[:, 0] = torch.arange(T, dtype=torch.float64, device=device)

    if V > 0:
        _, _, F, H, W = video_latent_shape

        pt = max(1, int(patch_t))
        ph = max(1, int(patch_h))
        pw = max(1, int(patch_w))

        Fp = F // pt
        Hp = H // ph
        Wp = W // pw

        if Fp * Hp * Wp == V:
            # rejilla espacial normalizada por área (una por frame, se reutiliza para
            # cada uno) + eje t continuando desde el origen text_len
            frame = _frame_grid(H, W, ph, pw, device)             # [Hp*Wp, 2]
            # El eje de anchura suelto: el audio se clava en sus dos extremos.
            # The bare width axis: audio rows pin to its two extremes.
            width_axis = _axis_from_sqrt_area(W, pw, math.sqrt(H * W), device)
            frame_rows = frame.shape[0]
            t_grid = _video_t_grid(Fp, T, device)                  # [Fp]
            video_pos = torch.empty(V, 3, dtype=torch.float64, device=device)
            for k in range(Fp):
                s0, s1 = k * frame_rows, (k + 1) * frame_rows
                video_pos[s0:s1, 0] = t_grid[k]
                video_pos[s0:s1, 1:] = frame
        else:
            video_pos = torch.zeros(V, 3, dtype=torch.float64, device=device)
    else:
        video_pos = torch.zeros(0, 3, dtype=torch.float64, device=device)

    if A > 0:
        # ------------------------------------------------------------------
        # POSICIONES DEL AUDIO, ahora las OFICIALES.
        #
        # Copiadas de MiniMaxH3PrepareLayoutStep.build_packed_sequence:
        #   - eje t: text_len + arange(a_lat), REPETIDO para cada canal
        #     (las filas son channel-major: todo el canal 0, luego el 1), o sea
        #     que el audio comparte el reloj rotatorio del video;
        #   - eje h: 0, el audio no tiene coordenada de altura;
        #   - eje w: las filas se clavan en los DOS EXTREMOS de la rejilla de
        #     anchura del video: el primer canal en width_grid[0] y el resto en
        #     width_grid[-1].
        # El esquema anterior (t = 0..A-1, h = w = 0) no se parecia a esto y
        # ademas hacia que la primera fila de audio cayera en (0,0,0), justo
        # encima del primer token de texto.
        #
        # OFFICIAL audio positions, copied from diffusers'
        # build_packed_sequence: the audio shares the video's rotary clock and
        # its rows are pinned to the two extremes of the width grid.
        # ------------------------------------------------------------------
        a_lat = max(1, A // max(1, int(audio_channels)))
        audio_pos = torch.zeros(A, 3, dtype=torch.float64, device=device)
        a_time = float(T) + torch.arange(a_lat, dtype=torch.float64, device=device)
        reps = (A + a_lat - 1) // a_lat
        audio_pos[:, 0] = a_time.repeat(reps)[:A]
        if V > 0 and width_axis is not None and width_axis.numel() > 0:
            audio_pos[:a_lat, 2] = float(width_axis[0])
            audio_pos[a_lat:, 2] = float(width_axis[-1])
    else:
        audio_pos = torch.zeros(0, 3, dtype=torch.float64, device=device)

    position_ids = torch.cat([text_pos, audio_pos, video_pos], dim=0).contiguous().to(pos_dtype)

    return (
        timestep_indices,
        token_tags,
        position_ids,
        video_indices,
        audio_indices,
        text_indices,
    )


# =============================================================================
# OPTIMIZACIONES
# =============================================================================

class _DropMessageFilter(logging.Filter):
    """Descarta los registros cuyo mensaje contiene `needle`.
    Drops log records whose message contains `needle`."""

    def __init__(self, needle):
        super().__init__()
        self._needle = needle

    def filter(self, record):
        try:
            return self._needle not in str(record.msg)
        except Exception:
            return True


def silence_diffusers_attention_backend_notice():
    """Calla el aviso "Attention backends are an experimental feature..." de diffusers.

    Lo emite `diffusers.models.modeling_utils` en CADA llamada a
    set_attention_backend, y aqui se prueban varios backends en cadena, asi que
    salia repetido cinco veces al arrancar. No aporta nada: es un aviso de API
    inestable dirigido a quien programa contra diffusers, no al usuario que esta
    entrenando un LoRA. Se filtra solo ESE mensaje, por texto, en vez de bajar el
    nivel del logger entero: cualquier otro aviso de diffusers sigue saliendo.

    Silences the repeated "Attention backends are an experimental feature" notice
    that diffusers emits on every set_attention_backend call. Only that exact
    message is filtered, so every other diffusers warning still comes through.
    """
    needle = "Attention backends are an experimental feature"
    for name in ("diffusers", "diffusers.models.modeling_utils"):
        try:
            logging.getLogger(name).addFilter(_DropMessageFilter(needle))
        except Exception:
            pass


def enable_memory_efficient_attention(transformer):
    silence_diffusers_attention_backend_notice()

    # Nombres REALES de diffusers. Los que habia aqui antes ("sdpa",
    # "torch_sdpa") no existen en ninguna version, asi que este bucle siempre
    # fallaba entero y se caia al fallback sin que nadie lo viera: el mensaje
    # estaba silenciado. No importaba porque el backend por defecto de diffusers
    # ya es `native` (el SDPA de PyTorch), que no materializa la matriz n^2.
    #
    # The REAL diffusers names. The ones here before ("sdpa", "torch_sdpa") exist
    # in no version, so this loop always failed completely and fell through
    # unseen, the message being silenced. It did not matter because diffusers'
    # default backend is already `native` (PyTorch SDPA), which does not
    # materialise the n^2 matrix.
    # SageAttention exige fp16 o bf16 en q/k/v y revienta con un assert si le
    # llegan en fp32:
    #
    #   sageattention/core.py:707
    #   AssertionError: Input tensors must be in dtype of torch.float16 or bfloat16
    #
    # Este entrenador corre con use_autocast=False a proposito, porque el forward
    # de H3 hace sus propios casts y hay modulos que TIENEN que quedarse en fp32
    # (_keep_in_fp32_modules). Con autocast apagado, q/k/v salen en fp32 y Sage no
    # puede usarse. Se detecta ANTES de cargar los 39 GB del modelo: morir en el
    # primer paso despues de cinco minutos de carga no es una forma aceptable de
    # descubrir una incompatibilidad conocida.
    #
    # SageAttention requires fp16/bf16 for q/k/v and asserts on fp32. This trainer
    # runs with use_autocast=False on purpose, so q/k/v come out in fp32 and Sage
    # cannot be used. Detected BEFORE loading the model's 39 GB: dying on the
    # first step after a five-minute load is not an acceptable way to discover a
    # known incompatibility.
    # ------------------------------------------------------------------
    # SAGEATTENTION NO SIRVE PARA ENTRENAR. NO ES UN PROBLEMA DE dtype.
    #
    # El aviso que habia aqui culpaba al dtype: q/k/v salian en fp32 y el kernel
    # exige fp16/bf16. Era verdad pero no era la causa, y encima el aviso no
    # saltaba nunca, porque miraba `not USE_AUTOCAST` y autocast esta ENCENDIDO;
    # q/k/v vuelven a fp32 despues, en attn.norm_q y en el rotary. Por eso el
    # error seguia apareciendo.
    #
    # El fallback tampoco podia funcionar: envolvia set_attention_backend(), que
    # se registra sin protestar. El AssertionError sale mucho despues, dentro del
    # forward, en dispatch_attention_fn -> _sage_attention. Un try/except
    # alrededor del registro no puede atrapar algo que ocurre en otra pila.
    #
    # Y lo de fondo: el paquete sageattention NO TIENE backward. Ni una
    # autograd.Function, ni un save_for_backward, ni un ctx en todo el modulo:
    # son kernels de inferencia. Medido en esta misma instalacion, la salida de
    # sageattn() viene con requires_grad=False y grad_fn=None.
    #
    # Eso convierte el "arreglo" evidente -- castear q/k/v a bf16 -- en una
    # trampa: el entrenamiento NO petaria, porque la rama residual
    # (hidden = hidden + attn(...)) mantiene el grafo vivo. Simplemente los LoRA
    # de to_q, to_k y to_v recibirian gradiente CERO y saldria un LoRA a medio
    # entrenar sin un solo mensaje de error. Mejor negarse aqui, en voz alta.
    #
    # Para previews (inferencia, sin backward) Sage si valdria. No esta hecho.
    #
    # SAGEATTENTION CANNOT TRAIN, AND IT IS NOT A dtype PROBLEM. The old warning
    # blamed the dtype -- q/k/v arrive fp32 and the kernel demands fp16/bf16 --
    # which was true but not the cause, and it never fired anyway because it
    # tested `not USE_AUTOCAST` while autocast is ON (q/k/v go back to fp32 later,
    # in attn.norm_q and the rotary). The fallback could not work either: it
    # wrapped set_attention_backend(), which registers happily; the AssertionError
    # comes much later inside the forward, in dispatch_attention_fn ->
    # _sage_attention, on a different stack. And underneath all that, the
    # sageattention package has NO backward -- not one autograd.Function,
    # save_for_backward or ctx in the whole module; they are inference kernels.
    # Measured on this install, sageattn() returns requires_grad=False, grad_fn
    # None. That makes the obvious "fix" (cast q/k/v to bf16) a trap: training
    # would NOT crash, because the residual branch keeps the graph alive -- the
    # to_q/to_k/to_v LoRAs would just get ZERO gradient and produce a half-trained
    # LoRA with no error message. Better to refuse here, loudly. For previews
    # (inference, no backward) Sage would be fine; that is not wired up.
    # ------------------------------------------------------------------
    if USE_SAGE_ATTENTION:
        log_print("[ATTN][WARN] SageAttention is ON but it CANNOT be used for training: the "
                  "package ships inference kernels with no backward pass (sageattn() returns "
                  "requires_grad=False, grad_fn=None), so the q/k/v LoRAs would silently "
                  "receive zero gradient. Using native attention, which is already "
                  "memory-efficient. / SageAttention esta activada pero NO se puede usar para "
                  "entrenar: el paquete son kernels de inferencia sin backward (sageattn() "
                  "devuelve requires_grad=False y grad_fn=None), asi que los LoRA de q/k/v se "
                  "quedarian sin gradiente y nadie avisaria. Se usa la atencion nativa, que ya "
                  "es eficiente en memoria.", flush=True)
    candidatos = ("native",)

    if hasattr(transformer, "set_attention_backend"):
        for backend in candidatos:
            try:
                transformer.set_attention_backend(backend)
                if backend.startswith("sage"):
                    log_print("[ATTN] SageAttention active ({}). Attention numerics are "
                              "quantized: faster and lighter, but not bit-exact against "
                              "native. / SageAttention activa ({}). La numerica de la "
                              "atencion va cuantizada: mas rapida y ligera, pero no "
                              "identica a la nativa.".format(backend, backend), flush=True)
                else:
                    log_print("[ATTN] Attention backend / Backend de atencion: {}".format(backend),
                              flush=True)
                return
            except Exception as exc:
                if backend.startswith("sage"):
                    log_print("[ATTN][WARN] SageAttention requested but unavailable ({}): {}. "
                              "Falling back. / Se pidio SageAttention pero no esta disponible "
                              "({}): {}. Se usa el siguiente.".format(backend, exc, backend, exc),
                              flush=True)

    try:
        transformer.enable_xformers_memory_efficient_attention()
        log_print("[ATTN] xformers memory efficient attention activado.", flush=True)
        return
    except Exception:
        pass

    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        log_print("[ATTN] Fell back to global PyTorch SDP flags (flash + mem-efficient). "
                  "The model may still use its own attention path. / Se han puesto las "
                  "banderas globales de SDP de PyTorch; el modelo puede seguir usando su "
                  "propia ruta de atencion.", flush=True)
    except Exception:
        log_print("[ATTN][WARN] No memory-efficient attention backend could be enabled. With "
                  "long video sequences the n^2 attention matrix will be materialised and OOM "
                  "is likely. / No se pudo activar ningun backend de atencion eficiente. Con "
                  "secuencias largas de video se materializara la matriz n^2 y el OOM es "
                  "probable.", flush=True)


def enable_gradient_checkpointing_safe(transformer, model):
    enabled = False

    for obj in (model, transformer):
        for fn_name in ("enable_gradient_checkpointing", "gradient_checkpointing_enable"):
            if hasattr(obj, fn_name):
                try:
                    getattr(obj, fn_name)()
                    log_print("[GC] {}.{} activado".format(type(obj).__name__, fn_name), flush=True)
                    enabled = True
                except Exception:
                    pass

    for obj in (model, transformer):
        try:
            obj.gradient_checkpointing = True
            enabled = True
        except Exception:
            pass

    for obj in (model, transformer):
        cfg = getattr(obj, "config", None)
        if cfg is not None:
            try:
                cfg.use_cache = False
            except Exception:
                pass

        try:
            obj.use_cache = False
        except Exception:
            pass

    checkpoint_classes = (
        "TransformerBlock",
        "BasicTransformerBlock",
        "JointTransformerBlock",
        "MiniMaxH3TransformerBlock",
        "SingleTransformerBlock",
        "Transformer2DModel",
        "Transformer3DModel",
    )

    checkpointed = 0

    for module in transformer.modules():
        try:
            module.training = True
        except Exception:
            pass

        cls_name = module.__class__.__name__
        if cls_name not in checkpoint_classes:
            continue

        try:
            module.gradient_checkpointing = True
        except Exception:
            pass

        try:
            module.use_checkpoint = True
        except Exception:
            pass

        try:
            module.use_gradient_checkpointing = True
        except Exception:
            pass

        checkpointed += 1

    for module in transformer.modules():
        for attr in ("past_key_values", "_past_key_values", "kv_cache", "_kv_cache", "cache"):
            if hasattr(module, attr):
                try:
                    setattr(module, attr, None)
                except Exception:
                    pass

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except Exception:
        pass

    try:
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception:
        pass

    try:
        torch.backends.cuda.enable_math_sdp(False)
    except Exception:
        pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    log_print("[GC] {} bloques preparados para checkpoint manual.".format(checkpointed), flush=True)

    return enabled


def _param_size_bytes(p):
    total = 0

    try:
        total += p.data.numel() * p.data.element_size()
    except Exception:
        pass

    qs = getattr(p, "quant_state", None)
    if qs is not None:
        for attr in ("absmax", "code", "nested_absmax", "nested_code", "offset", "state2"):
            t = getattr(qs, attr, None)
            if isinstance(t, torch.Tensor):
                try:
                    total += t.numel() * t.element_size()
                except Exception:
                    pass

    return total


def get_module_size_bytes(module):
    total = 0

    for p in module.parameters(recurse=True):
        total += _param_size_bytes(p)

    for b in module.buffers(recurse=True):
        try:
            total += b.numel() * b.element_size()
        except Exception:
            continue

    return total


def get_block_nontrainable_bytes(module):
    total = 0

    for p in module.parameters(recurse=True):
        if not getattr(p, "requires_grad", False):
            total += _param_size_bytes(p)

    for b in module.buffers(recurse=True):
        try:
            total += b.numel() * b.element_size()
        except Exception:
            pass

    return total


def get_block_nf4_bytes(module):
    total = 0

    for m in module.modules():
        if isinstance(m, Linear4bit):
            for p in m.parameters(recurse=False):
                if not getattr(p, "requires_grad", False):
                    total += _param_size_bytes(p)

    return total


def find_block_lists(root, min_blocks=4):
    candidates = []

    for name, module in root.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) >= min_blocks:
            size_bytes = sum(get_module_size_bytes(m) for m in module)
            candidates.append((name, module, size_bytes))

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


def print_block_diagnostics(transformer):
    log_print(flush=True)
    log_print("=" * 80, flush=True)
    log_print("[DIAG] Listas de bloques (ModuleList) detectadas en el transformer:", flush=True)

    candidates = find_block_lists(transformer)

    if not candidates:
        log_print("  (No se encontró ningún ModuleList con >=4 elementos.)", flush=True)
    else:
        for name, module, size_bytes in candidates:
            per_block_gb = (size_bytes / max(1, len(module))) / 1e9
            log_print(
                "  - '{}': {} bloques | {:.2f} GB total | ~{:.3f} GB/bloque".format(
                    name or "(root)", len(module), size_bytes / 1e9, per_block_gb
                ),
                flush=True,
            )

    log_print("=" * 80, flush=True)
    log_print(flush=True)

    return candidates


def wrap_blocks_with_explicit_checkpoint(block_list, label=""):
    import torch.utils.checkpoint as _cp

    wrapped = 0

    for block in block_list:
        if getattr(block, "_explicit_ckpt_wrapped", False):
            continue
        if getattr(block, "_nf4_swap_ckpt_wrapped", False):
            continue

        orig_forward = block.forward

        def make_checkpointed(orig_fwd):
            def checkpointed_forward(*args, **kwargs):
                if not torch.is_grad_enabled():
                    return orig_fwd(*args, **kwargs)

                kwarg_keys = list(kwargs.keys())
                flat_inputs = list(args) + [kwargs[k] for k in kwarg_keys]
                n_args = len(args)

                def custom_forward(*flat):
                    real_args = flat[:n_args]
                    real_kwargs = dict(zip(kwarg_keys, flat[n_args:]))
                    return orig_fwd(*real_args, **real_kwargs)

                return _cp.checkpoint(
                    custom_forward,
                    *flat_inputs,
                    # Ver checkpoint_use_reentrant en DEFAULTS: con False,
                    # bitsandbytes clava el peso NF4 de CUDA en el grafo hasta
                    # el backward y el block swap no libera nada.
                    # See checkpoint_use_reentrant: with False, bnb pins the
                    # CUDA NF4 weight in the graph and the swap frees nothing.
                    use_reentrant=CHECKPOINT_USE_REENTRANT,
                )

            return checkpointed_forward

        block.forward = make_checkpointed(orig_forward)
        block._explicit_ckpt_wrapped = True
        wrapped += 1

    if wrapped:
        log_print(
            "[CKPT] '{}': {} bloques envueltos con checkpoint explícito.".format(
                label, wrapped
            ),
            flush=True,
        )

    return wrapped


def apply_explicit_checkpointing_to_all_blocks(transformer, min_gb=0.05):
    candidates = find_block_lists(transformer)
    wrapped_prefixes = []
    total_wrapped = 0

    for name, block_list, size_bytes in candidates:
        if size_bytes / 1e9 < min_gb:
            continue

        if any(name == p or name.startswith(p + ".") for p in wrapped_prefixes):
            continue

        n = wrap_blocks_with_explicit_checkpoint(block_list, label=name or "(root)")
        if n:
            wrapped_prefixes.append(name)
            total_wrapped += n

    return total_wrapped


def wrap_block_range_with_explicit_checkpoint(
    block_list,
    start_index=0,
    end_index=None,
    label="",
):
    import torch.utils.checkpoint as _cp

    if end_index is None:
        end_index = len(block_list)

    start_index = max(0, int(start_index))
    end_index = min(int(end_index), len(block_list))

    wrapped = 0

    for idx in range(start_index, end_index):
        block = block_list[idx]

        if getattr(block, "_explicit_ckpt_wrapped", False):
            continue
        if getattr(block, "_nf4_swap_ckpt_wrapped", False):
            continue

        orig_forward = block.forward

        def make_checkpointed(orig_fwd):
            def checkpointed_forward(*args, **kwargs):
                if not torch.is_grad_enabled():
                    return orig_fwd(*args, **kwargs)

                kwarg_keys = list(kwargs.keys())
                flat_inputs = list(args) + [kwargs[k] for k in kwarg_keys]
                n_args = len(args)

                def custom_forward(*flat):
                    real_args = flat[:n_args]
                    real_kwargs = dict(zip(kwarg_keys, flat[n_args:]))
                    return orig_fwd(*real_args, **real_kwargs)

                return _cp.checkpoint(
                    custom_forward,
                    *flat_inputs,
                    # Ver checkpoint_use_reentrant en DEFAULTS: con False,
                    # bitsandbytes clava el peso NF4 de CUDA en el grafo hasta
                    # el backward y el block swap no libera nada.
                    # See checkpoint_use_reentrant: with False, bnb pins the
                    # CUDA NF4 weight in the graph and the swap frees nothing.
                    use_reentrant=CHECKPOINT_USE_REENTRANT,
                )

            return checkpointed_forward

        block.forward = make_checkpointed(orig_forward)
        block._explicit_ckpt_wrapped = True
        wrapped += 1

    if wrapped:
        log_print(
            "[CKPT] '{}': {} bloques envueltos con checkpoint explícito.".format(
                label or "rango", wrapped
            ),
            flush=True,
        )

    return wrapped


def setup_gradient_checkpointing_optimized(transformer, model, offload_info=None):
    for obj in (model, transformer):
        cfg = getattr(obj, "config", None)
        if cfg is not None:
            try:
                cfg.use_cache = False
            except Exception:
                pass

        try:
            obj.use_cache = False
        except Exception:
            pass

    if offload_info is not None and offload_info.get("checkpoint_configured", False):
        return

    if offload_info is None:
        if EXPLICIT_CHECKPOINTING_ENABLED:
            wrapped = apply_explicit_checkpointing_to_all_blocks(transformer)
            if wrapped == 0:
                enable_gradient_checkpointing_safe(transformer, model)
        else:
            enable_gradient_checkpointing_safe(transformer, model)
        return

    block_list = offload_info["block_list"]
    start_index = offload_info["start_index"]

    for obj in (model, transformer):
        try:
            obj.gradient_checkpointing = False
        except Exception:
            pass

    wrap_block_range_with_explicit_checkpoint(
        block_list,
        0,
        start_index,
        label="bloques GPU residentes",
    )

    wrap_nf4_swap_checkpoint_range(
        block_list,
        start_index,
        len(block_list),
        label="bloques NF4 swap",
    )


# =============================================================================
# NF4 BLOCK SWAP JIT
# =============================================================================

def _move_quant_state_to_device(qs, device):
    if qs is None:
        return None

    try:
        moved = qs.to(device)
        if moved is not None:
            return moved
    except Exception:
        pass

    for attr in ("absmax", "code", "nested_absmax", "nested_code", "offset", "state2"):
        t = getattr(qs, attr, None)
        if torch.is_tensor(t) and t.device.type != device:
            setattr(qs, attr, t.to(device))

    return qs


# =============================================================================
# SPILL A DISCO DE BLOQUES NF4 APARCADOS / DISK SPILL FOR PARKED NF4 BLOCKS
# =============================================================================
# Los bloques que no caben en VRAM se aparcan en RAM como tensores anonimos: o
# caben o el proceso muere. Pero los pesos NF4 son de SOLO LECTURA (estan
# congelados; el swap solo los copia a GPU y los descarta), asi que pueden vivir
# en un fichero mapeado en memoria. Entonces el SO los mantiene en cache
# mientras haya sitio y los desaloja bajo presion en vez de matar el proceso.
#
# Parked blocks live in RAM as anonymous tensors: they either fit or the process
# dies. But the NF4 weights are READ-ONLY (frozen; the swap only copies them to
# GPU and discards them), so they can live in a memory-mapped file instead. The
# OS then keeps them cached while there is room and evicts them under pressure
# rather than killing the process.
#
# ram_limit_gb = 0 lo deja apagado y el comportamiento es el de siempre.
# ram_limit_gb = 0 keeps it off and the behaviour is unchanged.
# =============================================================================

_SPILL = {
    "map": None, "path": None, "offset": 0, "size": 0, "dir": None,
    "file_gb": 0.0, "armed": False, "base_used": None,
    "ram_limit": 0, "ram_used": 0, "ram_blocks": 0, "disk_blocks": 0,
}


def spill_active():
    """Configurado (no implica que el fichero exista ya). / Configured."""
    return _SPILL["armed"]


def _system_ram_used_gb():
    """RAM del SISTEMA en uso, en GB.

    Es la metrica correcta para esta decision. El RSS del proceso no vale: en
    Windows el working set lo recorta el SO continuamente, y ademas los pesos
    NF4 aparcados pueden estar respaldados por el fichero del checkpoint, en
    cuyo caso no cuentan como memoria privada del proceso aunque ocupen RAM.
    Lo que de verdad importa es si la MAQUINA se esta quedando sin memoria.

    System RAM in use, in GB. The right metric here: the process RSS is useless
    because Windows trims the working set constantly, and the parked NF4 weights
    may be file-backed, so they do not show as private memory even though they
    occupy RAM. What matters is whether the MACHINE is running out.
    """
    try:
        import psutil
        vm = psutil.virtual_memory()
        return (vm.total - vm.available) / (1024.0 ** 3)
    except Exception:
        return None


def spill_init(ram_limit_gb, out_dir, file_gb=20.0):
    """Arma el spill. El fichero NO se crea hasta que hace falta de verdad.

    ram_limit_gb es el techo de RAM del SISTEMA, no del proceso: es lo que el
    usuario puede comprobar en el Administrador de tareas.

    Arms the spill. The file is NOT created until it is actually needed.
    ram_limit_gb is a SYSTEM RAM ceiling, not a process one: it is what the user
    can check in Task Manager.
    """
    if PARK_MODE == "ram":
        return False
    if PARK_MODE != "disk" and (ram_limit_gb is None or float(ram_limit_gb) <= 0):
        return False

    _SPILL.update({
        "map": None, "path": None, "offset": 0, "size": 0,
        "dir": out_dir or ".", "file_gb": float(file_gb),
        "ram_limit": float(ram_limit_gb),
        "ram_used": 0, "ram_blocks": 0, "disk_blocks": 0, "armed": True,
        "base_used": None,
    })
    if PARK_MODE == "disk":
        log_print("[SPILL] ON | mode DISK: every parked block goes to the mapped "
                  "file. / modo DISCO: todos los bloques aparcados van al fichero "
                  "mapeado.", flush=True)
        return True

    used = _system_ram_used_gb()
    log_print("[SPILL] ON | system RAM ceiling: {:.1f} GB (now at {:.1f} GB). The "
              "spill file is only created if a block does not fit. / Techo de RAM "
              "del sistema: {:.1f} GB (ahora en {:.1f} GB). El fichero de spill "
              "solo se crea si algun bloque no cabe.".format(
                  float(ram_limit_gb), used or 0.0,
                  float(ram_limit_gb), used or 0.0), flush=True)
    return True


def _spill_open():
    """Crea y mapea el fichero la primera vez que hace falta de verdad."""
    if _SPILL["map"] is not None:
        return True
    try:
        os.makedirs(_SPILL["dir"], exist_ok=True)
        path = os.path.join(_SPILL["dir"], "nf4_park_spill.bin")
        size = int(_SPILL["file_gb"] * (1024 ** 3))

        # Sin sitio no se puede: mejor seguir en RAM que reventar a mitad del
        # aparcado. Pedimos 1 GB extra de margen para no dejar el disco a cero.
        # No room means no spill: better to stay in RAM than to blow up halfway
        # through parking. We ask for 1 GB of slack so the disk is not left dry.
        free = shutil.disk_usage(_SPILL["dir"]).free
        if free < size + (1024 ** 3):
            log_print("[SPILL] Not enough free space on {}: {:.1f} GB free, "
                      "{:.1f} GB needed. Staying in RAM. / No hay espacio libre "
                      "en {}: {:.1f} GB libres, hacen falta {:.1f} GB. Todo se "
                      "queda en RAM.".format(
                          _SPILL["dir"], free / 1024 ** 3,
                          (size + 1024 ** 3) / 1024 ** 3,
                          _SPILL["dir"], free / 1024 ** 3,
                          (size + 1024 ** 3) / 1024 ** 3), flush=True)
            _SPILL["armed"] = False
            return False
        with open(path, "wb") as fh:
            fh.truncate(size)
        _SPILL["map"] = torch.from_file(path, shared=True, size=size,
                                        dtype=torch.uint8)
        _SPILL["path"] = path
        _SPILL["size"] = size
    except Exception as e:
        log_print("[SPILL] Could not create the spill file, everything stays in "
                  "RAM: {} / No se pudo crear el fichero de spill, todo se queda "
                  "en RAM: {}".format(e, e), flush=True)
        _SPILL["armed"] = False
        return False

    log_print("[SPILL] Spill file created: {} ({:.1f} GB) / Fichero de spill "
              "creado: {} ({:.1f} GB)".format(
                  _SPILL["path"], _SPILL["file_gb"],
                  _SPILL["path"], _SPILL["file_gb"]), flush=True)
    return True


def spill_write(t):
    """Copia t al fichero mapeado y devuelve una vista con los mismos bytes."""
    t = t.detach().contiguous()
    nbytes = t.numel() * t.element_size()
    off = (_SPILL["offset"] + 63) & ~63          # alineado a 64 B / 64 B aligned
    if off + nbytes > _SPILL["size"]:
        raise RuntimeError("spill file full / fichero de spill lleno")

    flat = t.reshape(-1)
    if flat.dtype != torch.uint8:
        flat = flat.view(torch.uint8)
    _SPILL["map"][off:off + nbytes].copy_(flat)
    _SPILL["offset"] = off + nbytes
    return _SPILL["map"][off:off + nbytes].view(t.dtype).reshape(t.shape)


# Cuanto crece la RAM del proceso DESPUES de aparcar, sin contar los bloques
# aparcados: buffers de torch, contexto CUDA, workspaces de cuBLAS. Medido en
# vivo: al aparcar el proceso estaba en 1,6 GiB y en meseta llega a 17,5 GiB con
# 5,6 GiB de bloques, o sea ~10,3 GiB de crecimiento. Hay que sumarlo, porque en
# el momento de decidir todavia no se ha gastado.
#
# How much process RAM grows AFTER parking, excluding the parked blocks: torch
# buffers, the CUDA context, cuBLAS workspaces. Measured live: 1.6 GiB at park
# time, 17.5 GiB at plateau with 5.6 GiB of blocks -> ~10.3 GiB of growth. It
# must be added, because at decision time it has not been spent yet.
POST_PARK_RAM_GROWTH_GIB = 10.3


def spill_park_block(linear_modules):
    """Decide si este bloque se queda en RAM o va al fichero mapeado.

    En modo "auto" proyecta el consumo FINAL en vez de mirar el actual: cuando
    se aparca, el proceso solo ha gastado una fraccion de lo que gastara en
    meseta, asi que mirar el momento presente nunca dispararia el techo.

    In "auto" mode it projects the FINAL usage instead of reading the current
    one: at park time the process has spent only a fraction of what it will use
    at plateau, so looking at the present moment would never trip the ceiling.
    """
    if not spill_active():
        return False

    nbytes = sum(m.weight.numel() * m.weight.element_size()
                 for m in linear_modules if m.weight is not None)

    if PARK_MODE != "disk":
        if _SPILL["base_used"] is None:
            _SPILL["base_used"] = _system_ram_used_gb()

        projected = _SPILL["base_used"]
        if projected is None:
            # Sin psutil no podemos proyectar nada: mejor no tocar el disco.
            # Without psutil we cannot project: better not to touch the disk.
            _SPILL["ram_used"] += nbytes
            _SPILL["ram_blocks"] += 1
            return False

        projected += (POST_PARK_RAM_GROWTH_GIB
                      + (_SPILL["ram_used"] + nbytes) / (1024.0 ** 3))

        if projected <= _SPILL["ram_limit"]:
            _SPILL["ram_used"] += nbytes
            _SPILL["ram_blocks"] += 1
            return False

    if not _spill_open():
        return False

    for m in linear_modules:
        if m.weight is None or m.weight.device.type != "cpu":
            continue
        try:
            view = spill_write(m.weight.data)
        except Exception as e:
            log_print("[SPILL] Write failed, this block stays in RAM: {} / "
                      "Fallo al escribir, este bloque se queda en RAM: "
                      "{}".format(e, e), flush=True)
            return False
        m.weight.data = view
        m._nf4_disk_view = view

    _SPILL["disk_blocks"] += 1
    return True


def spill_cleanup():
    """Suelta el mapeo y borra el fichero. El spill no sobrevive a la ejecucion:
    son ~13 GB de disco que no sirven para nada una vez terminado.

    Releases the mapping and deletes the file. The spill does not outlive the
    run: it is ~13 GB of disk that is useless once training is over."""
    path = _SPILL.get("path")
    _SPILL["map"] = None
    _SPILL["path"] = None
    _SPILL["armed"] = False
    if not path:
        return
    for _ in range(3):
        try:
            if os.path.isfile(path):
                os.remove(path)
            return
        except Exception:
            import time as _t
            _t.sleep(0.5)      # Windows tarda en soltar el mapeo / mapping lag
    log_print("[SPILL] Could not delete {}, remove it by hand. / No se pudo "
              "borrar {}, borralo a mano.".format(path, path), flush=True)


def spill_report():
    if not spill_active():
        return
    used = _system_ram_used_gb() or 0.0
    rss = _ram_rss_gb() or 0.0
    log_print("[SPILL] Parked blocks: {} in RAM ({:.2f} GB), {} on disk "
              "({:.2f} GB). System RAM {:.1f} GB / ceiling {:.1f} GB "
              "(process RSS {:.1f} GB). / Bloques aparcados: {} en RAM "
              "({:.2f} GB), {} en disco ({:.2f} GB). RAM del sistema {:.1f} GB "
              "/ techo {:.1f} GB (RSS del proceso {:.1f} GB).".format(
                  _SPILL["ram_blocks"], _SPILL["ram_used"] / 1024 ** 3,
                  _SPILL["disk_blocks"], _SPILL["offset"] / 1024 ** 3,
                  used, _SPILL["ram_limit"], rss,
                  _SPILL["ram_blocks"], _SPILL["ram_used"] / 1024 ** 3,
                  _SPILL["disk_blocks"], _SPILL["offset"] / 1024 ** 3,
                  used, _SPILL["ram_limit"], rss),
              flush=True)


def _new_linear4bit_empty(in_features, out_features, bias=False, **kwargs):
    """Crea un Linear4bit SIN inicializar sus pesos.

    nn.Linear.__init__ reserva la matriz completa y la rellena con kaiming
    uniform. Aqui ese relleno se tira dos lineas despues, al asignar el peso NF4
    real, pero cuesta lo que cuesta: medido sobre los tamanos de H3 salen 0,359 s
    por capa, y con 370 capas son los 133 segundos que tardaba el arranque.
    Practicamente TODO el tiempo de carga era generar numeros aleatorios para
    descartarlos.

    Construyendolo en el dispositivo `meta` no se reserva memoria ni se rellena
    nada: la misma capa de 5376x28672 pasa de 0,638 s a 0,0003 s. Despues se
    devuelve a CPU con to_empty(), que reserva el hueco sin escribir en el.

    El sesgo SI se materializa: es pequeno y el llamante puede copiarle datos.

    Creates a Linear4bit WITHOUT initializing its weights. nn.Linear.__init__
    allocates the full matrix and fills it with kaiming uniform; that fill is
    discarded two lines later when the real NF4 weight is assigned, but it costs
    0.359 s per layer on H3's shapes -- the 133 seconds the load used to take.
    Almost the entire load was generating random numbers to throw away.
    Building on the `meta` device allocates and fills nothing. The bias IS
    materialised: it is small and the caller may copy data into it.
    """
    with torch.device("meta"):
        layer = Linear4bit(in_features, out_features, bias=bias,
                           compute_dtype=torch.bfloat16, **kwargs)
    # to_empty reserva el almacenamiento en CPU sin inicializarlo.
    # to_empty allocates CPU storage without initialising it.
    return layer.to_empty(device="cpu")


def _move_params4bit_to_device(p, device, disk_view=None):
    """
    Mueve un Params4bit sin permitir que bitsandbytes materialice el peso
    cuantizado como FP32 al hacer .to("cpu").

    IMPORTANTE:
    - CUDA -> CPU: copiamos SOLO el tensor almacenado y el quant_state.
      No usamos Params4bit.to("cpu"), que puede provocar una expansión
      temporal enorme de memoria RAM.
    - CPU -> CUDA: usamos .to("cuda") sobre la representación ya compacta.
    """
    device = str(device)

    if p.device.type == device:
        qs = getattr(p, "quant_state", None)
        if qs is not None:
            try:
                p.quant_state = _move_quant_state_to_device(qs, device)
            except Exception:
                pass
        return p

    # ------------------------------------------------------------------
    # CUDA -> CPU: NO usar p.to("cpu")
    # ------------------------------------------------------------------
    if device == "cpu":
        old_data = p.data
        old_qs = getattr(p, "quant_state", None)

        # Copiamos únicamente la representación que realmente está
        # almacenada en Params4bit. No hacemos ninguna dequantización.
        if disk_view is not None:
            # Bloque respaldado en fichero: reutilizamos la vista mapeada en vez
            # de copiar a RAM anonima. Los pesos NF4 estan congelados, asi que
            # los bytes del fichero siguen siendo validos. Ademas ahorra la copia.
            # File-backed block: reuse the mapped view instead of copying into
            # anonymous RAM. The NF4 weights are frozen, so the bytes on file are
            # still valid. It also saves the copy.
            cpu_data = disk_view
        else:
            cpu_data = old_data.detach().to(device="cpu")

        try:
            new_p = Params4bit(
                cpu_data,
                requires_grad=False,
                quant_type=getattr(p, "quant_type", "nf4"),
                blocksize=getattr(p, "blocksize", 64),
                compress_statistics=getattr(p, "compress_statistics", True),
            )
        except Exception:
            # Compatibilidad con versiones de bitsandbytes con una firma
            # distinta de Params4bit.
            try:
                new_p = Params4bit(cpu_data, requires_grad=False)
            except Exception:
                new_p = torch.nn.Parameter(cpu_data, requires_grad=False)

        # Restaurar metadatos de cuantización.
        for attr in (
            "quant_type",
            "blocksize",
            "compress_statistics",
            "bnb_quantized",
            "quant_storage",
        ):
            if hasattr(p, attr):
                try:
                    setattr(new_p, attr, getattr(p, attr))
                except Exception:
                    pass

        if old_qs is not None:
            try:
                new_p.quant_state = _move_quant_state_to_device(old_qs, "cpu")
            except Exception:
                try:
                    new_p.quant_state = old_qs
                except Exception:
                    pass

        return new_p

    # ------------------------------------------------------------------
    # CPU -> CUDA: la representación CPU ya está compactada.
    # ------------------------------------------------------------------
    try:
        new_p = p.to(device)
    except Exception:
        new_p = Params4bit(p.data.to(device), requires_grad=False)
        for attr in (
            "quant_state",
            "quant_type",
            "blocksize",
            "compress_statistics",
            "bnb_quantized",
            "quant_storage",
        ):
            if hasattr(p, attr):
                try:
                    setattr(new_p, attr, getattr(p, attr))
                except Exception:
                    pass

    qs = getattr(new_p, "quant_state", getattr(p, "quant_state", None))
    if qs is not None:
        try:
            new_p.quant_state = _move_quant_state_to_device(qs, device)
        except Exception:
            pass

    return new_p


def _move_linear4bit_to_device(module, device):
    """
    Mueve un Linear4bit evitando module.to("cpu").

    En CPU sustituimos explícitamente el Params4bit por una copia compacta.
    Esto evita la posible dequantización implícita de bitsandbytes y elimina
    el principal pico de RAM del swap.
    """
    device = str(device)

    if module.weight is not None and module.weight.device.type != device:
        old_weight = module.weight

        # ------------------------------------------------------------------
        # La ruta original tiraba el tensor CPU al subir el bloque a la GPU, asi
        # que al bajarlo habia que rehacerlo con una copia D2H de ~0,28 GiB mas
        # una reserva nueva. Con 47 bloques swapeados y dos pasadas por paso son
        # 94 descargas: ~26 GB de PCIe y 94 ciclos de reservar/liberar 287 MB.
        #
        # Los pesos NF4 estan CONGELADOS: los bytes que ya hay en CPU siguen
        # siendo validos siempre. Guardamos esa copia ("home") y la reutilizamos
        # al volver. Medido: 16,58 -> 9,61 s/it y 51 -> 31 GB de RAM.
        #
        # The original path dropped the CPU tensor when uploading a block, so
        # coming back needed a ~0.28 GiB D2H copy plus a fresh allocation. With
        # 47 swapped blocks and two passes per step that is 94 downloads: ~26 GB
        # of PCIe traffic and 94 allocate/free cycles of 287 MB.
        #
        # The NF4 weights are FROZEN: the bytes already in CPU stay valid
        # forever. We keep that copy ("home") and reuse it on the way back.
        # ------------------------------------------------------------------
        if NF4_CPU_HOME and device == "cuda" and old_weight.device.type == "cpu":
            if getattr(module, "_nf4_cpu_home", None) is None:
                module._nf4_cpu_home = old_weight.data

        home = getattr(module, "_nf4_disk_view", None)
        if home is None and NF4_CPU_HOME:
            home = getattr(module, "_nf4_cpu_home", None)

        new_weight = _move_params4bit_to_device(
            old_weight, device,
            disk_view=(home if device == "cpu" else None),
        )
        module.weight = new_weight
        del old_weight

    if module.bias is not None and module.bias.device.type != device:
        module.bias.data = module.bias.data.to(device)

    try:
        w = module.weight
        qs = getattr(w, "quant_state", None)
        if qs is not None:
            w.quant_state = _move_quant_state_to_device(qs, device)
    except Exception:
        pass


# Bloque NF4 swap actualmente residente en CUDA. Solo puede haber uno
# activo a la vez dentro del mecanismo JIT.
_ACTIVE_NF4_SWAP_BLOCK = None


def _ram_rss_gb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024.0 ** 3)
    except Exception:
        return None


def _cleanup_after_nf4_swap(device, label=""):
    """Optional allocator cleanup; never runs gc/empty_cache per swap.

    The training path must not force Python GC or CUDA allocator cleanup for
    every Linear4bit movement because that serializes the CPU/GPU pipeline.
    Debug logging is deliberately reduced to one line per block.
    """
    if DEBUG_TRAINING and label:
        ram = _ram_rss_gb()
        if ram is not None:
            pass  # [SWAP]/[VRAM-LIMIT] log removed / log eliminado


def _move_nf4_swap_modules(mods, device):
    """Move a whole block's NF4 modules once, with no per-layer cleanup."""
    if not mods:
        return

    # Idempotent: do nothing if the block is already in the requested device.
    target = str(device)
    if all(m.weight.device.type == target for m in mods if m.weight is not None):
        return

    for m in mods:
        _move_linear4bit_to_device(m, target)



def park_nf4_block(block):
    linear_modules = []
    linear_param_ids = set()

    for m in block.modules():
        if isinstance(m, Linear4bit):
            linear_modules.append(m)
            for p in m.parameters(recurse=False):
                linear_param_ids.add(id(p))

    for p in block.parameters(recurse=True):
        if id(p) in linear_param_ids:
            continue

        if p.device.type != "cuda":
            p.data = p.data.to("cuda")

    for b in block.buffers(recurse=True):
        if b.device.type != "cuda":
            b.data = b.data.to("cuda")

    _move_nf4_swap_modules(linear_modules, "cpu")

    for p in block.parameters(recurse=True):
        if getattr(p, "requires_grad", False) and p.device.type != "cuda":
            p.data = p.data.to("cuda")

    # El bloque queda en estado CPU persistente: no se conserva una copia
    # CUDA del NF4 mientras está aparcado. Los módulos se reconstruyen en CUDA
    # solo durante el forward/backward y vuelven inmediatamente a CPU.
    spill_park_block(linear_modules)

    block._nf4_swap_modules = linear_modules
    block._nf4_swap_grad_enabled = False
    block._nf4_swap_persistent_cpu = True
    block._nf4_swap_est_bytes = int(max(get_block_nf4_bytes(block) * 4, get_block_nontrainable_bytes(block)))


def move_non_block_to_cuda(root, block_list):
    block_param_ids = set()
    block_buffer_ids = set()

    for b in block_list:
        for p in b.parameters(recurse=True):
            block_param_ids.add(id(p))

        for buf in b.buffers(recurse=True):
            block_buffer_ids.add(id(buf))

    for name, p in root.named_parameters():
        if id(p) in block_param_ids:
            continue

        if p.device.type == "cuda":
            continue

        if isinstance(p, Params4bit):
            parent, child_name = get_parent_module(root, name)
            new_p = _move_params4bit_to_device(p, "cuda")

            if child_name.isdigit():
                parent[int(child_name)] = new_p
            else:
                setattr(parent, child_name, new_p)
        else:
            p.data = p.data.to("cuda")

    for name, buf in root.named_buffers():
        if id(buf) in block_buffer_ids:
            continue

        if buf.device.type == "cuda":
            continue

        parent, child_name = get_parent_module(root, name)
        moved = buf.to("cuda")

        if child_name.isdigit():
            parent[int(child_name)] = moved
        else:
            if child_name in getattr(parent, "_buffers", {}):
                parent.register_buffer(child_name, moved)
            else:
                setattr(parent, child_name, moved)


def _nf4_swap_required_bytes(module):
    """Estimación conservadora de VRAM necesaria para el bloque swap activo."""
    cached = getattr(module, "_nf4_swap_est_bytes", None)
    if cached is not None:
        return int(cached)

    nf4 = get_block_nf4_bytes(module)
    # Conservador: la representación almacenada es 4-bit; al ejecutarse
    # bitsandbytes puede necesitar una representación mucho mayor en CUDA.
    est = int(max(nf4 * 4, get_block_nontrainable_bytes(module)))
    module._nf4_swap_est_bytes = est
    return est


def _enforce_manual_swap_budget(module):
    """Valida SOLO el presupuesto del bloque swap.

    No se compara contra torch.cuda.memory_allocated(): durante backward esa
    cifra contiene activaciones, gradientes y workspaces que no pertenecen al
    presupuesto de pesos. Hacer esa suma provocaba falsos errores.
    """
    if not torch.cuda.is_available():
        return

    required = _nf4_swap_required_bytes(module)
    allowed = VRAM_SWAP_MAX_BYTES

    if required > allowed:
        # ----------------------------------------------------------------
        # CUATRO DECIMALES, NO DOS.
        #
        # Con dos decimales este mensaje decia "block=1.33 GB | swap
        # allowed=1.33 GB" y fallaba igual: la comparacion real es
        # 1,3328 > 1,3300, pero el redondeo la ocultaba y no habia forma de
        # entender por que. Con cuatro decimales la diferencia se ve, y ademas
        # se dice directamente cual es el minimo que si funciona.
        # FOUR DECIMALS: with two, this printed "block=1.33 | swap allowed=1.33"
        # and still failed, hiding the actual 1.3328 > 1.3300 comparison.
        # ----------------------------------------------------------------
        needed_gb = required / 1e9
        minimum_gb = math.ceil(needed_gb * 100.0) / 100.0
        raise RuntimeError(
            "[ERROR] The NF4 block needs more memory than vram_swap_gb allows. "
            "block needs {:.4f} GB | vram_swap_gb allows {:.4f} GB | short by "
            "{:.4f} GB. Set vram_swap_gb to at least {:.2f}. "
            "(resident budget {:.2f} GB | headroom {:.2f} GB) / "
            "[ERROR] El bloque NF4 necesita mas memoria de la que permite "
            "vram_swap_gb. El bloque necesita {:.4f} GB | vram_swap_gb permite "
            "{:.4f} GB | faltan {:.4f} GB. Pon vram_swap_gb en {:.2f} como "
            "minimo. (budget residente {:.2f} GB | headroom {:.2f} GB)".format(
                needed_gb, allowed / 1e9, needed_gb - allowed / 1e9, minimum_gb,
                VRAM_BUDGET_GB, VRAM_HEADROOM_GB,
                needed_gb, allowed / 1e9, needed_gb - allowed / 1e9, minimum_gb,
                VRAM_BUDGET_GB, VRAM_HEADROOM_GB,
            )
        )


def _nf4_swap_pre_hook(module, args, kwargs):
    global _ACTIVE_NF4_SWAP_BLOCK

    mods = getattr(module, "_nf4_swap_modules", [])
    if mods:
        already_cuda = all(
            m.weight is None or m.weight.device.type == "cuda"
            for m in mods
        )

        if not already_cuda:
            # Evicción estricta: antes de cargar un bloque nuevo, el bloque
            # swap anterior debe estar de nuevo en CPU. Esto evita que los
            # bloques NF4 se acumulen en VRAM durante forward/recompute.
            previous = _ACTIVE_NF4_SWAP_BLOCK
            if previous is not None and previous is not module:
                prev_mods = getattr(previous, "_nf4_swap_modules", [])
                if prev_mods and any(
                    m.weight is not None and m.weight.device.type != "cpu"
                    for m in prev_mods
                ):
                    _move_nf4_swap_modules(prev_mods, "cpu")
                    if DEBUG_TRAINING:
                        _cleanup_after_nf4_swap(
                            "cpu", "evict block -> CPU before next block"
                        )
                previous._nf4_swap_active = False

            _enforce_manual_swap_budget(module)
            _move_nf4_swap_modules(mods, "cuda")
            module._nf4_swap_active = True
            _ACTIVE_NF4_SWAP_BLOCK = module
            _cleanup_after_nf4_swap("cuda", "block -> CUDA")
        else:
            module._nf4_swap_active = True
            _ACTIVE_NF4_SWAP_BLOCK = module

    module._nf4_swap_grad_enabled = torch.is_grad_enabled()
    return args, kwargs


def _nf4_swap_forward_hook(module, args, output):
    global _ACTIVE_NF4_SWAP_BLOCK
    # In training, checkpoint wrapper is responsible for releasing the block
    # after the original forward. We must NOT release it here, otherwise the
    # same block can be moved twice around checkpoint replay.
    if not getattr(module, "_nf4_swap_grad_enabled", False):
        mods = getattr(module, "_nf4_swap_modules", [])
        if mods and any(m.weight is not None and m.weight.device.type != "cpu" for m in mods):
            _move_nf4_swap_modules(mods, "cpu")
            if _ACTIVE_NF4_SWAP_BLOCK is module:
                _ACTIVE_NF4_SWAP_BLOCK = None
            module._nf4_swap_active = False
            _cleanup_after_nf4_swap("cpu", "block -> CPU (inference)")
    return output


def _nf4_swap_backward_hook(module, grad_input, grad_output):
    global _ACTIVE_NF4_SWAP_BLOCK

    # IMPORTANT: checkpointed blocks are released HERE, after the
    # bitsandbytes Linear4bit backward has completed. The checkpoint wrapper
    # deliberately keeps the block on CUDA during forward/recompute so that
    # B and quant_state remain valid for the bnb autograd function.
    mods = getattr(module, "_nf4_swap_modules", [])
    if mods and any(m.weight is not None and m.weight.device.type != "cpu" for m in mods):
        _move_nf4_swap_modules(mods, "cpu")
        if _ACTIVE_NF4_SWAP_BLOCK is module:
            _ACTIVE_NF4_SWAP_BLOCK = None
        module._nf4_swap_active = False
        _cleanup_after_nf4_swap("cpu", "block -> CPU (backward)")
    return None


# ---------------------------------------------------------------------------
# OVERHEAD DE ENTRENAMIENTO EN FUNCION DE LA SECUENCIA.
#
# El plan restaba un overhead FIJO de 2,5 GB antes de repartir bloques. Ese
# numero no estaba mal: estaba calibrado con imagenes, donde la secuencia son
# 300-700 tokens. Nadie lo reviso al llegar el video, y un clip son 1.100-2.400
# tokens: el plan creia tener 2,5 GB de margen cuando necesitaba 5,5, pedia 30
# bloques residentes y saturaba la tarjeta. Ese fue el primer OOM del proyecto.
#
# Calibrado con seis medidas reales de pico en dos geometrias (256x256/107f y
# 192x192/124f) y de 2 a 25 bloques residentes:
#
#     pico = 4,30 + 0,00216 x tokens + 0,3332 x N        (GiB)
#
# de donde, quitando base y swap que el plan ya cuenta aparte:
#
#     activaciones = 1,672 + 0,001663 x tokens
#
# Error maximo 0,33 GiB en las seis. Y da 2,14 GB para 500 tokens, o sea que
# REPRODUCE el 2,5 antiguo en el caso para el que se calibro: no es un cambio de
# criterio, es la misma constante convertida en la recta que siempre fue.
#
# The plan subtracted a FIXED 2.5 GB overhead before handing out blocks. That
# number was not wrong -- it was calibrated on images, where the sequence is
# 300-700 tokens. Nobody revisited it when video arrived, and a clip is
# 1,100-2,400 tokens: the plan thought it had 2.5 GB of headroom when it needed
# 5.5, asked for 30 resident blocks and saturated the card. Calibrated against
# six real peak measurements across two geometries and 2-25 resident blocks, with
# a 0.33 GiB worst-case error -- and it yields 2.14 GB at 500 tokens, so it
# REPRODUCES the old constant in the case it was calibrated for.
# ---------------------------------------------------------------------------
# Reajustado con ONCE picos reales (2-28 bloques) y, esto es lo importante,
# TRES longitudes de secuencia en vez de dos: 415, 1.480 y 2.240 tokens.
#     pico = 4,667 + 0,001663 x tokens + 0,3479 x N      error maximo 0,52 GiB
#
# POR QUE HACIA FALTA REAJUSTARLA. Las ocho medidas anteriores cubrian de 1.332
# a 2.240 tokens: un rango de 1,7x. Con tan poco brazo de palanca, la constante y
# el termino por token son casi indistinguibles -- cualquier pareja que sume lo
# mismo en mitad del rango reproduce las ocho medidas -- y el ajuste repartio mal
# el reparto entre los dos: demasiado en el token, demasiado poco en la
# constante. Dentro del rango daba igual, porque el error se cancelaba. Fuera no:
# a 415 tokens, que es lo que mide un dataset de SOLO AUDIO, la recta vieja se
# quedaba 0,95 GiB corta y el plan repartia dos o tres bloques de mas.
# Las tres medidas de audio dan el tercer punto de apoyo, y con el la constante
# sube de 3,264 a 4,667 mientras el termino por token baja de 0,002340 a
# 0,001663. Sobre las ONCE medidas el error maximo pasa de 1,51 a 0,52 GiB, y
# dentro del rango de video las predicciones apenas se mueven (< 0,45 GiB).
#
# Refit against ELEVEN real peaks (2-28 blocks) and -- this is what matters --
# THREE sequence lengths instead of two: 415, 1,480 and 2,240 tokens. The
# previous eight spanned 1,332 to 2,240, a 1.7x range, too little leverage to
# separate the constant from the per-token term: any pair summing the same
# mid-range fits them all. Inside that range the misallocation cancelled; outside
# it did not. At 415 tokens -- an AUDIO-ONLY dataset -- the old line fell 0.95
# GiB short and the plan handed out two or three blocks too many. Worst-case
# error over the eleven drops from 1.51 to 0.52 GiB, and predictions inside the
# video range barely move (< 0.45 GiB).
# donde `tokens` son los de la muestra MAS GRANDE, leidos de latent_shape, no
# una media: el pico lo marca la peor muestra. / where `tokens` is the LARGEST
# sample's, read from latent_shape rather than averaged.
# Quitando base (1,904) y swap (1,34) queda la parte de activaciones. El termino
# en N que sobra (0,021/bloque, la diferencia entre los 0,3542 medidos y los
# 0,3332 que pesa un bloque NF4 -- es el workspace de dequantizacion) se dobla en
# la constante usando un N tipico de 17, porque el plan necesita el overhead
# ANTES de decidir cuantos bloques caben y no puede depender de N.
#
# Refit against EIGHT real peaks (2-25 blocks, 1,332 and 2,048 tokens), worst
# error 0.24 GiB. The leftover per-block term (0.021/block: the gap between the
# measured 0.3542 and the 0.3332 a NF4 block weighs, i.e. dequant workspace) is
# folded into the constant at a typical N of 17, because the plan needs the
# overhead BEFORE deciding how many blocks fit and so cannot depend on N.
ACT_BASE_GB = 1.672
ACT_PER_TOKEN_GB = 0.001663


def cache_sequence_tokens(cache_dir):
    """Tokens de video por muestra, leidos de cache_info.json. 0 si no se sabe.

    Un token cubre 32x32 pixeles reales (VAE /16 y patch 2x2), asi que
    tokens_por_fotograma = area / 1024. Los fotogramas LATENTES salen de la
    geometria 17n+5 de H3: 5 latentes por cada 17 de pixel, mas 2.

    Se usa el area OBJETIVO y no los buckets reales de cada clip: los buckets
    redondean a multiplos de 32 y varian un 10% entre si, pero el area objetivo
    es justo la media alrededor de la que se reparten, que es con lo que se
    calibro la recta de arriba.

    Video tokens per sample from cache_info.json, 0 if unknown. One token covers
    32x32 real pixels, so tokens_per_frame = area / 1024; latent frames follow
    H3's 17n+5 geometry. The TARGET area is used rather than each clip's bucket:
    buckets round to multiples of 32 and vary ~10%, but the target area is the
    mean they scatter around, which is what the line above was calibrated on.
    """
    # Se lee el latent_shape de cada muestra y se coge el MAXIMO: el pico de
    # VRAM lo marca la muestra mas grande, no la media. Con el patch (1,2,2) de
    # H3, latent_shape [B, C, T, H, W] son T x (H/2) x (W/2) tokens.
    #
    # No sirve estimarlo desde cache_info: en un dataset MIXTO num_frames vale
    # None (no hay un solo valor, cada muestra lleva el suyo) y la estimacion
    # daria el tamano de una imagen, subestimando el overhead en mas de 3 GB y
    # repartiendo bloques de mas hasta el OOM.
    #
    # Each sample's latent_shape is read and the MAXIMUM taken: the VRAM peak is
    # set by the largest sample. Estimating from cache_info does not work: on a
    # MIXED dataset num_frames is None (there is no single value) and the estimate
    # would return an image-sized sequence, underestimating the overhead by over
    # 3 GB and handing out enough blocks to OOM.
    mayor = 0
    try:
        for nombre in os.listdir(cache_dir):
            if not nombre.endswith("_info.json") or nombre.startswith("_"):
                continue
            if nombre == "cache_info.json":
                continue
            try:
                with open(os.path.join(cache_dir, nombre), "r", encoding="utf-8") as fh:
                    _info = json.load(fh)
                shape = _info.get("latent_shape")
                forma_audio = _info.get("audio_latent_shape")
            except Exception:
                continue
            if shape and len(shape) >= 5:
                tk = int(shape[2]) * (int(shape[3]) // 2) * (int(shape[4]) // 2)
                # Las filas de audio van en la MISMA secuencia empaquetada
                # [texto | video | audio], asi que cuentan igual que las de video
                # para la VRAM. Un clip de 124 fotogramas son 414 filas: un 31%
                # mas de secuencia que ignorarlas costaria tres bloques
                # residentes de mas y un OOM a mitad de corrida.
                # Audio rows live in the SAME packed sequence, so they count like
                # video rows for VRAM. A 124 frame clip is 414 rows: 31% more
                # sequence, which ignoring would cost three resident blocks too
                # many and an OOM mid-run.
                if forma_audio and len(forma_audio) >= 3:
                    tk += int(forma_audio[2])
                mayor = max(mayor, tk)
    except Exception:
        return 0

    if mayor > 0:
        return mayor

    # Cache antigua sin latent_shape: se estima desde la geometria declarada.
    # Older cache with no latent_shape: estimate from the declared geometry.
    try:
        with open(os.path.join(cache_dir, "cache_info.json"), "r", encoding="utf-8") as fh:
            info = json.load(fh)
    except Exception:
        return 0

    area = int(info.get("target_area", 0) or 0)
    frames = int(info.get("num_frames", 1) or 1)
    if area <= 0:
        return 0
    latentes = 1 if frames <= 1 else 5 * ((frames - 5) // 17) + 2
    return int(round(area / 1024.0)) * max(1, latentes)


def training_overhead_gb(cache_dir, fallback):
    """GB que hay que reservar para el paso de entrenamiento, no para los pesos.

    Si no se puede leer la cache se devuelve el valor configurado, que es lo
    unico honesto: inventarse una secuencia seria peor que el numero fijo.

    GB to reserve for the training step rather than for weights. When the cache
    cannot be read the configured value is returned -- inventing a sequence
    length would be worse than the fixed number.
    """
    tokens = cache_sequence_tokens(cache_dir)
    if tokens <= 0:
        return float(fallback), 0

    # Nunca menos conservador que el valor fijo. La recta se calibro con VIDEO
    # (1.300-2.000 tokens) y extrapolada hacia abajo da 1,78 GB para una imagen de
    # 500, por debajo de los 2,5 que llevan funcionando desde siempre en imagen.
    # Bajar ahi seria cambiar, sin una sola medida nueva, el reparto de un caso
    # que ya va bien. El cruce esta en ~790 tokens: por debajo manda el fijo, por
    # encima manda la recta, que es justo donde el fijo se quedaba corto.
    #
    # Never less conservative than the fixed value. The line was calibrated on
    # VIDEO (1,300-2,000 tokens) and extrapolating down gives 1.78 GB for a 500
    # token image, below the 2.5 that has always worked for images -- lowering it
    # would change a working case without a single new measurement. The crossover
    # sits at ~790 tokens: below it the fixed value wins, above it the line does,
    # which is exactly where the fixed value fell short.
    return max(float(fallback), ACT_BASE_GB + ACT_PER_TOKEN_GB * tokens), tokens


def apply_nf4_block_swap_hooks(block_list, start_index):
    hooks = []

    for i in range(start_index, len(block_list)):
        block = block_list[i]

        if not hasattr(block, "_nf4_swap_modules"):
            block._nf4_swap_modules = [m for m in block.modules() if isinstance(m, Linear4bit)]

        block._nf4_swap_grad_enabled = False
        block._nf4_swap_est_bytes = int(max(
            get_block_nf4_bytes(block) * 4,
            get_block_nontrainable_bytes(block),
        ))

        hooks.append(block.register_forward_pre_hook(_nf4_swap_pre_hook, with_kwargs=True))
        hooks.append(block.register_forward_hook(_nf4_swap_forward_hook))

        if hasattr(block, "register_full_backward_hook"):
            hooks.append(block.register_full_backward_hook(_nf4_swap_backward_hook))
        else:
            hooks.append(block.register_backward_hook(_nf4_swap_backward_hook))

    pass  # [SWAP]/[VRAM-LIMIT] log removed / log eliminado
    return hooks


def wrap_nf4_swap_checkpoint_range(block_list, start_index=0, end_index=None, label=""):
    import torch.utils.checkpoint as _cp

    if end_index is None:
        end_index = len(block_list)

    start_index = max(0, int(start_index))
    end_index = min(int(end_index), len(block_list))

    wrapped = 0

    for idx in range(start_index, end_index):
        block = block_list[idx]

        if getattr(block, "_nf4_swap_ckpt_wrapped", False):
            continue

        orig_forward = block.forward

        def make_checkpointed(orig_fwd, blk):
            def checkpointed_forward(*args, **kwargs):
                if getattr(blk, "_nf4_inside_ckpt", False):
                    return orig_fwd(*args, **kwargs)

                mods = getattr(blk, "_nf4_swap_modules", [])

                if not torch.is_grad_enabled():
                    blk._nf4_inside_ckpt = True
                    saved_forward = blk.forward
                    blk.forward = orig_fwd

                    try:
                        out = blk(*args, **kwargs)
                    finally:
                        blk.forward = saved_forward
                        blk._nf4_inside_ckpt = False

                    if mods and any(m.weight is not None and m.weight.device.type != "cpu" for m in mods):
                        _move_nf4_swap_modules(mods, "cpu")
                        _cleanup_after_nf4_swap("cpu", "checkpoint block -> CPU")

                    return out

                kwarg_keys = list(kwargs.keys())
                flat_inputs = list(args) + [kwargs[k] for k in kwarg_keys]
                n_args = len(args)

                def custom_forward(*flat):
                    real_args = flat[:n_args]
                    real_kwargs = dict(zip(kwarg_keys, flat[n_args:]))

                    blk._nf4_inside_ckpt = True
                    saved_forward = blk.forward
                    blk.forward = orig_fwd

                    try:
                        return blk(*real_args, **real_kwargs)
                    finally:
                        # CRITICAL: never move Linear4bit/NF4 weights to CPU
                        # while autograd owns the checkpointed computation.
                        # This function executes during both the original
                        # forward and checkpoint recomputation. Moving the
                        # weight here can invalidate bitsandbytes B/quant_state
                        # and produces CUDA illegal-memory-access errors in
                        # bitsandbytes.autograd._functions.backward().
                        #
                        # The block is released by _nf4_swap_backward_hook only
                        # after the bnb backward for this module has completed.
                        blk.forward = saved_forward
                        blk._nf4_inside_ckpt = False

                out = _cp.checkpoint(
                    custom_forward,
                    *flat_inputs,
                    # Ver checkpoint_use_reentrant en DEFAULTS: con False,
                    # bitsandbytes clava el peso NF4 de CUDA en el grafo hasta
                    # el backward y el block swap no libera nada.
                    # See checkpoint_use_reentrant: with False, bnb pins the
                    # CUDA NF4 weight in the graph and the swap frees nothing.
                    use_reentrant=CHECKPOINT_USE_REENTRANT,
                )

                # The checkpoint wrapper intentionally leaves the NF4 block
                # resident until its backward hook has finished.
                return out

            return checkpointed_forward

        block.forward = make_checkpointed(orig_forward, block)
        block._nf4_swap_ckpt_wrapped = True
        wrapped += 1

    if wrapped:
        log_print(
            "[CKPT] '{}': {} bloques NF4-swapped con checkpoint swap-aware.".format(
                label or "rango",
                wrapped,
            ),
            flush=True,
        )

    return wrapped


def apply_vram_hard_cap():
    """Tope DURO de VRAM para el proceso / HARD per-process VRAM cap.

    Sin esto, `vram_budget_gb` era solo una sugerencia para colocar pesos: nada
    impedia que activaciones, gradientes, estado de AdamW y los picos de
    dequantizacion de bitsandbytes se comieran el resto de la tarjeta. De ahi que
    con budget 5.0 el consumo real fueran 16 GB.

    `set_per_process_memory_fraction` hace que PyTorch lance OOM al llegar al
    limite simulado en vez de seguir pidiendo memoria al driver. Es LO QUE HACE
    FALTA para el objetivo declarado: comprobar de verdad si esto entraria en una
    GPU de 8/10/12 GB sin tener esa GPU.

    Without this the budget was only a hint for weight placement; nothing stopped
    the rest of training from eating the card. set_per_process_memory_fraction makes
    PyTorch raise OOM at the simulated limit, which is what "test a smaller GPU"
    actually requires.
    """
    if not (VRAM_HARD_CAP_ENABLED and torch.cuda.is_available()):
        return None

    try:
        total_bytes = torch.cuda.get_device_properties(0).total_memory
    except Exception:
        return None

    cap_gb = (float(VRAM_BUDGET_GB) + float(VRAM_SWAP_GB)
              + float(VRAM_HEADROOM_GB) + float(VRAM_TRAINING_OVERHEAD_GB))
    cap_bytes = cap_gb * 1e9

    if cap_bytes >= total_bytes:
        log_print("[VRAM-CAP] Tope solicitado {:.2f} GB >= VRAM fisica {:.2f} GB: "
                  "no se aplica (no hay nada que simular). / cap >= physical VRAM, "
                  "not applied.".format(cap_gb, total_bytes / 1e9), flush=True)
        return None

    fraction = max(0.05, min(1.0, cap_bytes / float(total_bytes)))

    try:
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
    except Exception as e:
        log_print("[VRAM-CAP][WARN] No se pudo aplicar el tope: {}".format(e), flush=True)
        return None

    log_print(
        "[VRAM-CAP] Tope DURO en {:.2f} GB de {:.2f} GB fisicos (fraccion {:.3f}). "
        "budget {:.2f} + swap {:.2f} + headroom {:.2f} + overhead {:.2f}. A partir de "
        "aqui, pasarse = OOM, no 'usar toda la tarjeta'. / HARD cap applied.".format(
            cap_gb, total_bytes / 1e9, fraction, VRAM_BUDGET_GB, VRAM_SWAP_GB,
            VRAM_HEADROOM_GB, VRAM_TRAINING_OVERHEAD_GB),
        flush=True,
    )
    return cap_bytes


def setup_block_cpu_offload(transformer, target_vram_gb=None, reserve_gb=None):
    if not torch.cuda.is_available():
        return None

    if target_vram_gb is None:
        target_vram_gb = VRAM_BUDGET_GB

    swap_gb = VRAM_SWAP_GB
    headroom_gb = VRAM_HEADROOM_GB if reserve_gb is None else float(reserve_gb)

    candidates = find_block_lists(transformer)
    if not candidates:
        log_print("[VRAM-PLAN] No se encontró ModuleList de bloques; moviendo todo a GPU.", flush=True)
        transformer.to("cuda")
        free_vram()
        return None

    name, block_list, _ = candidates[0]
    n_blocks = len(block_list)

    for p in transformer.parameters():
        if getattr(p, "requires_grad", False) and p.device.type != "cuda":
            p.data = p.data.to("cuda")

    move_non_block_to_cuda(transformer, block_list)

    free_vram()
    vram_stats("Post base no-bloques")

    base_alloc = torch.cuda.memory_allocated()

    try:
        total_physical_bytes = torch.cuda.get_device_properties(0).total_memory
        total_physical_gb = total_physical_bytes / 1e9
    except Exception:
        total_physical_bytes = None
        total_physical_gb = None

    # El presupuesto sigue siendo MANUAL (permite simular GPUs de 8/10/12/16 GB
    # en la misma máquina), pero ahora se contrasta contra la VRAM libre REAL
    # ahora mismo (torch.cuda.mem_get_info) reservando swap+headroom, en vez de
    # depender solo de que salte una excepción de OOM más adelante para
    # corregirse. Así el "techo efectivo" queda explícito en el log y escala
    # de verdad con la VRAM física disponible en cada GPU.
    # --------------------------------------------------------------------
    # CONTABILIDAD HONESTA.
    #
    # max_base_alloc es lo que pueden ocupar los PESOS RESIDENTES. Pero el paso de
    # entrenamiento anade encima: gradientes de las LoRA, estado de AdamW (2 slots),
    # activaciones + grafo, los pesos bf16 que bitsandbytes materializa en el
    # backward de cada Linear4bit, y los workspaces de cuBLAS/SDPA. Nada de eso
    # estaba restado en ningun sitio: por eso el plan decia "5 GB" y la tarjeta
    # acababa a 16. Se descuenta explicitamente aqui.
    # The training step adds grads, AdamW state, activations, bnb dequant peaks and
    # cuBLAS/SDPA workspaces on top of resident weights. None of it was subtracted.
    # --------------------------------------------------------------------
    overhead_gb, seq_tokens = training_overhead_gb(CACHE_DIR, VRAM_TRAINING_OVERHEAD_GB)
    overhead_bytes = max(0.0, float(overhead_gb)) * 1e9
    max_base_alloc = max(0.0, float(target_vram_gb) * 1e9 - overhead_bytes)

    if overhead_bytes > 0:
        if seq_tokens > 0:
            log_print(
                "[VRAM-PLAN] Secuencia de {} tokens -> overhead de entrenamiento {:.2f} GB "
                "(1,672 + 0,001663 x tokens, calibrado sobre 11 picos reales). El valor fijo "
                "de {:.2f} GB solo valia para imagenes. / {} token sequence -> {:.2f} GB "
                "training overhead; the fixed {:.2f} GB only held for images."
                .format(seq_tokens, overhead_gb, float(VRAM_TRAINING_OVERHEAD_GB),
                        seq_tokens, overhead_gb, float(VRAM_TRAINING_OVERHEAD_GB)),
                flush=True,
            )
        else:
            log_print(
                "[VRAM-PLAN][WARN] No se pudo leer la geometria de la cache; se usa el "
                "overhead fijo de {:.2f} GB, que esta calibrado para IMAGENES y se queda "
                "corto con video. / Could not read the cache geometry; falling back to the "
                "fixed {:.2f} GB overhead, which is calibrated for IMAGES and is too small "
                "for video.".format(overhead_gb, overhead_gb),
                flush=True,
            )
        log_print(
            "[VRAM-PLAN] Presupuesto residente {:.2f} GB - overhead de entrenamiento "
            "{:.2f} GB = {:.2f} GB para pesos. / resident budget minus training "
            "overhead.".format(float(target_vram_gb), overhead_bytes / 1e9,
                               max_base_alloc / 1e9),
            flush=True,
        )

    if base_alloc > max_base_alloc:
        log_print(
            "[VRAM-PLAN][WARN] Los modulos FUERA de bloques ya ocupan {:.2f} GB, mas "
            "que el presupuesto para pesos ({:.2f} GB). Ningun bloque va a quedarse "
            "residente y aun asi te pasas del budget: sube vram_budget_gb o el modelo "
            "no cabe en esa GPU simulada. / non-block modules alone exceed the weight "
            "budget.".format(base_alloc / 1e9, max_base_alloc / 1e9),
            flush=True,
        )

    # ------------------------------------------------------------------
    # EL OVERHEAD NO SE SUMA AQUI: YA ESTA DENTRO DEL BUDGET.
    #
    # Unas lineas mas arriba se hace max_base_alloc = target_vram_gb - overhead,
    # o sea que vram_budget_gb es "pesos + overhead de entrenamiento". Sumarlo
    # otra vez lo contaba DOS veces: con budget 14,25 / swap 1,34 / headroom 0,10
    # daba 18,19 GB en vez de 15,69 y disparaba un aviso falso de "supera la VRAM
    # fisica" en cualquier tarjeta de 16 GB, precisamente en la configuracion que
    # si funciona.
    #
    # The overhead is NOT added here: it is already inside the budget, since
    # max_base_alloc = target_vram_gb - overhead a few lines above. Adding it
    # again counted it twice and raised a false "exceeds physical VRAM" warning
    # on the very configuration that works.
    # ------------------------------------------------------------------
    configured_total = float(target_vram_gb) + float(swap_gb) + float(headroom_gb)
    if total_physical_gb is not None and configured_total > total_physical_gb:
        log_print(
            "[VRAM-PLAN] AVISO / WARNING: presupuesto manual {:.4f} GB supera la VRAM "
            "fisica {:.4f} GB. / manual budget exceeds physical VRAM.".format(
                configured_total, total_physical_gb
            ),
            flush=True,
        )

    try:
        free_now_bytes, _total_now_bytes = torch.cuda.mem_get_info()
    except Exception:
        free_now_bytes = None

    if free_now_bytes is not None:
        safety_bytes = VRAM_SWAP_MAX_BYTES + VRAM_HEADROOM_MAX_BYTES
        # Techo físico real para bloques residentes: lo que hay libre ahora
        # mismo, más lo que ya está alojado como base, menos lo que hace falta
        # reservar para el swap JIT y el headroom de entrenamiento.
        physical_ceiling = max(0.0, (free_now_bytes + base_alloc) - safety_bytes)

        if physical_ceiling < max_base_alloc:
            log_print(
                "[VRAM-PLAN] Budget configurado ({:.2f} GB) no cabe en la VRAM libre real. "
                "Techo efectivo ajustado a {:.2f} GB (libre real {:.2f} GB - swap {:.2f} GB - headroom {:.2f} GB).".format(
                    float(target_vram_gb), physical_ceiling / 1e9,
                    free_now_bytes / 1e9, float(swap_gb), float(headroom_gb),
                ),
                flush=True,
            )
            max_base_alloc = physical_ceiling

    # Cuatro decimales en base_alloc y en el residente maximo: son las dos cifras
    # con las que se calibra vram_budget_gb, y a dos decimales la incertidumbre
    # (+-0,005 GB) basta para mover un escalon entero de bloque, que son 0,3332.
    # Four decimals on base_alloc and the effective resident max: these are the
    # two figures vram_budget_gb is calibrated against, and at two decimals the
    # +-0.005 GB rounding is enough to shift a whole 0.3332 GB block step.
    log_print(
        "[VRAM-PLAN] Base fuera de bloques / non-block base: {:.4f} GB | Residente MAX "
        "(efectivo) / effective resident max: {:.4f} GB | Swap MAX: {:.4f} GB | "
        "Headroom: {:.4f} GB | TOTAL MAX: {:.4f} GB".format(
            base_alloc / 1e9,
            max_base_alloc / 1e9,
            float(swap_gb),
            float(headroom_gb),
            configured_total,
        ),
        flush=True,
    )
    log_print(
        "[VRAM-PLAN] Bloque residente / resident block: {:.6f} GB | para N bloques hace "
        "falta vram_budget_gb >= base + N*bloque + overhead / for N blocks you need "
        "vram_budget_gb >= base + N*block + overhead ({:.4f} GB)".format(
            (get_block_nontrainable_bytes(block_list[0]) / 1e9) if n_blocks else 0.0,
            float(VRAM_TRAINING_OVERHEAD_GB),
        ),
        flush=True,
    )

    block_total_sizes = []
    block_nf4_sizes = []
    block_small_sizes = []

    prefix_total = [0] * (n_blocks + 1)

    for i, block in enumerate(block_list):
        total_nontrainable = get_block_nontrainable_bytes(block)
        nf4_bytes = get_block_nf4_bytes(block)
        small_bytes = max(0, total_nontrainable - nf4_bytes)

        block_total_sizes.append(total_nontrainable)
        block_nf4_sizes.append(nf4_bytes)
        block_small_sizes.append(small_bytes)

        prefix_total[i + 1] = prefix_total[i] + total_nontrainable

    suffix_small = [0] * (n_blocks + 1)
    for i in range(n_blocks - 1, -1, -1):
        suffix_small[i] = suffix_small[i + 1] + block_small_sizes[i]

    start_index = 0

    for s in range(n_blocks, -1, -1):
        estimated = base_alloc + prefix_total[s] + suffix_small[s]
        if estimated <= max_base_alloc:
            start_index = s
            break

    # Un valor manual manda sobre el plan. Se respeta aunque sea MAYOR que el
    # calculado: quien lo escribe esta midiendo, y si se pasa lo dira el OOM.
    # Lo unico que se recorta es el limite fisico de bloques.
    #
    # A manual value overrides the plan, even when it is HIGHER than the computed
    # one: whoever typed it is measuring, and an OOM will say if it was too much.
    # Only the physical block count is clamped.
    if RESIDENT_BLOCKS > 0:
        forzado = min(int(RESIDENT_BLOCKS), n_blocks)
        log_print(
            "[VRAM-PLAN] Manual override: {} resident blocks (the plan said {}). "
            "The plan only counts weights; with long video sequences the "
            "activations do not fit its overhead. / Forzado manual: {} bloques "
            "residentes (el plan decia {}). El plan solo cuenta pesos; con "
            "secuencias largas de video las activaciones no caben en su "
            "overhead.".format(forzado, start_index, forzado, start_index),
            flush=True)
        start_index = forzado

    log_print(
        "[VRAM-PLAN] Plan teórico: {} bloques residentes (0-{}), {} bloques swapeados ({}-{}).".format(
            start_index,
            start_index - 1,
            n_blocks - start_index,
            start_index,
            n_blocks - 1,
        ),
        flush=True,
    )

    # El spill tiene que estar listo ANTES de aparcar el primer bloque.
    # The spill must be ready BEFORE the first block is parked.
    # Dimensionamos el fichero por lo que ocupan de verdad los bloques NF4:
    # reservar 20 GB fijos gastaria disco para nada.
    # Size the file by what the NF4 blocks actually take: a fixed 20 GB
    # reservation would waste disk for nothing.
    try:
        _spill_gb = (get_block_nf4_bytes(block_list[0])
                     * max(n_blocks - start_index, 1) / (1024.0 ** 3) * 1.05)
    except Exception:
        _spill_gb = 20.0
    spill_init(RAM_LIMIT_GB, PARK_DISK_DIR or OUTPUT_DIR, file_gb=_spill_gb)
    atexit.register(spill_cleanup)

    moved = 0

    for i in range(start_index):
        block = block_list[i]

        try:
            block.to("cuda")
            free_vram()

            if torch.cuda.memory_allocated() > max_base_alloc + 0.5e9:
                log_print(
                    "[VRAM-PLAN] Bloque {} excede presupuesto real; devolviendo a CPU.".format(i),
                    flush=True,
                )
                park_nf4_block(block)
                free_vram()
                break

            moved += 1

        except Exception as e:
            if _is_cuda_oom_error(e):
                # Límite físico real de VRAM: es el único caso en el que
                # cortar silenciosamente la colocación de más bloques es
                # correcto.
                log_print(
                    "[VRAM-PLAN] OOM real de CUDA moviendo bloque {} a GPU (límite físico alcanzado): {}".format(i, e),
                    flush=True,
                )
                park_nf4_block(block)
                free_vram()
                break

            # Cualquier otro error NO es un problema de VRAM: si se traga aquí
            # en silencio, el plan corta la colocación de bloques en el mismo
            # punto siempre, dando la falsa impresión de que vram_budget_gb no
            # tiene efecto. Se registra completo y se relanza.
            log_print(
                "[VRAM-PLAN] ERROR NO relacionado con VRAM moviendo bloque {} a GPU:".format(i),
                flush=True,
            )
            traceback.print_exc()
            raise

    start_index = moved

    ram_stats("Antes de aparcar bloques NF4")
    for i in range(start_index, n_blocks):
        park_nf4_block(block_list[i])
    ram_stats("Después de aparcar bloques NF4")

    free_vram()
    actual_base = torch.cuda.memory_allocated()

    while actual_base > max_base_alloc + 0.75e9 and start_index > 0:
        start_index -= 1

        block = block_list[start_index]

        park_nf4_block(block)
        free_vram()

        actual_base = torch.cuda.memory_allocated()

        log_print(
            "[VRAM-PLAN] Ajuste: bloque {} parqueado en CPU. Base actual: {:.2f} GB.".format(
                start_index,
                actual_base / 1e9,
            ),
            flush=True,
        )

    if start_index < n_blocks:
        max_nf4_block_bytes = max(block_nf4_sizes[start_index:])
        max_dequant_est_bytes = max_nf4_block_bytes * 4

        log_print(
            "[VRAM-PLAN] Bloque swapeado más grande: NF4 {:.2f} GB | estimado dequant bf16 {:.2f} GB.".format(
                max_nf4_block_bytes / 1e9,
                max_dequant_est_bytes / 1e9,
            ),
            flush=True,
        )

        if max_dequant_est_bytes > float(swap_gb) * 1e9:
            log_print(
                "[VRAM-PLAN] AVISO: el bloque swapeado más grande puede necesitar más que "
                "vram_swap_gb={:.2f} GB. Si OOM, sube vram_swap_gb o baja vram_budget_gb.".format(
                    float(swap_gb)
                ),
                flush=True,
            )
        else:
            log_print(
                "[VRAM-PLAN] Swap máximo estimado cabe en vram_swap_gb={:.2f} GB.".format(
                    float(swap_gb)
                ),
                flush=True,
            )

    log_print(
        "[VRAM-PLAN] Post-plan real: {:.2f} GB | Objetivo base: {:.2f} GB | Swap reservado: {:.2f} GB".format(
            actual_base / 1e9,
            max_base_alloc / 1e9,
            float(swap_gb),
        ),
        flush=True,
    )

    if actual_base > max_base_alloc + 0.75e9:
        log_print(
            "[VRAM-PLAN] AVISO: el consumo base real sigue por encima del objetivo. "
            "Baja vram_budget_gb o aumenta vram_headroom_gb.",
            flush=True,
        )

    excluded_audio_ids = set()
    for blk in block_list[start_index:]:
        for nm, m in blk.named_modules():
            if "audio" in nm.lower():
                excluded_audio_ids.add(id(m))

    swap_hooks = apply_nf4_block_swap_hooks(block_list, start_index)

    wrap_block_range_with_explicit_checkpoint(
        block_list,
        0,
        start_index,
        label="bloques GPU residentes",
    )

    wrap_nf4_swap_checkpoint_range(
        block_list,
        start_index,
        n_blocks,
        label="bloques NF4 swap",
    )

    free_vram()
    spill_report()
    vram_stats("Post VRAM plan")

    return {
        "block_list_name": name,
        "block_list": block_list,
        "n_blocks": n_blocks,
        "start_index": start_index,
        "n_offloaded": n_blocks - start_index,
        "excluded_audio_ids": excluded_audio_ids,
        "swap_hooks": swap_hooks,
        "checkpoint_configured": True,
    }


# =============================================================================
# OPTIMIZADOR MIXTO
# =============================================================================

class DualOptimizer:
    def __init__(self, gpu_optimizer=None, cpu_optimizer=None):
        self.gpu_optimizer = gpu_optimizer
        self.cpu_optimizer = cpu_optimizer

    @property
    def param_groups(self):
        groups = []

        if self.gpu_optimizer is not None:
            groups.extend(self.gpu_optimizer.param_groups)

        if self.cpu_optimizer is not None:
            groups.extend(self.cpu_optimizer.param_groups)

        return groups

    def zero_grad(self, set_to_none=True):
        if self.gpu_optimizer is not None:
            self.gpu_optimizer.zero_grad(set_to_none=set_to_none)

        if self.cpu_optimizer is not None:
            self.cpu_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        if self.gpu_optimizer is not None:
            self.gpu_optimizer.step()

        if self.cpu_optimizer is not None:
            self.cpu_optimizer.step()

    def state_dict(self):
        return {
            # El tipo se guarda para poder RECHAZAR un optimizer.pt escrito por el otro
            # optimizador. El estado de AdamW8bit y el de AdamW no son intercambiables:
            # cargarlo a ciegas mete momentos corruptos y arruina la reanudacion en
            # silencio. / The type is stored so a state written by the other optimizer
            # can be REJECTED: AdamW8bit and AdamW states are not interchangeable and
            # loading one into the other silently corrupts the resumed run.
            "opt_type": OPTIMIZER_TYPE,
            "gpu": self.gpu_optimizer.state_dict() if self.gpu_optimizer is not None else None,
            "cpu": self.cpu_optimizer.state_dict() if self.cpu_optimizer is not None else None,
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict) or ("gpu" not in state and "cpu" not in state):
            if self.gpu_optimizer is not None:
                try:
                    self.gpu_optimizer.load_state_dict(state)
                except Exception as e:
                    log_print("[!] No se pudo restaurar optimizer GPU (formato antiguo): {}".format(e), flush=True)
            return

        saved_type = state.get("opt_type", None)

        if saved_type is not None and saved_type != OPTIMIZER_TYPE:
            log_print(
                "[!] WARNING: el checkpoint del optimizador es de tipo '{}' y ahora usas "
                "'{}'. NO se restaura el estado del optimizador (los momentos no son "
                "compatibles). Los pesos del LoRA si se han restaurado; Adam volvera a "
                "calentar sus momentos en unos pocos pasos. / Optimizer checkpoint is "
                "'{}' but the current optimizer is '{}': the optimizer state is NOT "
                "restored. LoRA weights were restored; Adam will re-warm its moments in "
                "a few steps.".format(saved_type, OPTIMIZER_TYPE, saved_type, OPTIMIZER_TYPE),
                flush=True,
            )
            return

        if self.gpu_optimizer is not None and state.get("gpu") is not None:
            self.gpu_optimizer.load_state_dict(state["gpu"])

        if self.cpu_optimizer is not None and state.get("cpu") is not None:
            self.cpu_optimizer.load_state_dict(state["cpu"])


def clip_grad_norm_mixed_device(parameters, max_norm):
    try:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm).item()
    except Exception:
        grads = [p.grad for p in parameters if p.grad is not None]

        if not grads:
            return 0.0

        total_sq = 0.0
        for g in grads:
            total_sq += float(g.detach().float().norm(2).item()) ** 2

        total_norm = total_sq ** 0.5
        clip_coef = max_norm / (total_norm + 1e-6)

        if clip_coef < 1:
            for g in grads:
                g.detach().mul_(clip_coef)

        return total_norm


def cast_frozen_to_bf16(root):
    """Castea a bf16 los pesos congelados, RESPETANDO _keep_in_fp32_modules.

    Antes casteaba absolutamente todo, y eso deshacia `fp32_repair_enabled` tres
    funciones despues de aplicarlo. Peor: dejaba en bf16 `rope.inv_freq` (el modelo
    hace `position_ids.to(float32)` a proposito antes de multiplicar por el) y
    `time_embedder`, cuyo redondeo diffusers advierte por escrito que "se acumula
    coherentemente a lo largo de la trayectoria" porque los 50 bloques leen el mismo
    `temb`. Los modulos protegidos suman <100 MB en fp32.

    Casts frozen weights to bf16 while HONOURING _keep_in_fp32_modules. It used to
    cast everything, undoing `fp32_repair_enabled` right after it ran and leaving
    `rope.inv_freq` and `time_embedder` in bf16 -- the exact rounding diffusers warns
    about in the model source. The protected modules add up to <100 MB in fp32.
    """
    keep_fp32 = tuple(getattr(type(root), "_keep_in_fp32_modules", None)
                      or ["proj_in", "audio_proj_in", "time_embedder",
                          "proj_out", "audio_proj_out", "rope"])

    def _protected(n):
        if not CAST_FROZEN_RESPECT_FP32:
            return False
        return any(n == p or n.startswith(p + ".") or ".{}.".format(p) in n
                   for p in keep_fp32)

    kept_params, kept_buffers = [], []

    for name, param in root.named_parameters():
        if isinstance(param, Params4bit):
            continue

        if param.requires_grad:
            continue

        if _protected(name):
            if param.is_floating_point() and param.dtype != torch.bfloat16:
                kept_params.append(name)
            continue

        if param.is_floating_point() and param.dtype != torch.bfloat16:
            param.data = param.data.to(torch.bfloat16)

    for name, buf in root.named_buffers():
        if buf.is_floating_point() and buf.dtype != torch.bfloat16:
            lower = name.lower()

            # inv_freq es la tabla de frecuencias del RoPE, y el forward hace
            # `position_ids.to(float32)` a proposito antes de multiplicar por ella.
            # Error de fase medido si va en bf16 (d=16, theta=10000):
            #   t=37  -> 0.38 grados | t=191 -> 1.95 grados | t=1024 -> 10.5 grados
            # Es pequeno con prompts cortos y crece linealmente con la posicion.
            # Mantenerla en fp32 no cuesta nada (16 floats), asi que no hay motivo
            # para pagarlo.
            # inv_freq is the RoPE frequency table, and the forward deliberately does
            # `position_ids.to(float32)` before multiplying by it. Measured phase error
            # in bf16: 0.38 deg at t=37, 1.95 deg at t=191, 10.5 deg at t=1024 -- small
            # for short prompts, linear in position. Keeping it fp32 costs 16 floats.
            if CAST_FROZEN_RESPECT_FP32 and ("inv_freq" in lower or _protected(name)):
                kept_buffers.append(name)
                continue

            if any(k in lower for k in ("norm", "ln", "layernorm")):
                continue

            buf.data = buf.data.to(torch.bfloat16)

    if CAST_FROZEN_RESPECT_FP32:
        log_print("[CAST] Kept out of bf16 (_keep_in_fp32_modules + RoPE): {} params, "
                  "{} buffers / Mantenidos fuera de bf16: {} params, {} buffers".format(
                      len(kept_params), len(kept_buffers),
                      len(kept_params), len(kept_buffers)), flush=True)
        for n in (kept_params[:8] + kept_buffers[:8]):
            log_print("  [CAST-KEEP] {}".format(n), flush=True)
    else:
        log_print("[CAST][WARN] cast_frozen_respect_fp32_modules=false: casting EVERYTHING "
                  "to bf16, including _keep_in_fp32_modules and rope.inv_freq. This undoes "
                  "fp32_repair. / casteando TODO a bf16, incluidos _keep_in_fp32_modules y "
                  "rope.inv_freq. Esto deshace fp32_repair.", flush=True)


# =============================================================================
# EXPORT LoRA
# =============================================================================

def _fuse_qkv_lora(q_A, q_B, k_A, k_B, v_A, v_B, target_rank):
    """Fusiona 3 LoRA independientes (to_q/to_k/to_v) en una única LoRA
    equivalente para una capa qkv_proj fusionada, recomprimiendo a target_rank
    vía QR + SVD truncada."""
    orig_dtype = q_A.dtype

    r_q, r_k, r_v = q_A.shape[0], k_A.shape[0], v_A.shape[0]
    r_total = r_q + r_k + r_v

    out_q, in_q = q_B.shape[0], q_A.shape[1]
    out_k, out_v = k_B.shape[0], v_B.shape[0]
    out_total = out_q + out_k + out_v

    B_concat = torch.zeros((out_total, r_total), dtype=torch.float32)
    B_concat[0:out_q, 0:r_q] = q_B.float()
    B_concat[out_q:out_q + out_k, r_q:r_q + r_k] = k_B.float()
    B_concat[out_q + out_k:out_total, r_q + r_k:r_total] = v_B.float()

    A_concat = torch.cat([q_A.float(), k_A.float(), v_A.float()], dim=0)

    Q_B, R_B = torch.linalg.qr(B_concat)
    Q_A, R_A = torch.linalg.qr(A_concat.T)

    K_mat = R_B @ R_A.T
    U_k, S_k, Vh_k = torch.linalg.svd(K_mat)

    r = min(target_rank, len(S_k))
    U_r = U_k[:, :r]
    S_r = S_k[:r]
    Vh_r = Vh_k[:r, :]

    sqrt_S = torch.diag(torch.sqrt(S_r))

    qkv_lora_B = (Q_B @ (U_r @ sqrt_S)).to(orig_dtype)
    qkv_lora_A = (sqrt_S @ (Vh_r @ Q_A.T)).to(orig_dtype)

    return qkv_lora_A, qkv_lora_B


def build_lora_metadata(prefix, scaling, extra=None):
    """Cabecera de metadatos del .safetensors.

    Va en el header del fichero, no en los tensores: no pesa nada, no cambia un
    solo bit de los pesos y cualquier lector puede ignorarla sin romperse.

    Se escriben DOS familias de claves:

      - Legibles (`trigger_word`, `trained_with`, ...) para quien abra el fichero
        con safetensors y quiera enterarse de algo.
      - `ss_*`, la convencion de kohya-ss. No es un capricho: es lo que leen
        CivitAI y los gestores de LoRAs para rellenar solos las palabras clave y
        los parametros. Sin ellas el LoRA llega sin ficha.

    CivitAI saca las palabras de activacion de `ss_tag_frequency`, que espera un
    JSON {carpeta: {tag: veces}}. Por eso el trigger va ahi ademas de en su
    propia clave.

    Metadata header for the .safetensors. It lives in the file header, not in the
    tensors: it costs nothing, changes no weight, and any reader can ignore it.
    Two families are written: readable keys, and the kohya-ss `ss_*` convention,
    which is what CivitAI and LoRA managers actually parse to fill in trigger
    words and training parameters. CivitAI reads the activation words from
    `ss_tag_frequency`, a JSON of {folder: {tag: count}}, so the trigger goes
    there as well as in its own key.
    """
    meta = {
        "format": "minimaxh3_lora",
        "lora_key_prefix": prefix,
        "baked_scaling": "{:.6f}".format(scaling),
        "qkv_fused": "true",
        "swiglu_fc1_halves_swapped": "true",

        # --- procedencia / provenance ---
        "trained_with": "AcademiaSD LoRAlab MiniMax-H3",
        "ss_sd_model_name": "MiniMax-H3",
        "ss_base_model_version": "MiniMax-H3",

        # --- receta / recipe ---
        "ss_network_module": "peft.LoraModel",
        "ss_network_dim": str(LORA_RANK),
        "ss_network_alpha": str(LORA_ALPHA),
        "ss_learning_rate": str(LR),
        "ss_lr_scheduler": str(LR_SCHEDULE),
        "ss_max_train_steps": str(TOTAL_STEPS),
        "ss_gradient_accumulation_steps": str(GRAD_ACCUM_STEPS),
        "ss_seed": str(SEED),
        "ss_mixed_precision": str(LORA_DTYPE_STR),
    }

    trigger = (TRIGGER_WORD or "").strip()
    if trigger:
        meta["trigger_word"] = trigger
        meta["ss_output_name"] = PROJECT_NAME or trigger
        # CivitAI lee de aqui las palabras de activacion.
        # CivitAI reads the activation words from here.
        meta["ss_tag_frequency"] = json.dumps({"dataset": {trigger: 1}})

    if PROJECT_NAME:
        meta["project_name"] = PROJECT_NAME
        meta.setdefault("ss_output_name", PROJECT_NAME)

    # Resolucion y frames: los fijo el pre-cache, no el entrenador.
    # Resolution and frames come from the pre-cache, not from the trainer.
    area = _CACHE_INFO.get("target_area")
    if area:
        side = int(round(float(area) ** 0.5))
        meta["ss_resolution"] = "({},{})".format(side, side)
    frames = _CACHE_INFO.get("num_frames")
    if frames:
        meta["ss_num_frames"] = str(frames)

    if extra:
        meta.update(extra)

    # safetensors exige que TODO sea str: un int cuelga el guardado con un error
    # que no dice cual de las claves lo provoco.
    # safetensors requires every value to be a str: an int fails the save with an
    # error that does not say which key caused it.
    return {str(k): str(v) for k, v in meta.items() if v is not None}


def save_lora(model, path, prefix=None):
    if prefix is None:
        prefix = LORA_KEY_PREFIX

    if prefix is None:
        prefix = ""

    scaling = float(LORA_ALPHA) / float(max(1, LORA_RANK))

    # 1) Recolectar todas las keys LoRA en crudo (nombres "diffusers"), con el
    #    scaling ya horneado en lora_B, en fp32 para no perder precisión en la
    #    fusión QKV.
    raw = {}
    for name, tensor in model.state_dict().items():
        if "lora_" not in name:
            continue

        clean = name.replace("base_model.model.", "")
        clean = clean.replace(".default.", ".")

        t = tensor.detach().to(torch.float32).cpu()

        if ".lora_B." in clean:
            t = t * scaling

        raw[prefix + clean] = t

    # 2) Renombrar al esquema correcto (qkv_proj / out_proj / mlp.fc1 / mlp.fc2)
    #    y fusionar Q,K,V en una única LoRA qkv_proj por bloque.
    block_prefixes = set()
    for k in raw.keys():
        if k.endswith("attn.to_q.lora_A.weight"):
            block_prefixes.add(k[: -len("attn.to_q.lora_A.weight")])

    # ------------------------------------------------------------------
    # SWIGLU: LAS DOS MITADES DE fc1 VAN AL REVES.
    #
    # El checkpoint original de MiniMax-H3 guarda `mlp.fc1` fusionado como
    # [gate; value] y calcula fc2(silu(gate) * value). El `SwiGLU` de diffusers
    # calcula value * silu(gate) leyendo un fusionado [value; gate], asi que el
    # conversor oficial (scripts/convert_minimax_h3_to_diffusers.py) INTERCAMBIA
    # las dos mitades al pasar mlp.fc1 -> ff.net.0.proj:
    #
    #     gate, value = tensor.chunk(2, dim=0)
    #     return [(target_key, torch.cat([value, gate], dim=0))]
    #
    # Nosotros entrenamos ff.net.0.proj (layout diffusers) y exportamos con el
    # nombre nativo mlp.fc1, o sea que hay que DESHACER ese swap. Sin esto, el
    # delta aprendido para el gate se suma al value y viceversa en los 50
    # bloques: el LoRA modifica la generacion pero lo que hace no tiene nada que
    # ver con lo que aprendio. Es exactamente el sintoma de "cambia el video
    # pero no da parecido".
    #
    # El swap es sobre la dimension de SALIDA, asi que solo afecta a lora_B
    # (filas de dW = B @ A). lora_A no se toca.
    #
    # SWIGLU: fc1's two halves are stored in the opposite order. The original
    # checkpoint holds [gate; value] and diffusers' SwiGLU holds [value; gate],
    # so the official converter swaps them. We train in diffusers layout and
    # export under the native name, so the swap has to be undone. It applies to
    # the OUTPUT dimension, hence lora_B only.
    # ------------------------------------------------------------------
    def _swiglu_swap_halves(t):
        """[value; gate] (diffusers) -> [gate; value] (checkpoint original)."""
        if t.ndim != 2 or t.shape[0] % 2 != 0:
            raise RuntimeError(
                "mlp.fc1 lora_B con shape inesperada {}: no se puede intercambiar "
                "las mitades SwiGLU / unexpected shape, cannot swap SwiGLU halves"
                .format(tuple(t.shape)))
        value, gate = t.chunk(2, dim=0)
        return torch.cat([gate, value], dim=0).contiguous()

    # (origen diffusers, destino nativo, transformacion del tensor)
    simple_map = [
        ("ff.net.0.proj.lora_A.weight", "mlp.fc1.lora_A.weight", None),
        ("ff.net.0.proj.lora_B.weight", "mlp.fc1.lora_B.weight", _swiglu_swap_halves),
        ("ff.net.2.lora_A.weight", "mlp.fc2.lora_A.weight", None),
        ("ff.net.2.lora_B.weight", "mlp.fc2.lora_B.weight", None),
        ("attn.to_out.0.lora_A.weight", "attn.out_proj.lora_A.weight", None),
        ("attn.to_out.0.lora_B.weight", "attn.out_proj.lora_B.weight", None),
    ]

    state = {}
    processed = set()
    _swiglu_swapped = 0

    for old_prefix in sorted(block_prefixes):
        new_prefix = old_prefix.replace("transformer_blocks", "blocks").replace("refiner_blocks", "blocks")

        for old_sub, new_sub, transform in simple_map:
            old_k = old_prefix + old_sub
            if old_k in raw:
                t = raw[old_k]
                if transform is not None:
                    t = transform(t)
                    _swiglu_swapped += 1
                state[new_prefix + new_sub] = t.to(torch.bfloat16).contiguous()
                processed.add(old_k)

        q_A = raw.get(old_prefix + "attn.to_q.lora_A.weight")
        q_B = raw.get(old_prefix + "attn.to_q.lora_B.weight")
        k_A = raw.get(old_prefix + "attn.to_k.lora_A.weight")
        k_B = raw.get(old_prefix + "attn.to_k.lora_B.weight")
        v_A = raw.get(old_prefix + "attn.to_v.lora_A.weight")
        v_B = raw.get(old_prefix + "attn.to_v.lora_B.weight")

        qkv_keys = [
            old_prefix + "attn.to_q.lora_A.weight", old_prefix + "attn.to_q.lora_B.weight",
            old_prefix + "attn.to_k.lora_A.weight", old_prefix + "attn.to_k.lora_B.weight",
            old_prefix + "attn.to_v.lora_A.weight", old_prefix + "attn.to_v.lora_B.weight",
        ]

        if all(x is not None for x in (q_A, q_B, k_A, k_B, v_A, v_B)):
            processed.update(qkv_keys)
            # target_rank = suma de los 3 rangos: fusión SIN pérdida. Truncar a
            # LORA_RANK (como antes) tira ~2/3 de lo aprendido en Q/K/V por SVD.
            _full_rank = q_A.shape[0] + k_A.shape[0] + v_A.shape[0]
            qkv_A, qkv_B = _fuse_qkv_lora(q_A, q_B, k_A, k_B, v_A, v_B, _full_rank)
            state[new_prefix + "attn.qkv_proj.lora_A.weight"] = qkv_A.to(torch.bfloat16).contiguous()
            state[new_prefix + "attn.qkv_proj.lora_B.weight"] = qkv_B.to(torch.bfloat16).contiguous()

    # 3) Cualquier otra key LoRA que no pertenezca a un bloque attn/mlp detectado
    #    se copia tal cual (solo renombrando transformer_blocks/refiner_blocks).
    for k, v in raw.items():
        if k in processed:
            continue
        new_k = k.replace("transformer_blocks", "blocks").replace("refiner_blocks", "blocks")
        state[new_k] = v.to(torch.bfloat16).contiguous()

    # ------------------------------------------------------------------
    # INFORME DE KEYS.
    #
    # Este bloque de exportacion traduce del layout diffusers (el que entrena
    # PEFT) al del checkpoint original, que es el que cargan ComfyUI y
    # ai-toolkit. Las cuatro transformaciones, todas verificadas contra
    # scripts/convert_minimax_h3_to_diffusers.py de diffusers:
    #   1. "transformer_blocks" -> "blocks", "refiner_blocks" -> "blocks",
    #   2. ff.net.0.proj -> mlp.fc1 CON INTERCAMBIO DE LAS DOS MITADES SwiGLU
    #      (el original es [gate; value], diffusers es [value; gate]),
    #      ff.net.2 -> mlp.fc2, attn.to_out.0 -> attn.out_proj,
    #   3. FUSIONA to_q/to_k/to_v en un unico attn.qkv_proj, orden [q; k; v],
    #      que es el layout en memoria del modelo de referencia.
    # Comparado contra Lain_MiniMax.safetensors (ai-toolkit, funciona en
    # ComfyUI): mismas 416 keys, mismos nombres, mismos shapes salvo el rango
    # de qkv_proj (48 = 3x16 por la fusion exacta, contra 16 del suyo).
    # This export translates the PEFT/diffusers layout to the original
    # checkpoint layout that ComfyUI and ai-toolkit load. All four transforms
    # verified against diffusers' own conversion script.
    # ------------------------------------------------------------------
    _sample = sorted(state.keys())[:4]
    log_print("[SAVE] {} keys exportadas. Ejemplos:".format(len(state)), flush=True)
    for _k in _sample:
        log_print("[SAVE]   {}".format(_k), flush=True)
    # `print` directo y no `log_print`: esta linea es la CONFIRMACION de que la
    # conversion al layout del checkpoint original se ha aplicado, asi que tiene
    # que verse tambien con debug_training=False.
    # Direct `print`, not `log_print`: this line confirms the conversion ran, so
    # it must stay visible with debug_training=False too.
    print("[SAVE] SwiGLU: {} tensores mlp.fc1.lora_B con las mitades "
          "intercambiadas [value;gate] -> [gate;value] / {} mlp.fc1.lora_B "
          "tensors had their SwiGLU halves swapped".format(
              _swiglu_swapped, _swiglu_swapped), flush=True)
    if _swiglu_swapped == 0:
        log_print("[SAVE][WARN] NINGUN mlp.fc1 exportado. Si el LoRA deberia "
                  "incluir el MLP, algo va mal en el descubrimiento de targets. "
                  "/ NO mlp.fc1 exported; check LoRA target discovery.",
                  flush=True)

    save_file(state, path, metadata=build_lora_metadata(prefix, scaling))

    # ------------------------------------------------------------------
    # SEGUNDO ARCHIVO CON LAS KEYS EN CRUDO (nombres diffusers, sin fusion QKV).
    #
    # Es la prueba directa: si este archivo SI hace efecto en el inferenciador y el
    # convertido no, el problema es la conversion de arriba y no el entrenamiento.
    # Si NINGUNO de los dos hace nada, el problema esta en otro sitio.
    # Direct A/B test: if the raw-key file works and the converted one doesn't, the
    # conversion above is the bug.
    # ------------------------------------------------------------------
    if LORA_SAVE_RAW_COPY:
        raw_state = {}
        for k, v in raw.items():
            raw_state[k] = v.to(torch.bfloat16).contiguous()

        raw_path = path.replace(".safetensors", ".rawkeys.safetensors")
        if raw_path == path:
            raw_path = path + ".rawkeys.safetensors"

        try:
            save_file(
                raw_state,
                raw_path,
                metadata=build_lora_metadata(prefix, scaling, extra={
                    "format": "minimaxh3_lora_rawkeys",
                    "qkv_fused": "false",
                    "swiglu_fc1_halves_swapped": "false",
                }),
            )
            _rs = sorted(raw_state.keys())[:2]
            log_print("[SAVE] Copia con keys en CRUDO -> {} ({} keys). Ejemplos: {}"
                      .format(os.path.basename(raw_path), len(raw_state), _rs), flush=True)
        except Exception as e:
            log_print("[SAVE][WARN] No se pudo guardar la copia en crudo: {}".format(e),
                      flush=True)


# =============================================================================
# PREVISUALIZACION DE PROGRESO / TRAINING PROGRESS PREVIEW
# =============================================================================
#
# COMO FUNCIONA / HOW IT WORKS
# ----------------------------
# Cada PREVIEW_EVERY pasos se genera una imagen con el LoRA en su estado actual,
# usando el prompt del PRIMER elemento del dataset. La imagen se guarda como
# <OUTPUT_DIR>/preview_step_<N>.png, que es exactamente lo que ya sirven
# server.py (/api/previews, /api/preview/<fichero>) y la galeria del HTML, que
# refresca cada 8 s. O sea: no hay que tocar nada del transporte, solo generar
# el PNG. Con PREVIEW_EVERY = 0 no se ejecuta NADA de este bloque.
#
# POR QUE NO SE PARECE AL SISTEMA DE KREA2
# ----------------------------------------
# Krea2 es un DiT de imagen: latente 2D, un solo stream, y el VAE es un
# AutoencoderKLQwenImage estandar. MiniMax-H3 corre una UNICA secuencia
# empaquetada [texto | audio | video] con atencion completa, position_ids
# continuos normalizados por area, tags de modalidad por fila y DOS cabezas de
# salida. Nada de la funcion run_preview de Krea2 es reutilizable; lo unico que
# se copia es la convencion de nombres de fichero y el flujo de la galeria.
#
# LAS TRES PIEZAS QUE HUBO QUE RESOLVER
# -------------------------------------
# 1. TEXTO. No hace falta el text encoder: el pre-cache ya guardo el
#    hidden_state de la capa 50 de cada caption. La preview reutiliza el del
#    primer elemento, recortando el padding igual que el bucle de entrenamiento.
#    Esto es lo que hace que la preview no cueste los 20+ GB del Qwen3-VL-32B.
#
# 2. MUESTREO. MiniMaxH3Scheduler es flow rectificado Euler con eta=0 y una
#    particularidad que hay que respetar: t = 1 - sigma con t=1 = LIMPIO, y la
#    velocidad es "hacia el dato" (x0 = x_t + sigma*v, con SUMA). Es la misma
#    convencion que ya usa el entrenamiento, asi que el bucle de sampling y el
#    de train hablan el mismo idioma. La secuencia empaquetada se construye con
#    build_minimax_packed_indices, LA MISMA funcion que usa el entrenamiento:
#    si algun dia cambia la geometria, cambia para los dos a la vez.
#
# 3. VAE DECODER. Aqui estaba el problema real. El script 1 solo implementa el
#    ENCODER (MiniMaxH3VideoVAEEncoder), asi que no habia forma de volver de
#    latente a pixeles. Resulta que el fichero del repo NF4
#    (vae/minimax_h3_video_vae_fp16.safetensors, 5,2 GB) trae 562 tensores y el
#    pre-cache solo lee 120: los otros 442 SON el decoder. No hace falta el repo
#    original de 10 GB.
#
#    El decoder es un ViT de 36 bloques (dim 2048, 32 cabezas x 64) que expande
#    cada voxel latente a un bloque de 4 x 16 x 16 pixeles. Sus keys estan en la
#    nomenclatura del checkpoint ORIGINAL, no en la de diffusers, asi que hay que
#    traducirlas. Las reglas estan verificadas contra
#    scripts/convert_minimax_h3_to_diffusers.py de diffusers:
#      - decoder.x_embedder.  -> decoder.proj_in.
#      - .attn.to_out.        -> .attn.to_out.0.   (diffusers lo envuelve en ModuleList)
#      - .ff.w2.              -> .ff.net.2.
#      - .ff.w1.              -> .ff.net.0.proj.  CON INTERCAMBIO DE MITADES
#        (el original guarda [gate; up], el SwiGLU de diffusers lee [up; gate]:
#         es EXACTAMENTE el mismo swap que mlp.fc1 en el DiT)
#      - .attn.to_qkv.        -> to_q / to_k / to_v, previo des-entrelazado por
#        cabeza (el checkpoint guarda [head0: q k v, head1: q k v, ...])
#      - decoder.mask_token   -> se descarta (diffusers no lo usa)
#      - encoder.down.{i}.block.{j} -> encoder.down_blocks.{i}.resnets.{j}, etc.
#
#    OJO CON vae.decode(): asume latentes de video en la rejilla 17n+5. Con UN
#    solo frame latente calcula num_chunks = 0 y devuelve un tensor vacio. Para
#    una imagen suelta hay que llamar a _decode_clip(), que es post_quant_conv +
#    decoder (con tiling), y quedarse con un frame de los 4 que devuelve.
#
# MEMORIA
# -------
# El decoder son ~2,4 G parametros: 4,8 GB en bf16. En una tarjeta de 16 GB con
# el DiT residente NO cabe, por eso preview_vae_device viene en "cpu" por
# defecto: se decodifica en RAM (tienen que caber ~9,7 GB en fp32) sin tocar ni
# un byte de VRAM ni interferir con el block swap. Tarda ~1-2 min. Quien tenga
# 24 GB o mas puede poner "cuda" y baja a segundos.
#
# SEGURIDAD
# ---------
# Todo el bloque va envuelto en try/except en el punto de llamada: una preview
# que falle NUNCA puede tumbar un entrenamiento de horas. Y con preview_every=0
# (el valor por defecto) no se importa ni se carga nada.
# =============================================================================

# VAE de preview cacheado entre pasos: cargarlo cuesta segundos y ~10 GB de RAM,
# no tiene sentido repetirlo en cada preview.
# Preview VAE cached across steps: loading costs seconds and ~10 GB of RAM.
_PREVIEW_VAE = {"model": None, "device": None}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _vae_reorder_interleaved_qkv(weight, num_heads, head_dim):
    """[head0: q k v, head1: q k v, ...] -> [q_all; k_all; v_all].

    Vale igual para weight (2-D) que para bias (1-D): shape[1:] queda vacio.
    Works for both the 2-D weight and the 1-D bias."""
    expected = num_heads * 3 * head_dim
    if weight.shape[0] != expected:
        raise ValueError(
            "qkv fusionado con {} filas, se esperaban {} = {} cabezas * 3 * {} / "
            "fused qkv has {} rows, expected {}".format(
                weight.shape[0], expected, num_heads, head_dim, weight.shape[0], expected))
    grouped = weight.reshape(num_heads, 3 * head_dim, *weight.shape[1:])
    q, k, v = grouped.split(head_dim, dim=1)
    return [t.reshape(num_heads * head_dim, *weight.shape[1:]).contiguous() for t in (q, k, v)]


def _convert_video_vae_key(source_key, tensor, num_heads, head_dim):
    """Una key del checkpoint original -> la(s) key(s) diffusers que produce.
    One original-checkpoint key -> the diffusers key(s) it maps to."""
    # No son parametros del modulo: viven en la config.
    # Not module parameters: they live in the config.
    if source_key in ("latents_mean", "latents_std", "decoder.mask_token"):
        return []

    if ".attn.to_qkv." in source_key:
        q, k, v = _vae_reorder_interleaved_qkv(tensor, num_heads, head_dim)
        prefix, suffix = source_key.split(".attn.to_qkv.")
        return [
            ("{}.attn.to_q.{}".format(prefix, suffix), q),
            ("{}.attn.to_k.{}".format(prefix, suffix), k),
            ("{}.attn.to_v.{}".format(prefix, suffix), v),
        ]

    target = source_key
    if target.startswith("encoder.down."):
        level, rest = target[len("encoder.down."):].split(".", 1)
        rest = rest.replace("block.", "resnets.", 1)
        rest = rest.replace("nin_shortcut.", "conv_shortcut.", 1)
        rest = rest.replace("downsample.", "downsamplers.0.", 1)
        target = "encoder.down_blocks.{}.{}".format(level, rest)
    target = target.replace("decoder.x_embedder.", "decoder.proj_in.")
    target = target.replace(".attn.to_out.", ".attn.to_out.0.")
    target = target.replace(".ff.w1.", ".ff.net.0.proj.")
    target = target.replace(".ff.w2.", ".ff.net.2.")

    if ".ff.w1." in source_key:
        # Mismo swap SwiGLU que mlp.fc1 en el DiT, en la otra direccion.
        # Same SwiGLU swap as the DiT's mlp.fc1, in the other direction.
        gate, up = tensor.chunk(2, dim=0)
        return [(target, torch.cat([up, gate], dim=0).contiguous())]

    return [(target, tensor)]


def load_preview_video_vae(nf4_dir, device):
    """Carga el VAE de video del repo NF4 traduciendo las keys al layout diffusers.
    Loads the NF4 repo's video VAE, translating the keys to the diffusers layout."""
    cached = _PREVIEW_VAE.get("model")
    if cached is not None and _PREVIEW_VAE.get("device") == device:
        return cached

    release_preview_video_vae()

    from diffusers import AutoencoderKLMiniMaxH3

    vae_dir = os.path.join(nf4_dir, "vae")
    config_path = os.path.join(vae_dir, "config.json")
    candidates = [
        os.path.join(vae_dir, "minimax_h3_video_vae_fp16.safetensors"),
        os.path.join(vae_dir, "minimax_h3_video_vae.safetensors"),
    ]
    ckpt = next((p for p in candidates if os.path.isfile(p)), None)
    if ckpt is None or not os.path.isfile(config_path):
        raise FileNotFoundError(
            "No se encontro el VAE de video en {} (se busco: {}) / video VAE not found"
            .format(vae_dir, [os.path.basename(c) for c in candidates]))

    with open(config_path, "r", encoding="utf-8") as f:
        vae_config = json.load(f)

    num_heads = int(vae_config.get("decoder_num_attention_heads", 32))
    head_dim = int(vae_config.get("decoder_attention_head_dim", 64))

    log_print("[PREVIEW-VAE] Loading decoder from / Cargando decoder desde: {}"
              .format(os.path.abspath(ckpt)), flush=True)

    converted = {}
    with safe_open(ckpt, framework="pt", device="cpu") as f:
        for key in f.keys():
            for new_key, tensor in _convert_video_vae_key(
                    key, f.get_tensor(key), num_heads, head_dim):
                converted[new_key] = tensor.to(torch.float32)

    vae = AutoencoderKLMiniMaxH3.from_config(vae_config)
    missing, unexpected = vae.load_state_dict(converted, strict=False)
    # inv_freq del RoPE del decoder es un buffer NO persistente: se calcula en el
    # __init__ y por eso aparece como "missing". No es un peso que falte.
    # The decoder RoPE's inv_freq is a NON-persistent buffer computed in __init__,
    # so it shows up as missing without being an actual missing weight.
    real_missing = [k for k in missing if not k.endswith("inv_freq")]
    if real_missing or unexpected:
        raise RuntimeError(
            "[PREVIEW-VAE] La traduccion de keys no cuadra: {} faltan, {} sobran. "
            "Faltan: {} | Sobran: {} / key translation mismatch"
            .format(len(real_missing), len(unexpected),
                    real_missing[:6], list(unexpected)[:6]))

    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    if device == "cuda":
        # Solo la ruta de decodificacion: el encoder no se usa nunca aqui y son
        # cientos de MB de VRAM regalados.
        # Decode path only: the encoder is never used here.
        vae.post_quant_conv.to("cuda", dtype=torch.bfloat16)
        vae.decoder.to("cuda", dtype=torch.bfloat16)
    vae.enable_tiling()

    _PREVIEW_VAE["model"] = vae
    _PREVIEW_VAE["device"] = device
    log_print("[PREVIEW-VAE] Ready on / Listo en: {} ({} translated tensors / "
              "tensores traducidos)".format(device, len(converted)), flush=True)
    return vae


def release_preview_video_vae():
    """Libera el VAE cacheado. / Frees the cached VAE."""
    if _PREVIEW_VAE.get("model") is not None:
        _PREVIEW_VAE["model"] = None
        _PREVIEW_VAE["device"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def decode_latent_to_png(latent_norm, nf4_dir, out_path, device="cpu", frame_index=0):
    """Latente NORMALIZADO [1,24,1,H,W] -> PNG.

    Deshace las dos normalizaciones que aplico el pre-cache: primero la del
    latente (z * std + mean) y despues la de pixel, que es ImageNet sobre base
    [0,1] y NO el [-1,1] habitual.
    Undoes the two normalizations the pre-cache applied: the latent one and then
    the pixel one, which is ImageNet over a [0,1] base, NOT the usual [-1,1].
    """
    from PIL import Image

    vae = load_preview_video_vae(nf4_dir, device)

    compute_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    z = latent_norm.detach().to(device=device, dtype=torch.float32)
    if z.ndim == 4:
        z = z.unsqueeze(0)

    lm = torch.tensor(vae.config.latents_mean, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    ls = torch.tensor(vae.config.latents_std, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    z = (z * ls + lm).to(compute_dtype)

    with torch.no_grad():
        # _decode_clip y NO decode(): decode() asume la rejilla temporal 17n+5 y
        # con un unico frame latente calcula num_chunks=0 y devuelve un vacio.
        # _decode_clip, NOT decode(): decode() assumes the 17n+5 temporal grid and
        # computes num_chunks=0 for a single latent frame, returning an empty tensor.
        px = vae._decode_clip(z)

    px = px.float()
    mean = torch.tensor(_IMAGENET_MEAN, device=px.device).view(1, 3, 1, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=px.device).view(1, 3, 1, 1, 1)
    px = (px * std + mean).clamp(0, 1)

    idx = max(0, min(int(frame_index), px.shape[2] - 1))
    frame = px[0, :, idx].permute(1, 2, 0).cpu().numpy()
    Image.fromarray((frame * 255.0).astype("uint8")).save(out_path)
    return out_path


def pick_preview_prompt(entries, step):
    """Elige de donde sale el prompt de esta preview segun PREVIEW_CAPTION_MODE.

    Devuelve (entry, prompt_result, etiqueta). `entry` solo se usa para la
    GEOMETRIA del latente (alto/ancho), asi que en modo "custom" —que no tiene
    imagen asociada— se usa la primera del dataset, igual que hace Krea2.

    Nota sobre `random`: el bucle re-siembra el RNG al principio de CADA paso, y
    la preview corre al FINAL, cuando las tiradas del entrenamiento ya se han
    hecho. Consumir aqui numeros del stream global no altera nada de lo que ve
    el entrenamiento, ni en esta corrida ni al reanudar.

    Returns (entry, prompt_result, label). `entry` is only used for the latent
    GEOMETRY, so "custom" borrows the first image's shape. On `random`: the loop
    re-seeds the RNG at the start of every step and the preview runs at the end,
    so drawing here cannot perturb training.
    """
    mode = PREVIEW_CAPTION_MODE

    if mode == "custom":
        custom = load_prompt_structure(CACHE_DIR, "_custom")
        if custom is not None:
            # Se lee el prompt REALMENTE codificado, no el que hay ahora en el
            # JSON: si se edito el texto y no se relanzo el pre-cache, son
            # distintos y hay que verlo.
            # Read the prompt that was ACTUALLY encoded, not the one currently
            # in the JSON: if the text was edited without re-running the
            # pre-cache they differ, and that has to be visible.
            text = ""
            try:
                text = str(custom.get("prompt", "") or "")
            except Exception:
                pass
            return entries[0], custom, "custom: '{}'".format(
                text[:80] + ("..." if len(text) > 80 else ""))
        log_print("[PREVIEW][WARN] preview_caption_mode='custom' pero la cache no tiene "
                  "_custom_structure.json. El entrenador NO puede codificar texto: "
                  "escribe el prompt en la pestana de Pre-Cache, pulsa Save JSON y "
                  "relanza el Pre-Cache (salta las imagenes ya cacheadas). Se usa el "
                  "primer caption. / no _custom in cache; the trainer cannot encode "
                  "text. Set the prompt in Pre-Cache, save and re-run it. Falling back "
                  "to the first caption.", flush=True)
        mode = "first"

    if mode == "random" and len(entries) > 1:
        entry = random.choice(entries)
    elif mode == "rotate" and len(entries) > 1:
        entry = entries[(step // max(1, PREVIEW_EVERY)) % len(entries)]
    else:
        entry = entries[0]

    return entry, entry["prompt"], "{}: {}".format(mode, entry.get("name", "?"))


def run_training_preview(model, entries, step, output_dir, nf4_dir,
                         patch_t, patch_h, patch_w, audio_channels=32):
    """Genera <output_dir>/preview_step_<step>.png con el LoRA en su estado actual.
    Renders the current LoRA state to <output_dir>/preview_step_<step>.png."""
    from diffusers import MiniMaxH3Scheduler

    entry, prompt_result, prompt_label = pick_preview_prompt(entries, step)
    log_print("[PREVIEW] Prompt / Prompt: {}".format(prompt_label), flush=True)

    was_training = model.training
    model.eval()

    try:
        prompt_embeds, attention_mask = get_prompt_pair(prompt_result)
        text = prompt_embeds.to("cuda", dtype=torch.bfloat16)
        if text.ndim == 2:
            text = text.unsqueeze(0)

        # Mismo recorte de padding que el entrenamiento: el eje t del RoPE del
        # video arranca en origin = text_len, asi que un padding distinto
        # desplazaria la geometria respecto a lo que el LoRA aprendio.
        # Same padding trim as training: the video RoPE t axis starts at text_len.
        if TRIM_TEXT_PADDING and attention_mask is not None:
            try:
                am = attention_mask.detach().reshape(-1)[: text.shape[1]] > 0
                n_real = int(am.sum().item())
                if 0 < n_real < text.shape[1] and bool(am[:n_real].all()) \
                        and not bool(am[n_real:].any()):
                    text = text[:, :n_real, :].contiguous()
            except Exception:
                pass
        if MAX_TEXT_TOKENS > 0 and text.shape[1] > MAX_TEXT_TOKENS:
            text = text[:, :MAX_TEXT_TOKENS, :].contiguous()

        ref = entry["video"]
        if ref.ndim == 4:
            ref = ref.unsqueeze(0)
        ref = align_video_latent_to_patch(ref, patch_h, patch_w, patch_t)
        latent_shape = tuple(ref.shape)
        _, _, n_f, lat_h, lat_w = latent_shape

        # ----------------------------------------------------------------
        # CUANTOS FRAMES LATENTES GENERAR.
        #
        # Por defecto 1: modo imagen, el mismo que entrena. Con
        # preview_num_frames > 1 se genera un clip corto ajustado a la rejilla
        # 17n+5 del VAE, que produce 5n+2 frames latentes, y se guarda un
        # fotograma del centro. Cuesta proporcionalmente mas (la atencion crece
        # con el cuadrado de la secuencia), pero pone al DiT en su regimen
        # nativo de video.
        # How many latent frames to generate. Default 1 (image mode, what
        # training uses). >1 renders a clip on the VAE's 17n+5 grid (5n+2 latent
        # frames) and saves a middle frame.
        # ----------------------------------------------------------------
        preview_frames = max(1, int(PREVIEW_NUM_FRAMES))
        if preview_frames > 1:
            n_chunks = max(0, (preview_frames - 5 + 16) // 17)
            preview_frames = 17 * n_chunks + 5
            n_lat = 5 * n_chunks + 2
        else:
            n_lat = 1
        latent_shape = (latent_shape[0], latent_shape[1], n_lat, lat_h, lat_w)
        if n_lat != n_f:
            log_print("[PREVIEW] Clip mode / modo clip: {} frames -> {} latent frames "
                      "({}x the video tokens / {}x los tokens de video)".format(
                          preview_frames, n_lat, n_lat, n_lat), flush=True)

        seed = PREVIEW_SEED if PREVIEW_SEED > 0 else (SEED if SEED > 0 else random.randint(1, 2 ** 31 - 1))
        gen = torch.Generator(device="cuda").manual_seed(int(seed))
        latent = torch.randn(latent_shape, generator=gen, device="cuda", dtype=torch.float32)
        tokens = patch_video_latent(latent.to(torch.bfloat16), patch_h, patch_w, patch_t).float()

        # Cero filas de audio, igual que el entrenamiento con
        # drop_audio_rows_when_unused: solo importa que la ultima dimension
        # coincida con audio_proj_in.in_features.
        # Zero audio rows, exactly as training does; only the last dim matters.
        # ----------------------------------------------------------------
        # FILAS DE AUDIO: SI van, tambien para una imagen suelta.
        #
        # El entrenamiento las quita (drop_audio_rows_when_unused) porque ahi
        # solo son una perturbacion sin gradiente. Pero al MUESTREAR desde ruido
        # puro el modelo tiene que ver el layout con el que se entreno, y ese
        # layout SIEMPRE lleva audio. Verificado en el pipeline de ai-toolkit:
        # `a_lat` se calcula sin condicion, tambien con num_frames == 1, y su
        # flag `with_audio` solo decide si al final se DECODIFICA la forma de
        # onda, no si las filas estan en la secuencia.
        #
        # Para un frame: a_lat = round(1 / 24 fps * 40 latentes/s) = 2, por 2
        # canales = 4 filas. Son 4 de ~620, pero atraviesan los 50 bloques.
        #
        # AUDIO ROWS ARE INCLUDED, even for a still image. Training drops them
        # (they carry no gradient there), but sampling from pure noise has to
        # show the model the layout it was trained on, and that layout always
        # carries audio. Verified against ai-toolkit's pipeline, where `a_lat`
        # is computed unconditionally and `with_audio` only controls whether the
        # waveform is decoded at the end.
        # ----------------------------------------------------------------
        _AUDIO_LATENTS_PER_SECOND = 40.0
        a_lat = max(1, int(round(float(preview_frames) / float(FRAME_RATE)
                                 * _AUDIO_LATENTS_PER_SECOND)))
        a_rows = a_lat * int(audio_channels)
        audio = torch.randn((1, a_rows, 32), generator=gen, device="cuda",
                            dtype=torch.float32)

        (timestep_indices, token_tags, position_ids,
         video_indices, audio_indices, text_indices) = build_minimax_packed_indices(
            B=1, text_len=text.shape[1], video_len=tokens.shape[1], audio_len=a_rows,
            video_latent_shape=latent_shape,
            patch_t=patch_t, patch_h=patch_h, patch_w=patch_w, device="cuda",
            audio_channels=int(audio_channels),
        )

        # ----------------------------------------------------------------
        # DOS TIMESTEPS EN EL MISMO FORWARD.
        #
        # El video y el audio recorren schedules DISTINTOS: shift del video (el
        # que elija el usuario) y shift 3.0 el audio, acoplados por el remap de
        # sigma para que ambos esten en la misma posicion del schedule. El
        # forward de H3 admite varios timestep a la vez: `timestep` lleva los
        # valores distintos y `timestep_indices` dice cual usa cada fila.
        # Texto y video -> 0, audio -> 1.
        #
        # TWO TIMESTEPS IN ONE FORWARD: video and audio run different sigma
        # schedules, coupled by the closed-form shift remap. H3's forward takes
        # the distinct values in `timestep` and the per-row index in
        # `timestep_indices`.
        # ----------------------------------------------------------------
        timestep_indices = timestep_indices.clone()
        timestep_indices[audio_indices] = 1

        scheduler = MiniMaxH3Scheduler(shift=float(PREVIEW_SHIFT))
        scheduler.set_timesteps(int(PREVIEW_STEPS) + 1, device="cuda")

        def _remap_sigma(sigma, from_shift, to_shift):
            """Sigma del schedule `from_shift` -> el equivalente en `to_shift`,
            en la misma posicion subyacente. Es el acoplamiento video/audio que
            usa la implementacion de referencia.
            Maps a sigma from one shift schedule onto another at the same
            underlying position: the reference video/audio coupling."""
            base = sigma / (from_shift + sigma * (1.0 - from_shift))
            return to_shift * base / (1.0 + (to_shift - 1.0) * base)

        _AUDIO_SIGMA_SHIFT = 3.0
        sigmas_v = scheduler.sigmas
        sigmas_a = _remap_sigma(sigmas_v, float(PREVIEW_SHIFT), _AUDIO_SIGMA_SHIFT)

        neg_text = None
        if PREVIEW_CFG > 1.0:
            neg = load_prompt_structure(CACHE_DIR, "_neg")
            if neg is not None:
                neg_embeds, _ = get_prompt_pair(neg)
                if neg_embeds is not None:
                    neg_text = neg_embeds.to("cuda", dtype=torch.bfloat16)
                    if neg_text.ndim == 2:
                        neg_text = neg_text.unsqueeze(0)
                    # El layout depende de la longitud del texto, asi que un
                    # negativo de distinta longitud exigiria otra secuencia
                    # empaquetada. Se iguala por recorte/relleno.
                    # The layout depends on text length, so the negative is
                    # trimmed/padded to match.
                    if neg_text.shape[1] > text.shape[1]:
                        neg_text = neg_text[:, : text.shape[1], :]
                    elif neg_text.shape[1] < text.shape[1]:
                        pad = text.shape[1] - neg_text.shape[1]
                        neg_text = torch.cat(
                            [neg_text, torch.zeros_like(neg_text[:, :1, :]).repeat(1, pad, 1)], dim=1)
            if neg_text is None:
                log_print("[PREVIEW][WARN] preview_cfg > 1 pero la cache no tiene el "
                          "prompt vacio (_neg). Se genera sin CFG. / no empty prompt in "
                          "cache; running without CFG.", flush=True)

        # El transformer envuelto por PEFT no acepta kwargs que no existan en su
        # forward real; filter_forward_kwargs los descarta como en el train loop.
        # The PEFT-wrapped transformer rejects kwargs its real forward lacks.
        transformer_forward = model.forward

        def _forward(hidden, audio_hidden, text_cond, t_pair):
            kwargs = {
                "hidden_states": hidden.to(torch.bfloat16),
                "audio_hidden_states": audio_hidden.to(torch.bfloat16),
                "encoder_hidden_states": text_cond,
                # Los DOS timesteps distintos: indice 0 = video/texto, 1 = audio.
                # Both distinct timesteps: index 0 = video/text, 1 = audio.
                "timestep": t_pair,
                "timestep_indices": timestep_indices,
                "token_tags": token_tags,
                "position_ids": position_ids,
                "video_indices": video_indices,
                "audio_indices": audio_indices,
                "text_indices": text_indices,
                "return_dict": False,
            }
            out = model(**filter_forward_kwargs(kwargs, transformer_forward))
            if isinstance(out, (tuple, list)):
                return out[0].float(), (out[1].float() if len(out) > 1 else None)
            return out.sample.float(), getattr(out, "audio_sample", None)

        with torch.no_grad():
            for i, t in enumerate(scheduler.timesteps):
                sv, sv_next = float(sigmas_v[i]), float(sigmas_v[i + 1])
                sa, sa_next = float(sigmas_a[i]), float(sigmas_a[i + 1])
                t_pair = torch.tensor([1.0 - sv, 1.0 - sa],
                                      device="cuda", dtype=torch.bfloat16)

                pred, pred_audio = _forward(tokens, audio, text, t_pair)
                if neg_text is not None:
                    pred_u, _ = _forward(tokens, audio, neg_text, t_pair)
                    pred = pred_u + float(PREVIEW_CFG) * (pred - pred_u)

                # Video: por el scheduler de diffusers, que ya esta probado.
                # Video through the diffusers scheduler, which is tested.
                tokens = scheduler.step(pred, t, tokens, return_dict=False)[0]

                # Audio: Euler manual sobre SU rejilla de sigma, igual que la
                # implementacion de referencia. El audio no se decodifica nunca
                # aqui; se denoisa solo para que la secuencia siga siendo la que
                # el modelo espera en cada paso.
                # Audio: manual Euler on ITS own sigma grid, as the reference
                # does. The audio is never decoded here; it is denoised only so
                # the sequence stays the one the model expects at every step.
                if pred_audio is not None and pred_audio.shape[1] == audio.shape[1] and sa > 0:
                    denoised_a = audio.float() + sa * pred_audio
                    ratio_a = sa_next / sa
                    audio = (ratio_a * audio.float() + (1.0 - ratio_a) * denoised_a)

        latent_out = unpack_video_latent(tokens, latent_shape, patch_h, patch_w, patch_t)

        free_vram(clear_cache=True, collect=True)

        out_path = os.path.join(output_dir, "preview_step_{}.png".format(step))
        # Con un clip, el fotograma del centro: los extremos temporales son los
        # que peor se resuelven. Con una imagen suelta manda preview_frame_index.
        # For a clip, take a middle frame: the temporal edges resolve worst.
        _frame_idx = PREVIEW_FRAME_INDEX if n_lat == 1 else (n_lat * 4) // 2
        decode_latent_to_png(latent_out.cpu(), nf4_dir, out_path,
                             device=PREVIEW_VAE_DEVICE, frame_index=_frame_idx)
        return out_path

    finally:
        if was_training:
            model.train()
        free_vram(clear_cache=True, collect=True)


# =============================================================================
# GLOBAL CONFIG REFERENCE
# =============================================================================

CURRENT_CONFIG = None
CURRENT_TRANSFORMER = None
PATCH_T = 1
PATCH_H = 1
PATCH_W = 1

# =========================================================================
# RUNTIME PERSISTENTE PARA PARAR / REANUDAR SIN RECARGAR NF4
# =========================================================================
TRAIN_RUNTIME = globals().get("TRAIN_RUNTIME", None)
STOP_REQUESTED = globals().get("STOP_REQUESTED", False)


def reset_train_runtime(clear_memory=True):
    """
    Fuerza recarga completa en la siguiente llamada.
    Úsalo si quieres reconstruir todo desde disco.
    """
    global TRAIN_RUNTIME, STOP_REQUESTED

    rt = TRAIN_RUNTIME
    if isinstance(rt, dict):
        for h in rt.get("all_hooks", []):
            try:
                h.remove()
            except Exception:
                pass
        rt.clear()

    TRAIN_RUNTIME = None
    STOP_REQUESTED = False

    if clear_memory:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _runtime_cache_key():
    """
    Clave para detectar si el runtime cargado sigue siendo válido.
    No incluimos debug_training porque cambiarlo no debería obligar a recargar.
    """
    return json.dumps(
        {
            "nf4_cache_dir": NF4_CACHE_DIR,
            "cache_dir": CACHE_DIR,
            "output_dir": OUTPUT_DIR,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_only_attn": LORA_ONLY_ATTN,
            # Cambiar el dtype de los pesos LoRA o los modulos objetivo obliga a
            # reconstruir modelo y optimizador: sin esto, un A/B desde la GUI
            # reutilizaba el runtime viejo y comparaba dos veces lo mismo.
            "lora_dtype": LORA_DTYPE_STR,
            "lora_exclude_refiner": LORA_EXCLUDE_REFINER,
            "lora_skip_first_n_blocks": LORA_SKIP_FIRST_N_BLOCKS,
            "lora_key_prefix": LORA_KEY_PREFIX,
            "use_audio_loss": USE_AUDIO_LOSS,
            # Cambiar de optimizador exige reconstruir el runtime: los objetos
            # optimizador se crean una sola vez al cargar.
            # Switching optimizer requires rebuilding the runtime: the optimizer
            # objects are only built once, at load time.
            "optimizer_type": OPTIMIZER_TYPE,
            "checkpoint_use_reentrant": CHECKPOINT_USE_REENTRANT,
            "cpu_offload_blocks_enabled": CPU_OFFLOAD_BLOCKS_ENABLED,
            "audio_cpu_offload_enabled": AUDIO_CPU_OFFLOAD_ENABLED,
            "vram_budget_gb": VRAM_BUDGET_GB,
            "vram_swap_gb": VRAM_SWAP_GB,
            "vram_headroom_gb": VRAM_HEADROOM_GB,
            "cast_frozen_bf16": CAST_FROZEN_BF16,
            # Sin estas dos, cambiarlas en el JSON no reconstruia el transformer y el
            # A/B comparaba dos veces exactamente el mismo modelo.
            # Without these two, changing them in the JSON did not rebuild the
            # transformer and the A/B compared the exact same model twice.
            "fp32_repair_enabled": FP32_REPAIR_ENABLED,
            "cast_frozen_respect_fp32_modules": CAST_FROZEN_RESPECT_FP32,
        },
        sort_keys=True,
    )


def _ensure_train_runtime():
    """
    Carga una sola vez:
      - transformer NF4
      - dataset cacheado
      - LoRA
      - offload / hooks
      - optimizer

    Si ya existe un runtime válido, lo reutiliza directamente.
    """
    global TRAIN_RUNTIME
    global CURRENT_CONFIG, CURRENT_TRANSFORMER, PATCH_T, PATCH_H, PATCH_W

    key = _runtime_cache_key()

    if isinstance(TRAIN_RUNTIME, dict) and TRAIN_RUNTIME.get("loaded", False):
        if TRAIN_RUNTIME.get("key", None) == key:
            log_print("[CACHE] Runtime ya cargado. No se recarga NF4 ni dataset.", flush=True)
            return TRAIN_RUNTIME

        log_print("[CACHE] Runtime previo incompatible con la configuración actual. Reiniciando.", flush=True)
        reset_train_runtime(clear_memory=False)

    # ------------------------------------------------------------------
    # La pre-cache se valida ANTES de tocar el modelo. Cargar 39 GB de NF4 para
    # despues descubrir que no hay latentes cuesta varios minutos y suelta un
    # error enorme que no dice lo unico que importa: que falta la pre-cache.
    # No basta con que exista la carpeta: una pre-cache interrumpida deja el
    # directorio creado y vacio, y ese caso pasaba el filtro.
    #
    # The pre-cache is validated BEFORE touching the model. Loading 39 GB of NF4
    # only to find there are no latents costs minutes and throws a huge error
    # that hides the one thing that matters: the pre-cache is missing. Checking
    # that the folder exists is not enough — an interrupted pre-cache leaves it
    # created and empty, and that case used to slip through.
    # ------------------------------------------------------------------
    _latents = []
    if os.path.isdir(CACHE_DIR):
        try:
            _latents = [f for f in os.listdir(CACHE_DIR)
                        if f.endswith("_video_latent.pt")]
        except Exception:
            _latents = []

    if not _latents:
        _why = ("the folder does not exist / la carpeta no existe"
                if not os.path.isdir(CACHE_DIR)
                else "the folder is empty or has no latents / la carpeta esta "
                     "vacia o no tiene latentes")
        raise RuntimeError(
            "\n"
            "================================================================\n"
            "  NO PRE-CACHE FOR THIS PROJECT / NO HAY PRE-CACHE DE ESTE PROYECTO\n"
            "================================================================\n"
            "  Folder / Carpeta : {}\n"
            "  Reason / Motivo  : {}\n"
            "\n"
            "  Run step 1 (Pre-Cache) before training. Training has no text\n"
            "  encoder and no VAE by design: every embedding and latent must\n"
            "  be computed beforehand.\n"
            "\n"
            "  Ejecuta primero el paso 1 (Pre-Cache). El entrenamiento no lleva\n"
            "  text encoder ni VAE por diseno: todos los embeddings y latentes\n"
            "  tienen que calcularse antes.\n"
            "================================================================\n"
            .format(os.path.abspath(CACHE_DIR), _why))

    if not os.path.isdir(NF4_CACHE_DIR):
        raise FileNotFoundError(
            "No existe NF4_CACHE_DIR: {}. Ejecuta primero el conversor NF4.".format(NF4_CACHE_DIR)
        )

    log_print("[CACHE] Primera ejecución en este proceso Python: cargando NF4 y dataset...", flush=True)

    # El tope se aplica ANTES de cargar nada: si el modelo no cabe en la GPU
    # simulada, tiene que fallar aqui y no 40 minutos despues.
    # Applied BEFORE loading anything: if it doesn't fit the simulated GPU it must
    # fail here, not 40 minutes later.
    apply_vram_hard_cap()

    transformer = load_transformer_from_nf4(NF4_CACHE_DIR)
    transformer.requires_grad_(False)

    try:
        _sig_params = list(inspect.signature(transformer.forward).parameters.keys())
        log_print("[FORWARD-SIG] transformer.forward acepta: {}".format(_sig_params), flush=True)
    except Exception as _e:
        log_print("[FORWARD-SIG] No se pudo inspeccionar la firma: {}".format(_e), flush=True)

    try:
        _cfg_path = os.path.join(_find_transformer_cache_dir(NF4_CACHE_DIR) or "", "config.json")
        if os.path.exists(_cfg_path):
            with open(_cfg_path, "r", encoding="utf-8") as _f:
                _cfg_json = json.load(_f)
            log_print("[CHECKPOINT] partition/variant relevante del config.json: {}".format(
                {k: v for k, v in _cfg_json.items()
                 if any(s in k.lower() for s in ("pruned", "partition", "variant", "task"))}
            ), flush=True)
    except Exception as _e:
        log_print("[CHECKPOINT] No se pudo leer config.json para detectar variante: {}".format(_e), flush=True)

    if CAST_FROZEN_BF16:
        cast_frozen_to_bf16(transformer)

    free_vram()

    config = getattr(transformer, "config", {})

    audio_channels = safe_int(
        config_get(config, "audio_in_channels", 128),
        128,
    )

    patch_t, patch_h, patch_w = get_patch_sizes(config)

    log_print("[CONFIG] patch_size t/h/w: {}/{}/{}".format(patch_t, patch_h, patch_w), flush=True)
    log_print("[CONFIG] audio_in_channels: {}".format(audio_channels), flush=True)

    entries = load_cached_entries(CACHE_DIR, audio_channels)
    log_print("[OK] {} entradas de dataset cargadas.".format(len(entries)), flush=True)

    sample_video = entries[0]["video"]
    sample_F = sample_video.shape[2] if sample_video.ndim == 5 else 1

    if patch_t > 1 and sample_F < patch_t:
        raise RuntimeError(
            "El cache tiene latentes de vídeo con F={} frame(s) pero el transformer "
            "usa patch_size_t={}. align_video_latent_to_patch() dejaría el vídeo vacío. "
            "El pre-cache de imagen de H3 usa T=1 a propósito; si tu transformer "
            "requiere patch_size_t>1, este cache de imagen no es compatible tal cual "
            "y hay que revisar el pipeline de imagen->vídeo del pre-cache.".format(
                sample_F, patch_t
            )
        )

    inspect_transformer_for_lora(transformer)
    target_modules = discover_lora_targets(transformer)

    log_print("Target LoRA Layers: {}".format(len(target_modules)), flush=True)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        target_modules=target_modules,
        bias="none",
        task_type=None,
        init_lora_weights=True,
    )

    model = get_peft_model(transformer, lora_config)

    # ------------------------------------------------------------------
    # PRECISION DE LOS PESOS ENTRENABLES.
    #
    # Con LORA_DTYPE=fp32 los pesos LoRA son master weights fp32 y, como AdamW crea
    # su estado con zeros_like(param), exp_avg/exp_avg_sq TAMBIEN son fp32. Con bf16
    # (lo que habia) el estado del optimizador era bf16 y los updates pequenos se
    # redondeaban a cero: el LoRA aprendia lo grueso y la cara nunca cerraba.
    # El computo sigue en bf16 gracias a autocast, asi que solo cuesta memoria.
    #
    # fp32 LoRA = fp32 master weights AND fp32 Adam state (zeros_like). bf16 params
    # meant bf16 optimizer state and updates rounding to zero.
    # ------------------------------------------------------------------
    for module in model.modules():
        if hasattr(module, "lora_A"):
            for adapter in module.lora_A.values():
                adapter.to(dtype=LORA_DTYPE)

        if hasattr(module, "lora_B"):
            for adapter in module.lora_B.values():
                adapter.to(dtype=LORA_DTYPE)

    log_print("[LoRA] dtype de los pesos entrenables: {} (estado de AdamW hereda este "
              "dtype) / trainable dtype: {}".format(LORA_DTYPE, LORA_DTYPE), flush=True)

    print_block_diagnostics(transformer)

    offload_info = None
    excluded_audio_ids = set()
    offload_hooks = []

    if CPU_OFFLOAD_BLOCKS_ENABLED:
        offload_info = setup_block_cpu_offload(
            transformer,
            target_vram_gb=VRAM_BUDGET_GB,
            reserve_gb=VRAM_HEADROOM_GB,
        )

        if offload_info is not None:
            excluded_audio_ids = offload_info["excluded_audio_ids"]
            offload_hooks = offload_info.get("swap_hooks", [])
        else:
            model.to("cuda")
    else:
        model.to("cuda")

    setup_gradient_checkpointing_optimized(transformer, model, offload_info)
    ensure_trainable_parameters_on_cuda(model)
    enable_memory_efficient_attention(transformer)

    if AUDIO_CPU_OFFLOAD_ENABLED:
        audio_hooks = apply_audio_cpu_offload(model, exclude_ids=excluded_audio_ids)
    else:
        audio_hooks = []
        log_print("[AUDIO] CPU offload de audio desactivado por configuración.", flush=True)

    model.print_trainable_parameters()

    input_hooks = []

    def make_inputs_require_grad(module, inputs, output):
        if not torch.is_grad_enabled():
            return

        def _try_require_grad(x):
            if isinstance(x, torch.Tensor) and x.is_floating_point():
                try:
                    x.requires_grad_(True)
                except Exception:
                    pass

        if isinstance(output, torch.Tensor):
            _try_require_grad(output)
        elif isinstance(output, (tuple, list)):
            for item in output:
                _try_require_grad(item)

    for name in ("proj_in", "video_in", "audio_in", "x_embedder", "context_embedder"):
        if hasattr(transformer, name):
            try:
                input_hooks.append(
                    getattr(transformer, name).register_forward_hook(make_inputs_require_grad)
                )
                log_print("[VRAM] Hook registrado en '{}'.".format(name), flush=True)
                break
            except Exception:
                pass

    trainable_gpu = [
        p for p in model.parameters()
        if p.requires_grad and p.device.type == "cuda"
    ]

    trainable_cpu = [
        p for p in model.parameters()
        if p.requires_grad and p.device.type == "cpu"
    ]

    trainable = trainable_gpu + trainable_cpu

    log_print(
        "[OPT] Parámetros entrenables: {} en GPU, {} en CPU.".format(
            len(trainable_gpu), len(trainable_cpu)
        ),
        flush=True,
    )

    gpu_optimizer = None
    if trainable_gpu:
        if OPTIMIZER_TYPE == "adamw8bit":
            gpu_optimizer = bnb.optim.PagedAdamW8bit(
                trainable_gpu,
                lr=LR,
                weight_decay=WEIGHT_DECAY,
            )
            log_print(
                "[OPT] GPU: bnb.optim.PagedAdamW8bit (estado del optimizador en 8 bits). "
                "En H3 esto deja la cara blanda; considera optimizer_type='adamw'.",
                flush=True,
            )
        else:
            gpu_optimizer = torch.optim.AdamW(
                trainable_gpu,
                lr=LR,
                weight_decay=WEIGHT_DECAY,
            )
            log_print(
                "[OPT] GPU: torch.optim.AdamW (estado del optimizador en fp32).",
                flush=True,
            )

    cpu_optimizer = None
    if trainable_cpu:
        cpu_optimizer = torch.optim.AdamW(
            trainable_cpu,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
        log_print(
            "[OPT] {} parámetros LoRA en CPU usarán torch.optim.AdamW normal.".format(
                len(trainable_cpu)
            ),
            flush=True,
        )

    optimizer = DualOptimizer(gpu_optimizer, cpu_optimizer)

    runtime = {
        "loaded": True,
        "key": key,
        "transformer": transformer,
        "model": model,
        "entries": entries,
        "config": config,
        "audio_channels": audio_channels,
        "patch": (patch_t, patch_h, patch_w),
        "offload_info": offload_info,
        "input_hooks": _as_hook_list(input_hooks, "input_hooks"),
        "audio_hooks": _as_hook_list(audio_hooks, "audio_hooks"),
        "offload_hooks": _as_hook_list(offload_hooks, "offload_hooks"),
        "all_hooks": (_as_hook_list(input_hooks, "input_hooks")
                      + _as_hook_list(audio_hooks, "audio_hooks")
                      + _as_hook_list(offload_hooks, "offload_hooks")),
        "trainable": trainable,
        "optimizer": optimizer,
        "last_step": 0,
        "ema_loss": None,
        "running_loss": 0.0,
        "resume_checked": False,
        "completed": False,
    }

    # --- RESUME LOGIC (solo la primera vez que se crea el runtime) ---
    if not runtime.get("resume_checked", False):
        adapter_path = os.path.join(RESUME_DIR, "adapter_model.safetensors")
        adapter_cfg_path = os.path.join(RESUME_DIR, "adapter_config.json")

        resume_compatible = True
        start_step = 0
        resumed_ema_loss = None
        resumed_running_loss = 0.0
        resumed_elapsed = 0.0

        if os.path.exists(adapter_cfg_path):
            try:
                with open(adapter_cfg_path, "r", encoding="utf-8") as f:
                    acfg = json.load(f)
                saved_r = int(acfg.get("r", -1))
                if saved_r != LORA_RANK:
                    resume_compatible = False
                    log_print("=" * 65)
                    log_print("[!] Checkpoint INCOMPATIBLE: rank guardado={}, actual={}.".format(saved_r, LORA_RANK))
                    log_print("=" * 65)
            except Exception:
                pass

        if resume_compatible and os.path.exists(adapter_path) and os.path.exists(STEP_FILE):
            log_print("=" * 65)
            log_print("Checkpoint detected! Restoring state...", flush=True)
            try:
                with open(STEP_FILE, "r", encoding="utf-8") as f:
                    start_step = int(f.read().strip())

                state = load_file(adapter_path, device="cpu")
                set_peft_model_state_dict(model, state)

                if os.path.exists(OPT_FILE):
                    try:
                        optimizer.load_state_dict(torch.load(OPT_FILE, weights_only=False))
                        log_print("Optimizer restaurado.", flush=True)
                    except Exception:
                        log_print("[!] No se pudo restaurar optimizer.", flush=True)

                if os.path.exists(LOSS_STATE_FILE):
                    try:
                        with open(LOSS_STATE_FILE, "r", encoding="utf-8") as f:
                            loss_state = json.load(f)
                        resumed_ema_loss = loss_state.get("ema_loss", None)
                        resumed_running_loss = float(loss_state.get("running_loss", 0.0) or 0.0)
                        # Tiempo ya invertido en corridas anteriores: sin esto, el
                        # "tiempo total" de una corrida reanudada mentiria y solo
                        # contaria el ultimo tramo.
                        # Time already spent in previous runs; without it the total
                        # of a resumed run would only count the last stretch.
                        resumed_elapsed = float(loss_state.get("elapsed_seconds", 0.0) or 0.0)
                        if resumed_ema_loss is not None:
                            resumed_ema_loss = float(resumed_ema_loss)
                        log_print("Loss EMA restaurado: {}".format(resumed_ema_loss), flush=True)
                    except Exception as e:
                        log_print("[!] No se pudo restaurar estado del loss: {}".format(e), flush=True)

                log_print("Resuming from step {}...".format(start_step), flush=True)
            except Exception as e:
                log_print("[!] Warning reading checkpoint: {}".format(e), flush=True)
                start_step = 0
            log_print("=" * 65)

        runtime["last_step"] = start_step
        runtime["ema_loss"] = resumed_ema_loss
        runtime["running_loss"] = resumed_running_loss
        runtime["elapsed_seconds"] = resumed_elapsed
        runtime["resume_checked"] = True

    TRAIN_RUNTIME = runtime

    CURRENT_CONFIG = config
    CURRENT_TRANSFORMER = transformer
    PATCH_T, PATCH_H, PATCH_W = patch_t, patch_h, patch_w

    log_print("[CACHE] Runtime cargado y conservado en RAM/VRAM para parar/reanudar.", flush=True)

    return runtime


# =============================================================================
# AUDIO CPU OFFLOAD
# =============================================================================

def _as_hook_list(value, label):
    """Normaliza a lista y avisa si alguien devolvió None.
    Normalize to a list and report when something returned None.

    Un None aquí reventaba el arranque con 'can only concatenate list (not NoneType)'.
    A None here crashed startup with 'can only concatenate list (not NoneType)'.
    """
    if value is None:
        print("[HOOKS][WARN] '{}' was None; using an empty list. Hooks of this kind will "
              "not be tracked or removed. / '{}' era None; se usa lista vacia. Los hooks de "
              "este tipo no se registraran ni se eliminaran.".format(label, label), flush=True)
        return []
    try:
        return list(value)
    except TypeError:
        print("[HOOKS][WARN] '{}' is not iterable ({}); using an empty list. / '{}' no es "
              "iterable ({}); se usa lista vacia.".format(label, type(value).__name__,
                                                          label, type(value).__name__),
              flush=True)
        return []


def apply_audio_cpu_offload(root_module, exclude_ids=None):
    exclude_ids = exclude_ids or set()
    audio_modules = []

    for name, module in root_module.named_modules():
        if "audio" in name.lower() and id(module) not in exclude_ids:
            audio_modules.append(module)

    if not audio_modules:
        return []

    def pre_hook(m, inp):
        if not any(p.is_cuda for p in m.parameters()):
            m.to("cuda")

    def post_hook(m, inp, out):
        m.to("cpu")

    hooks = []

    for m in audio_modules:
        m.to("cpu")
        hooks.append(m.register_forward_pre_hook(pre_hook))
        hooks.append(m.register_forward_hook(post_hook))

    log_print("[AUDIO] CPU offload aplicado a {} módulos de audio.".format(len(audio_modules)), flush=True)

    return hooks


# =============================================================================
# TRAIN
# =============================================================================

# =============================================================================
# AJUSTES EN CALIENTE / LIVE SETTINGS
# =============================================================================
#
# Permite cambiar ajustes CON EL ENTRENAMIENTO CORRIENDO: se editan los campos
# en la GUI, se pulsa "Save JSON" y el cambio entra en el paso siguiente. El
# servidor escribe train_settings.json aunque el subproceso del entrenador este
# vivo, asi que no hace falta nada mas por su parte.
#
# NOTA HISTORICA: esto NO existia en el entrenador de Krea2. Alli
# train_settings.json se lee UNA sola vez, en el import (linea 63), y ninguna
# funcion lo vuelve a leer. Lo que se podia hacer era Stop -> editar -> Start,
# que relanza el proceso y reanuda desde el checkpoint: el efecto se parece,
# pero pasa por parar. Aqui el cambio entra sin parar nada.
#
# QUE SE PUEDE CAMBIAR Y QUE NO
# -----------------------------
# Solo esta la lista blanca de abajo, y es deliberado. Todo lo demas (rank,
# alpha, optimizador, dtype, presupuestos de VRAM, carpetas) esta horneado en el
# runtime: el modelo, el optimizador y el plan de block swap se construyen UNA
# vez al arrancar, y cambiarlos a mitad exigiria reconstruirlo todo. Esos siguen
# necesitando Stop -> editar -> Start/Resume, que ya funciona.
#
# total_steps es el caso especial: se puede REDUCIR (el bucle termina de forma
# ordenada en cuanto lo alcanza y guarda el LoRA final), pero no AMPLIAR, porque
# el rango del bucle se fija al entrar. Ampliarlo se avisa por pantalla.
#
# Se mira la fecha de modificacion del fichero antes de abrirlo, asi que el
# coste por paso es un os.path.getmtime y nada mas.
#
# Lets settings change WHILE TRAINING: edit the fields, press "Save JSON", the
# change lands on the next step. Only the whitelist below is live; everything
# else is baked into the runtime at startup and still needs Stop -> Start.
# A getmtime guard means the per-step cost is one stat call.
# =============================================================================

_LIVE_SETTINGS_STATE = {"mtime": None}


def hot_reload_live_settings():
    """Relee train_settings.json si ha cambiado y aplica la lista blanca.
    Devuelve una lista de textos "clave: viejo -> nuevo".
    Re-reads train_settings.json when it changed and applies the whitelist."""
    global PREVIEW_EVERY, PREVIEW_STEPS, PREVIEW_CFG, PREVIEW_SHIFT, PREVIEW_SEED
    global PREVIEW_VAE_DEVICE, PREVIEW_FRAME_INDEX, PREVIEW_CAPTION_MODE
    global PREVIEW_NUM_FRAMES
    global SAVE_EVERY, LR, MAX_GRAD_NORM, TOTAL_STEPS, DEBUG_TRAINING

    try:
        mtime = os.path.getmtime(CONFIG_PATH)
    except OSError:
        return []

    if _LIVE_SETTINGS_STATE["mtime"] is None:
        # Primera llamada: solo se memoriza la fecha. Sin esto, el guardado que
        # la propia GUI hace justo antes de lanzar el entrenamiento se
        # interpretaria como un cambio en caliente en el paso 1.
        # First call only memorizes the timestamp, so the GUI's own save right
        # before launching is not reported as a live change on step 1.
        _LIVE_SETTINGS_STATE["mtime"] = mtime
        return []

    if mtime == _LIVE_SETTINGS_STATE["mtime"]:
        return []
    _LIVE_SETTINGS_STATE["mtime"] = mtime

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return []
        raw = {str(k).strip(): v for k, v in raw.items()}
    except Exception:
        # Un guardado a medias deja el JSON invalido un instante. Se marca con un
        # centinela que NO puede ser un mtime real, para que el paso siguiente
        # vuelva a leerlo. Usar None aqui seria un error: None significa "primera
        # llamada" y esa rama solo memoriza la fecha, asi que el cambio que venia
        # en ese guardado se perderia para siempre.
        # A half-written save leaves invalid JSON for an instant. Mark it with a
        # sentinel that can never be a real mtime so the next step re-reads it.
        # None would be wrong here: None means "first call" and that branch only
        # memorizes the timestamp, silently swallowing the pending change.
        _LIVE_SETTINGS_STATE["mtime"] = -1.0
        return []

    changes = []

    def _fresh(key, caster, current):
        """Devuelve el valor nuevo si la clave existe, casea y ha cambiado; si no,
        el actual. Anota el cambio de paso.
        Returns the new value when the key exists, casts and differs."""
        if key not in raw:
            return current
        try:
            value = caster(raw[key])
        except Exception:
            return current
        if value != current:
            changes.append("{}: {} -> {}".format(key, current, value))
        return value

    def _as_bool(v):
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in ("1", "true", "yes", "on")

    def _cast_mode(v):
        v = str(v).strip().lower()
        return v if v in PREVIEW_CAPTION_MODES else PREVIEW_CAPTION_MODE

    PREVIEW_CAPTION_MODE = _fresh("preview_caption_mode", _cast_mode, PREVIEW_CAPTION_MODE)
    PREVIEW_EVERY = _fresh("preview_every", lambda v: int(v or 0), PREVIEW_EVERY)
    PREVIEW_STEPS = _fresh("preview_steps", lambda v: max(2, int(v or 12)), PREVIEW_STEPS)
    PREVIEW_CFG = _fresh("preview_cfg", lambda v: float(v or 0.0), PREVIEW_CFG)
    PREVIEW_SHIFT = _fresh("preview_shift", lambda v: float(v or 12.0), PREVIEW_SHIFT)
    PREVIEW_SEED = _fresh("preview_seed", lambda v: int(v if v is not None else -1), PREVIEW_SEED)
    PREVIEW_FRAME_INDEX = _fresh("preview_frame_index", lambda v: int(v or 0), PREVIEW_FRAME_INDEX)
    PREVIEW_NUM_FRAMES = _fresh("preview_num_frames", lambda v: max(1, int(v or 1)),
                                PREVIEW_NUM_FRAMES)

    def _cast_device(v):
        v = str(v).strip().lower()
        return v if v in ("cpu", "cuda") else PREVIEW_VAE_DEVICE

    _old_device = PREVIEW_VAE_DEVICE
    PREVIEW_VAE_DEVICE = _fresh("preview_vae_device", _cast_device, PREVIEW_VAE_DEVICE)
    if PREVIEW_VAE_DEVICE != _old_device:
        # El VAE cacheado esta en el dispositivo viejo: hay que soltarlo para
        # que la siguiente preview lo recargue donde toca.
        # The cached VAE sits on the old device; drop it so it reloads.
        release_preview_video_vae()

    SAVE_EVERY = _fresh("save_every", lambda v: int(v or 0), SAVE_EVERY)
    LR = _fresh("lr", float, LR)
    MAX_GRAD_NORM = _fresh("max_grad_norm", float, MAX_GRAD_NORM)
    DEBUG_TRAINING = _fresh("debug_training", _as_bool, DEBUG_TRAINING)

    if "total_steps" in raw:
        try:
            new_total = int(raw["total_steps"])
        except Exception:
            new_total = TOTAL_STEPS
        if new_total != TOTAL_STEPS:
            if new_total > TOTAL_STEPS:
                changes.append(
                    "total_steps: {} -> {} (la corrida se amplia) / the run is "
                    "extended".format(TOTAL_STEPS, new_total))
            else:
                changes.append("total_steps: {} -> {} (la corrida terminara antes) / the "
                               "run will finish earlier".format(TOTAL_STEPS, new_total))
            TOTAL_STEPS = new_total

    return changes


def reload_runtime_config():
    """Re-lee train_settings.json en caliente / Re-read train_settings.json at run time.

    La GUI reutiliza el mismo proceso de Python entre lanzamientos, así que las constantes
    de nivel de módulo se quedaban congeladas del primer import: editar el JSON no tenía
    ningún efecto hasta cerrar y abrir la aplicación.
    The GUI reuses the same Python process across launches, so module-level constants stayed
    frozen from the first import: editing the JSON had no effect until restarting the app.
    """
    global cfg
    global TIMESTEP_CONVENTION, FLOW_CONV_DEBUG_STEPS, FLOW_TARGET_SIGN, SIGMA_SHIFT
    global LOGIT_NORMAL_MU, LOGIT_NORMAL_STD, SIGMA_RESOLUTION_SHIFT
    global BF16_REDUCED_PRECISION, DATASET_SAMPLER
    global LORA_DTYPE_STR, LORA_DTYPE, LORA_EXCLUDE_REFINER
    # Antes solo se releian 6 globales, asi que un A/B sobre fp32_repair_enabled desde la
    # GUI (que reutiliza el proceso) daba tres veces el mismo numero.
    # Only 6 globals used to be re-read, so an A/B on fp32_repair_enabled from the GUI
    # (which reuses the process) returned the same number three times.
    global FP32_REPAIR_ENABLED, CAST_FROZEN_BF16, CAST_FROZEN_RESPECT_FP32
    global USE_AUTOCAST, FP32_NOISE_CONSTRUCTION, TRIM_TEXT_PADDING
    global DROP_AUDIO_ROWS_WHEN_UNUSED, ALLOC_PROBE
    global LR, MIN_LR_RATIO, WARMUP_STEPS, TOTAL_STEPS, GRAD_ACCUM_STEPS
    global LORA_RANK, LORA_ALPHA, MAX_GRAD_NORM, WEIGHT_DECAY, SAVE_EVERY
    global MAX_TEXT_TOKENS, USE_AUDIO_LOSS, OPTIMIZER_TYPE
    global VRAM_BUDGET_GB, VRAM_SWAP_GB, VRAM_HEADROOM_GB
    global VRAM_TRAINING_OVERHEAD_GB, VRAM_HARD_CAP_ENABLED, VRAM_EMPTY_CACHE_EVERY
    global CHECKPOINT_USE_REENTRANT
    global CAPTION_DROPOUT
    global LORA_SAVE_RAW_COPY
    global LR_SCHEDULE, GRAD_PROFILE_EVERY
    global OVERFIT_PROBE_EVERY
    global VRAM_RESIDENT_MAX_BYTES, VRAM_SWAP_MAX_BYTES, VRAM_HEADROOM_MAX_BYTES
    global VRAM_RUNTIME_MAX_BYTES
    global PROJECT_NAME, TRIGGER_WORD, NF4_CACHE_DIR, SEED
    global CACHE_DIR, OUTPUT_DIR, RESUME_DIR, OPT_FILE, STEP_FILE, LOSS_STATE_FILE
    global PREVIEW_EVERY, PREVIEW_STEPS, PREVIEW_CFG, PREVIEW_SHIFT, PREVIEW_SEED
    global PREVIEW_CAPTION_MODE, PREVIEW_NUM_FRAMES

    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                cfg = {str(k).strip(): v for k, v in raw.items()}
        except Exception as e:
            print("[CONFIG][WARN] Could not re-read {}: {} / No se pudo releer {}: {}".format(
                CONFIG_PATH, e, CONFIG_PATH, e), flush=True)

    _tc = str(cfg_get("timestep_convention", DEFAULTS["timestep_convention"])).strip().lower()
    TIMESTEP_CONVENTION = _tc if _tc in ("one_minus_sigma", "sigma") else "one_minus_sigma"
    FLOW_CONV_DEBUG_STEPS = int(
        cfg_get("flow_convention_debug_steps", DEFAULTS["flow_convention_debug_steps"]) or 0)
    FLOW_TARGET_SIGN = int(cfg_get("flow_target_sign", DEFAULTS["flow_target_sign"]) or -1)
    _ss = cfg_get("sigma_shift", DEFAULTS["sigma_shift"])
    SIGMA_SHIFT = float(_ss) if _ss is not None else None
    LOGIT_NORMAL_MU = float(cfg_get("logit_normal_mu", DEFAULTS["logit_normal_mu"]))
    LOGIT_NORMAL_STD = float(cfg_get("logit_normal_std", DEFAULTS["logit_normal_std"]))
    SIGMA_RESOLUTION_SHIFT = _cfg_bool(
        "sigma_resolution_shift", DEFAULTS["sigma_resolution_shift"])
    BF16_REDUCED_PRECISION = _cfg_bool(
        "bf16_reduced_precision_reduction", DEFAULTS["bf16_reduced_precision_reduction"])
    DATASET_SAMPLER = str(
        cfg_get("dataset_sampler", DEFAULTS["dataset_sampler"])).strip().lower()
    LORA_DTYPE_STR = str(cfg_get("lora_dtype", DEFAULTS["lora_dtype"])).strip().lower()
    LORA_DTYPE = torch.bfloat16 if LORA_DTYPE_STR in ("bf16", "bfloat16") else torch.float32
    LORA_EXCLUDE_REFINER = _cfg_bool(
        "lora_exclude_refiner", DEFAULTS["lora_exclude_refiner"])
    _mult = safe_float(cfg_get("timestep_scale_multiplier", 1.0), 1.0)

    FP32_REPAIR_ENABLED = _cfg_bool("fp32_repair_enabled", DEFAULTS["fp32_repair_enabled"])
    CAST_FROZEN_BF16 = _cfg_bool("cast_frozen_bf16", DEFAULTS["cast_frozen_bf16"])
    CAST_FROZEN_RESPECT_FP32 = _cfg_bool(
        "cast_frozen_respect_fp32_modules", DEFAULTS["cast_frozen_respect_fp32_modules"])
    USE_AUTOCAST = _cfg_bool("use_autocast", DEFAULTS["use_autocast"])
    FP32_NOISE_CONSTRUCTION = _cfg_bool(
        "fp32_noise_construction", DEFAULTS["fp32_noise_construction"])
    TRIM_TEXT_PADDING = _cfg_bool("trim_text_padding", DEFAULTS["trim_text_padding"])
    DROP_AUDIO_ROWS_WHEN_UNUSED = _cfg_bool(
        "drop_audio_rows_when_unused", DEFAULTS["drop_audio_rows_when_unused"])
    ALLOC_PROBE = _cfg_bool("alloc_probe", DEFAULTS["alloc_probe"])

    LR = float(cfg_get("lr", DEFAULTS["lr"]))
    MIN_LR_RATIO = float(cfg_get("min_lr_ratio", DEFAULTS["min_lr_ratio"]))
    LR_SCHEDULE = str(cfg_get("lr_schedule", DEFAULTS["lr_schedule"])).strip().lower()
    GRAD_PROFILE_EVERY = int(cfg_get("grad_profile_every", DEFAULTS["grad_profile_every"]) or 0)
    OVERFIT_PROBE_EVERY = int(cfg_get("overfit_probe_every", DEFAULTS["overfit_probe_every"]) or 0)
    WARMUP_STEPS = int(cfg_get("warmup_steps", DEFAULTS["warmup_steps"]))
    TOTAL_STEPS = int(cfg_get("total_steps", DEFAULTS["total_steps"]))
    GRAD_ACCUM_STEPS = max(1, int(cfg_get("grad_accum_steps", DEFAULTS["grad_accum_steps"])))
    LORA_RANK = int(cfg_get("lora_rank", DEFAULTS["lora_rank"]))
    LORA_ALPHA = int(cfg_get("lora_alpha", DEFAULTS["lora_alpha"]))
    MAX_GRAD_NORM = float(cfg_get("max_grad_norm", DEFAULTS["max_grad_norm"]))
    WEIGHT_DECAY = float(cfg_get("weight_decay", DEFAULTS["weight_decay"]))
    OPTIMIZER_TYPE = _normalize_optimizer_type(
        cfg_get("optimizer_type", DEFAULTS["optimizer_type"]))
    SAVE_EVERY = int(cfg_get("save_every", DEFAULTS["save_every"]))

    PREVIEW_CAPTION_MODE = str(
        cfg_get("preview_caption_mode", DEFAULTS["preview_caption_mode"])).strip().lower()
    if PREVIEW_CAPTION_MODE not in PREVIEW_CAPTION_MODES:
        PREVIEW_CAPTION_MODE = "first"
    PREVIEW_EVERY = int(cfg_get("preview_every", DEFAULTS["preview_every"]) or 0)
    PREVIEW_STEPS = max(2, int(cfg_get("preview_steps", DEFAULTS["preview_steps"]) or 12))
    PREVIEW_CFG = float(cfg_get("preview_cfg", DEFAULTS["preview_cfg"]) or 0.0)
    PREVIEW_SHIFT = float(cfg_get("preview_shift", DEFAULTS["preview_shift"]) or 12.0)
    PREVIEW_SEED = int(cfg_get("preview_seed", DEFAULTS["preview_seed"]) or -1)
    PREVIEW_VAE_DEVICE = str(
        cfg_get("preview_vae_device", DEFAULTS["preview_vae_device"])).strip().lower()
    if PREVIEW_VAE_DEVICE not in ("cpu", "cuda"):
        PREVIEW_VAE_DEVICE = "cpu"
    PREVIEW_FRAME_INDEX = int(
        cfg_get("preview_frame_index", DEFAULTS["preview_frame_index"]) or 0)
    PREVIEW_NUM_FRAMES = max(1, int(
        cfg_get("preview_num_frames", DEFAULTS["preview_num_frames"]) or 1))

    MAX_TEXT_TOKENS = int(cfg_get("max_text_tokens", DEFAULTS["max_text_tokens"]) or 0)
    CAPTION_DROPOUT = float(cfg_get("caption_dropout", DEFAULTS["caption_dropout"]) or 0.0)
    LORA_SAVE_RAW_COPY = _cfg_bool("lora_save_raw_copy", DEFAULTS["lora_save_raw_copy"])
    USE_AUDIO_LOSS = _cfg_bool("use_audio_loss", DEFAULTS["use_audio_loss"])

    VRAM_BUDGET_GB = float(cfg_get("vram_budget_gb", DEFAULTS["vram_budget_gb"]))
    VRAM_SWAP_GB = float(cfg_get("vram_swap_gb", DEFAULTS["vram_swap_gb"]))
    VRAM_HEADROOM_GB = float(cfg_get("vram_headroom_gb", DEFAULTS["vram_headroom_gb"]))
    VRAM_TRAINING_OVERHEAD_GB = float(
        cfg_get("vram_training_overhead_gb", DEFAULTS["vram_training_overhead_gb"]))
    VRAM_HARD_CAP_ENABLED = _cfg_bool(
        "vram_hard_cap_enabled", DEFAULTS["vram_hard_cap_enabled"])
    VRAM_EMPTY_CACHE_EVERY = int(
        cfg_get("vram_empty_cache_every", DEFAULTS["vram_empty_cache_every"]) or 0)
    CHECKPOINT_USE_REENTRANT = _cfg_bool(
        "checkpoint_use_reentrant", DEFAULTS["checkpoint_use_reentrant"])
    # Las derivadas hay que recalcularlas o el presupuesto del swap se queda congelado.
    # The derived values must be recomputed or the swap budget stays frozen.
    VRAM_RESIDENT_MAX_BYTES = max(0.0, VRAM_BUDGET_GB) * 1e9
    VRAM_SWAP_MAX_BYTES = max(0.0, VRAM_SWAP_GB) * 1e9
    VRAM_HEADROOM_MAX_BYTES = max(0.0, VRAM_HEADROOM_GB) * 1e9
    VRAM_RUNTIME_MAX_BYTES = (VRAM_RESIDENT_MAX_BYTES + VRAM_SWAP_MAX_BYTES
                              + VRAM_HEADROOM_MAX_BYTES)

    # Rutas: cambiar de proyecto en la GUI sin reiniciar dejaba CACHE_DIR/OUTPUT_DIR
    # congelados del primer import, es decir, entrenando el dataset nuevo dentro de la
    # carpeta de salida del proyecto anterior. La clave de runtime incluye ambos, asi
    # que al recalcularlos aqui el modelo y el dataset se recargan solos.
    # Paths: switching project in the GUI without restarting left CACHE_DIR/OUTPUT_DIR
    # frozen from the first import -- training the new dataset into the previous
    # project's output folder. The runtime key includes both, so recomputing them here
    # makes the model and dataset reload on their own.
    PROJECT_NAME = str(cfg_get("project_name", DEFAULTS["project_name"])).strip()
    TRIGGER_WORD = str(cfg_get("trigger_word", DEFAULTS["trigger_word"])).strip()
    NF4_CACHE_DIR = str(cfg_get("nf4_cache_dir", DEFAULTS["nf4_cache_dir"])).strip()
    SEED = int(cfg_get("seed", DEFAULTS["seed"]))

    if PROJECT_NAME:
        CACHE_DIR = "./cached_data_minimaxh3_{}".format(PROJECT_NAME)
        OUTPUT_DIR = "./minimaxh3_lora_output_{}".format(PROJECT_NAME)
    else:
        CACHE_DIR = str(cfg_get("cache_dir", DEFAULTS["cache_dir"])).strip()
        OUTPUT_DIR = str(cfg_get("output_dir", DEFAULTS["output_dir"])).strip()

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except Exception as e:
        print("[CONFIG][WARN] Could not create OUTPUT_DIR {}: {} / No se pudo crear "
              "OUTPUT_DIR {}: {}".format(OUTPUT_DIR, e, OUTPUT_DIR, e), flush=True)

    RESUME_DIR = os.path.join(OUTPUT_DIR, "resume_checkpoint")
    OPT_FILE = os.path.join(OUTPUT_DIR, "optimizer.pt")
    STEP_FILE = os.path.join(OUTPUT_DIR, "current_step.txt")
    LOSS_STATE_FILE = os.path.join(OUTPUT_DIR, "loss_state.json")

    # Se instala AQUI, con OUTPUT_DIR ya resuelto y creado, para que el volcado
    # [CONFIG] de mas abajo tambien quede en el fichero: es justo lo que hay que
    # poder releer cuando una corrida sale rara.
    # Installed HERE, with OUTPUT_DIR resolved and created, so the [CONFIG] dump
    # below also lands in the file.
    install_train_log(OUTPUT_DIR)

    print("=" * 78, flush=True)
    # ------------------------------------------------------------------
    # QUIEN MANDA: train_settings.json GANA a los DEFAULTS del script. Editar los
    # defaults y dejar el JSON con los valores viejos es la forma mas silenciosa de
    # "aplicar" un cambio y entrenar con lo de antes. Aqui se listan explicitamente
    # las claves de entrenamiento que el JSON esta pisando.
    # train_settings.json OVERRIDES the script DEFAULTS. This lists the training keys
    # the JSON is currently overriding, so a stale JSON can't silently win.
    # ------------------------------------------------------------------
    _critical = ("total_steps", "grad_accum_steps", "warmup_steps", "lr", "lora_rank",
                 "lora_alpha", "lora_dtype", "sigma_shift", "logit_normal_mu",
                 "optimizer_type", "dataset_sampler", "lora_exclude_refiner",
                 "vram_budget_gb", "min_lr_ratio", "max_grad_norm")
    _over = [(k, DEFAULTS.get(k), cfg[k]) for k in _critical
             if isinstance(cfg, dict) and k in cfg and cfg[k] != DEFAULTS.get(k)]
    if _over:
        print("[CONFIG][WARN] {} pisa estos DEFAULTS del script / overrides these script "
              "defaults:".format(CONFIG_PATH), flush=True)
        for k, d, v in _over:
            print("[CONFIG][WARN]   {:<22} script={!r:<10} -> JSON={!r}".format(k, d, v),
                  flush=True)
        print("[CONFIG][WARN] Si querias los valores nuevos, borra esas claves del JSON. "
              "/ Delete those keys from the JSON to use the new defaults.", flush=True)
    print("[CONFIG] EFFECTIVE flow settings / ajustes de flow EFECTIVOS:", flush=True)
    print("[CONFIG]   timestep_convention      = {}".format(TIMESTEP_CONVENTION), flush=True)
    print("[CONFIG]   timestep_scale_multiplier= {:g}".format(_mult), flush=True)
    print("[CONFIG]   flow_target_sign         = {} ({})".format(
        FLOW_TARGET_SIGN, "x0-noise" if FLOW_TARGET_SIGN < 0 else "noise-x0"), flush=True)
    print("[CONFIG]   sigma_shift              = {}".format(
        SIGMA_SHIFT if SIGMA_SHIFT is not None
        else "logit-normal (mu={:g}, std={:g}, res_shift={})".format(
            LOGIT_NORMAL_MU, LOGIT_NORMAL_STD, SIGMA_RESOLUTION_SHIFT)), flush=True)
    print("[CONFIG]   lora_dtype               = {} (dtype del estado de AdamW)".format(
        LORA_DTYPE_STR), flush=True)
    print("[CONFIG]   lora_exclude_refiner     = {}".format(LORA_EXCLUDE_REFINER), flush=True)
    print("[CONFIG]   dataset_sampler          = {}".format(DATASET_SAMPLER), flush=True)
    print("[CONFIG]   updates reales           = {} (total_steps {} / grad_accum {})".format(
        int(TOTAL_STEPS / max(1, GRAD_ACCUM_STEPS)), TOTAL_STEPS, GRAD_ACCUM_STEPS),
        flush=True)
    print("[CONFIG]   -> t = {} * {:g}".format(
        "sigma" if TIMESTEP_CONVENTION == "sigma" else "(1 - sigma)", _mult), flush=True)
    print("[CONFIG]   fp32_repair_enabled      = {}".format(FP32_REPAIR_ENABLED), flush=True)
    print("[CONFIG]   cast_frozen_bf16         = {} (respect _keep_in_fp32_modules = {})".format(
        CAST_FROZEN_BF16, CAST_FROZEN_RESPECT_FP32), flush=True)
    print("[CONFIG]   use_autocast             = {}".format(USE_AUTOCAST), flush=True)
    print("[CONFIG]   fp32_noise_construction  = {}".format(FP32_NOISE_CONSTRUCTION), flush=True)
    print("[CONFIG]   trim_text_padding        = {}".format(TRIM_TEXT_PADDING), flush=True)
    print("[CONFIG]   drop_audio_rows_unused   = {}".format(DROP_AUDIO_ROWS_WHEN_UNUSED),
          flush=True)
    print("[CONFIG]   lr / rank / alpha        = {:g} / {} / {}".format(
        LR, LORA_RANK, LORA_ALPHA), flush=True)
    print("[CONFIG]   optimizer_type           = {}{}".format(
        OPTIMIZER_TYPE,
        "" if OPTIMIZER_TYPE == "adamw"
        else "  <-- en H3 deja la cara blanda; usa 'adamw'"), flush=True)
    # Con que ajustes se genero cada preview tiene que quedar por escrito: si no,
    # al mirar una imagen semanas despues no hay forma de saber con que pasos o
    # que shift salio. Va con `print`, no con log_print, para que se vea tambien
    # con debug_training=False.
    # The preview settings must be on the record: otherwise, looking at an image
    # weeks later there is no way to know which steps or shift produced it.
    if PREVIEW_EVERY > 0:
        print("[CONFIG]   preview cada/steps/cfg   = {} / {} / {:g}".format(
            PREVIEW_EVERY, PREVIEW_STEPS, PREVIEW_CFG), flush=True)
        print("[CONFIG]   preview shift/seed/vae   = {:g} / {} / {}{}".format(
            PREVIEW_SHIFT, PREVIEW_SEED if PREVIEW_SEED > 0 else "(usa seed)",
            PREVIEW_VAE_DEVICE,
            "" if PREVIEW_SHIFT <= 6.0 else
            "  <-- shift 12 es el del muestreador de VIDEO: en una imagen suelta "
            "ningun paso baja de sigma 0.30 y satura el color. Prueba 3.0"),
            flush=True)
    else:
        print("[CONFIG]   previews                 = OFF (preview_every = 0)", flush=True)
    print("[CONFIG]   vram budget/swap/headroom= {:.2f} / {:.2f} / {:.2f} GB".format(
        VRAM_BUDGET_GB, VRAM_SWAP_GB, VRAM_HEADROOM_GB), flush=True)
    print("[CONFIG]   vram overhead entrenam.  = {:.2f} GB | TECHO TOTAL {:.2f} GB "
          "(hard cap {})".format(
              VRAM_TRAINING_OVERHEAD_GB,
              VRAM_BUDGET_GB + VRAM_SWAP_GB + VRAM_HEADROOM_GB + VRAM_TRAINING_OVERHEAD_GB,
              "ON" if VRAM_HARD_CAP_ENABLED else "OFF"), flush=True)
    print("[CONFIG]   checkpoint reentrant     = {}{}".format(
        CHECKPOINT_USE_REENTRANT,
        "" if CHECKPOINT_USE_REENTRANT
        else "  <-- con False bitsandbytes clava los pesos NF4 en el grafo"), flush=True)
    print("[CONFIG]   empty_cache cada         = {} paso(s)".format(
        VRAM_EMPTY_CACHE_EVERY if VRAM_EMPTY_CACHE_EVERY > 0 else "nunca"), flush=True)
    print("[CONFIG]   cache_dir                = {}".format(CACHE_DIR), flush=True)
    print("[CONFIG]   output_dir               = {}".format(OUTPUT_DIR), flush=True)
    print("=" * 78, flush=True)


def train_minimaxh3():
    global ACTIVATION_OFFLOAD_ACTIVE
    # Se reasigna a 0 si una preview falla, para no reintentarla en cada ciclo.
    # Reassigned to 0 when a preview fails, so it is not retried every cycle.
    global PREVIEW_EVERY
    global CURRENT_CONFIG
    global CURRENT_TRANSFORMER
    global PATCH_T
    global PATCH_H
    global PATCH_W
    global STOP_REQUESTED
    global TRAIN_RUNTIME

    # Releer la config ANTES de nada: si no, editar el JSON no surte efecto mientras la
    # GUI mantenga vivo el proceso de Python.
    reload_runtime_config()

    # Debe instalarse ANTES de cualquier forward: parchea la clase, no la instancia.
    # Must be installed BEFORE any forward: it patches the class, not the instance.
    install_adaln_dtype_fix()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False

    try:
        # False = los matmul bf16 acumulan en fp32. Con LoRA de rango bajo, el
        # gradiente util es la parte pequena de la suma; acumular en bf16 se la come.
        # False = fp32 accumulation for bf16 matmuls: keeps the small-magnitude part
        # of the gradient that a low-rank LoRA actually learns from.
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = bool(
            BF16_REDUCED_PRECISION)
    except Exception:
        pass

    configure_cpu_backend()

    if LOW_VRAM_12GB:
        try:
            torch.cuda.memory._set_allocator_settings("garbage_collection_threshold:0.6")
        except Exception:
            pass

        try:
            if platform.system() != "Windows":
                torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        except Exception:
            pass

    if ACTIVATION_OFFLOAD and not _SAVE_ON_CPU_AVAILABLE:
        ACTIVATION_OFFLOAD_ACTIVE = False
        log_print("[VRAM] activation_offload=True pero save_on_cpu no disponible.", flush=True)
    else:
        ACTIVATION_OFFLOAD_ACTIVE = bool(ACTIVATION_OFFLOAD and _SAVE_ON_CPU_AVAILABLE)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")

    runtime = _ensure_train_runtime()

    if runtime.get("completed", False):
        log_print("[TRAIN] Entrenamiento ya completado. Usa reset_train_runtime() para reiniciar.", flush=True)
        return

    model = runtime["model"]
    transformer = runtime["transformer"]
    entries = runtime["entries"]
    optimizer = runtime["optimizer"]
    trainable = runtime["trainable"]
    config = runtime["config"]
    patch_t, patch_h, patch_w = runtime["patch"]
    audio_channels = int(runtime.get("audio_channels", 32) or 32)

    CURRENT_CONFIG = config
    CURRENT_TRANSFORMER = transformer
    PATCH_T, PATCH_H, PATCH_W = patch_t, patch_h, patch_w

    start_step = runtime.get("last_step", 0)
    resumed_ema_loss = runtime.get("ema_loss", None)
    resumed_running_loss = runtime.get("running_loss", 0.0)
    resumed_elapsed = float(runtime.get("elapsed_seconds", 0.0) or 0.0)
    # Cronometro de ESTA sesion. El total que se reporta al final es
    # resumed_elapsed + lo de esta sesion, para que una corrida reanudada no
    # cuente solo el ultimo tramo.
    # This session's stopwatch; the reported total adds the time already
    # spent in previous runs so a resumed run does not under-report.
    session_start_wall = time.time()

    def total_elapsed_seconds():
        return resumed_elapsed + (time.time() - session_start_wall)

    def save_checkpoint_now(current_s):
        if current_s <= 0:
            return

        log_print()
        log_print("Saving checkpoint at step {}...".format(current_s), flush=True)

        os.makedirs(RESUME_DIR, exist_ok=True)

        # --------------------------------------------------------------
        # Orden e INVARIANTE del checkpoint.
        #
        # STEP_FILE es lo que decide desde dónde se reanuda, así que se escribe
        # EL ÚLTIMO y sólo si los pesos y el optimizador se guardaron bien. Antes,
        # los fallos se tragaban con `except: pass` y STEP_FILE avanzaba igual:
        # al reanudar cargabas pesos VIEJOS declarando el paso nuevo, con momentos
        # de Adam que ya no correspondían a esos pesos.
        # --------------------------------------------------------------
        weights_ok = False
        try:
            model.save_pretrained(RESUME_DIR)
            weights_ok = True
        except Exception as e:
            log_print("[CKPT][ERROR] No se pudieron guardar los pesos LoRA: {}".format(e),
                      flush=True)

        opt_ok = False
        try:
            # temporal + replace atómico: un kill a mitad no deja un OPT_FILE corrupto
            _opt_tmp = OPT_FILE + ".tmp"
            torch.save(optimizer.state_dict(), _opt_tmp)
            os.replace(_opt_tmp, OPT_FILE)
            opt_ok = True
        except Exception as e:
            log_print("[CKPT][ERROR] No se pudo guardar el optimizador: {}".format(e), flush=True)
            try:
                if os.path.exists(OPT_FILE + ".tmp"):
                    os.remove(OPT_FILE + ".tmp")
            except Exception:
                pass

        if weights_ok and opt_ok:
            try:
                _step_tmp = STEP_FILE + ".tmp"
                with open(_step_tmp, "w", encoding="utf-8") as f:
                    f.write(str(current_s))
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(_step_tmp, STEP_FILE)
            except Exception as e:
                log_print("[CKPT][ERROR] No se pudo escribir STEP_FILE: {}".format(e), flush=True)
        else:
            log_print(
                "[CKPT][ERROR] Checkpoint INCOMPLETO (pesos={}, optimizador={}). "
                "NO se actualiza STEP_FILE: al reanudar se volverá al último "
                "checkpoint íntegro en vez de continuar con estado inconsistente."
                .format(weights_ok, opt_ok), flush=True)

        try:
            with open(LOSS_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ema_loss": float(ema_loss) if ema_loss is not None else None,
                        "running_loss": float(running_loss),
                        "step": int(current_s),
                        "elapsed_seconds": float(total_elapsed_seconds()),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            log_print("[!] No se pudo guardar estado del loss: {}".format(e), flush=True)

        try:
            ckpt = os.path.join(
                OUTPUT_DIR,
                "MiniMaxH3_LoRA_step_{}.safetensors".format(current_s),
            )
            save_lora(model, ckpt)
            log_print("Checkpoint saved: {}".format(ckpt), flush=True)
        except Exception:
            pass

        if TRAIN_RUNTIME is not None:
            TRAIN_RUNTIME["last_step"] = int(current_s)
            TRAIN_RUNTIME["ema_loss"] = float(ema_loss) if ema_loss is not None else None
            TRAIN_RUNTIME["running_loss"] = float(running_loss)
            TRAIN_RUNTIME["elapsed_seconds"] = float(total_elapsed_seconds())

    def handle_signal(sig, frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
        log_print()
        log_print("Signal received ({}). Convirtiendo en parada suave.".format(sig), flush=True)
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, handle_signal)
    except Exception:
        pass

    def lr_at(step):
        # El schedule va en ACTUALIZACIONES del optimizador, no en micro-pasos: con
        # grad_accum=4 el warmup de "100 pasos" eran en realidad 25 updates y el coseno
        # se calculaba sobre un eje 4x mas largo que el numero real de updates.
        # The schedule now runs on optimizer UPDATES, not micro-steps.
        opt_step = step / float(max(1, GRAD_ACCUM_STEPS))
        opt_total = TOTAL_STEPS / float(max(1, GRAD_ACCUM_STEPS))

        if opt_step < WARMUP_STEPS:
            return LR * opt_step / max(1, WARMUP_STEPS)

        if LR_SCHEDULE == "flat":
            # Sin decaimiento: el LoRA arranca en cero y necesita todo el
            # presupuesto de movimiento que pueda.
            return LR

        progress = (opt_step - WARMUP_STEPS) / max(1.0, opt_total - WARMUP_STEPS)
        progress = min(1.0, max(0.0, progress))

        return LR * (
            MIN_LR_RATIO
            + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * progress))
        )

    if SEED > 0:
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    running_loss = resumed_running_loss

    # Dos medidas del tiempo por paso, porque sirven para cosas distintas.
    #
    #   avg_time  EMA con alpha 0,1, o sea una ventana efectiva de ~10 pasos.
    #             Reacciona rapido, que es lo que quiere el ETA, pero con un
    #             dataset de buckets mezclados NO converge: se pasea segun que
    #             clips hayan caido en los ultimos 10 pasos. Medido en un cache
    #             de 9 clips, el mas barato y el mas caro se llevan un 11% de
    #             tokens, asi que esta cifra baila varias decimas eternamente y
    #             no vale para comparar dos configuraciones.
    #
    #   mean_time media de TODOS los pasos de esta sesion. Promedia los buckets,
    #             asi que se asienta de verdad. Es la que hay que mirar para
    #             decidir si 18 bloques residentes van mejor que 20.
    #
    # No se restaura al reanudar: mezclar los tiempos de una sesion anterior,
    # quiza con otra configuracion, daria una media que no describe nada.
    #
    # Two measures of step time, for different jobs. `avg_time` is an EMA with
    # alpha 0.1 (~10 step window): reactive, which is what the ETA wants, but on
    # a mixed-bucket dataset it never converges -- it drifts with whichever clips
    # landed in the last ten steps (measured 11% token spread across a 9 clip
    # cache), so it cannot be used to compare two configurations. `mean_time`
    # averages every step of this session, so it averages the buckets out and
    # actually settles: that is the number to read when deciding whether 18
    # resident blocks beat 20. It is not restored on resume, since mixing in a
    # previous session's timings -- possibly from another configuration -- would
    # produce a mean that describes nothing.
    avg_time = 0.0
    step_time_sum = 0.0
    step_time_count = 0
    ema_loss = resumed_ema_loss

    # ------------------------------------------------------------------
    # ULTIMO PASO REALMENTE COMPLETADO.
    #
    # El boton de STOP de la GUI manda CTRL_BREAK_EVENT y `handle_signal` lanza
    # KeyboardInterrupt AL INSTANTE, o sea en mitad del forward, del backward o
    # del propio optimizer.step(). El manejador guardaba `step`, el paso que
    # estaba a medias: si la interrupcion caia antes de optimizer.step(), el
    # checkpoint declaraba completado un paso cuyo update nunca se aplico y al
    # reanudar ese paso se saltaba.
    #
    # Esta variable solo avanza DESPUES de que el update este aplicado y los
    # gradientes limpiados, que con grad_accum>1 es ademas el unico punto donde
    # no hay gradiente a medio acumular en el buffer. Es el unico punto de
    # reanudacion consistente que existe.
    #
    # LAST FULLY COMPLETED STEP. The GUI's stop button raises KeyboardInterrupt
    # mid-step, and the handler used to checkpoint `step` — the in-flight one.
    # This only advances after the update is applied and the grads are cleared,
    # which with grad_accum>1 is also the only point with no half-accumulated
    # gradient pending. It is the only consistent resume point.
    # ------------------------------------------------------------------
    last_completed_step = start_step

    log_print("STARTING TRAINING! {} entradas cacheadas.".format(len(entries)), flush=True)

    # ------------------------------------------------------------------
    # RESUMEN DEL DATASET / DATASET SUMMARY
    #
    # Informativo, no alarmista. Estas cifras son utiles para saber donde
    # invertir el esfuerzo la proxima vez, pero NO predicen el resultado: un
    # dataset de 8 imagenes a 768x768 produce aqui LoRAs con parecido excelente.
    #
    # Habia un tercer aviso que decia que por debajo de 600 tokens de video no
    # habia informacion suficiente para codificar una identidad. Era FALSO y se
    # ha eliminado: 570 tokens (768x768) dan parecido perfecto, y otros
    # entrenadores de esta familia trabajan a 512x512 sin problema. Un aviso que
    # contradice el resultado medido no es un aviso, es ruido.
    #
    # Informational, not alarmist. These figures say where to invest effort next
    # time; they do NOT predict the outcome. A third warning claiming that under
    # 600 video tokens there is not enough information to encode an identity was
    # FALSE and has been removed: 570 tokens (768x768) gives excellent likeness
    # here, and other trainers in this family work fine at 512x512.
    # ------------------------------------------------------------------
    _n_entries = len(entries)
    _epochs = TOTAL_STEPS / float(max(1, _n_entries))
    log_print("[DATASET] {} images | {} steps | {:.1f} epochs (each image seen ~{:.0f} "
              "times) / {} imagenes | {} pasos | {:.1f} epocas (cada imagen se ve ~{:.0f} "
              "veces)".format(_n_entries, TOTAL_STEPS, _epochs, _epochs,
                              _n_entries, TOTAL_STEPS, _epochs, _epochs), flush=True)

    if _n_entries < 20:
        log_print(
            "[DATASET] Small set: {} images. This can work very well for a character: a "
            "tight, consistent set is often enough. A larger one (a practical reference "
            "is 35-45, mixing medium shots with face close-ups) mostly buys better "
            "generalisation to new poses and prompts. / Conjunto pequeno: {} imagenes. "
            "Puede funcionar muy bien para un personaje: un conjunto corto y consistente "
            "suele bastar. Uno mayor (referencia practica: 35-45, mezclando planos medios "
            "con primeros planos de cara) sobre todo compra mejor generalizacion a poses "
            "y prompts nuevos.".format(_n_entries, _n_entries), flush=True)

    if _epochs > 60:
        log_print(
            "[DATASET] {:.0f} epochs over {} images. Past roughly this point the LoRA "
            "leans more on memorising framing and background than on generalising, so "
            "extra images tend to pay off more than extra steps. / {:.0f} epocas sobre {} "
            "imagenes. Pasado mas o menos este punto el LoRA se apoya mas en memorizar "
            "encuadre y fondo que en generalizar, asi que suele rendir mas anadir "
            "imagenes que anadir pasos.".format(
                _epochs, _n_entries, _epochs, _n_entries), flush=True)

    try:
        _probe = entries[0]
        _lat = _probe.get("video") if isinstance(_probe, dict) else None
        if _lat is None:
            _lat = _probe.get("latent") if isinstance(_probe, dict) else None
        if torch.is_tensor(_lat) and _lat.ndim == 5:
            _, _c, _f, _h, _w = _lat.shape
            _gh = _h // max(1, PATCH_H)
            _gw = _w // max(1, PATCH_W)
            log_print(
                "[DATASET] Latent {}x{} -> {}x{} grid = {} video tokens per image / "
                "Latente {}x{} -> rejilla {}x{} = {} tokens de video por imagen".format(
                    _h, _w, _gh, _gw, _gh * _gw,
                    _h, _w, _gh, _gw, _gh * _gw), flush=True)
    except Exception:
        pass

    def _steps_from(start):
        """Genera los pasos leyendo TOTAL_STEPS en CADA iteracion.

        Con range(start, TOTAL_STEPS+1) el limite queda congelado al entrar en el
        bucle, asi que subir los pasos en caliente no tenia ningun efecto. Un
        generador lo vuelve a leer cada vez, de modo que ampliar la corrida desde
        la GUI funciona igual que acortarla.

        Yields steps re-reading TOTAL_STEPS on EVERY iteration. With a range() the
        limit is frozen when the loop starts, so raising the step count live did
        nothing; a generator re-reads it, making "extend the run" work like
        "shorten the run" already did.
        """
        s = start
        while True:
            s += 1
            if s > TOTAL_STEPS:
                return
            yield s

    try:
        for step in _steps_from(start_step):
            if STOP_REQUESTED:
                log_print("[STOP] Stopping on request / Deteniendo por solicitud de "
                          "parada.", flush=True)
                # last_completed_step y no step-1: con grad_accum>1 el paso
                # anterior puede haber sido solo de acumulacion, sin update, y
                # declararlo completo perderia el gradiente a medio acumular.
                # last_completed_step, not step-1: with grad_accum>1 the previous
                # micro-step may have only accumulated, and calling it complete
                # would drop the half-accumulated gradient.
                save_checkpoint_now(last_completed_step)
                log_print("Total training time / Tiempo total de entrenamiento: {}"
                          .format(format_duration_bilingual(total_elapsed_seconds())),
                          flush=True)
                STOP_REQUESTED = False
                close_train_log()
                return

            # ----------------------------------------------------------------
            # AJUSTES EN CALIENTE. Coste: un getmtime por paso. Si el fichero no
            # ha cambiado, no se abre siquiera. Ver hot_reload_live_settings.
            # LIVE SETTINGS: one getmtime per step; the file is not even opened
            # unless it changed.
            # ----------------------------------------------------------------
            try:
                _live_changes = hot_reload_live_settings()
                if _live_changes:
                    log_print("")
                    log_print("[LIVE] Settings reloaded without stopping training / "
                              "Ajustes recargados sin parar el entrenamiento:", flush=True)
                    for _c in _live_changes:
                        log_print("[LIVE]   {}".format(_c), flush=True)
            except Exception as _e_live:
                log_print("[LIVE][WARN] Could not reload settings: {} / No se pudieron "
                          "recargar los ajustes: {}".format(_e_live, _e_live), flush=True)

            # total_steps bajado en caliente: se termina de forma ordenada por la
            # via normal (guarda checkpoint y LoRA final), no con un corte seco.
            # total_steps lowered live: finish through the normal completion path.
            if step > TOTAL_STEPS:
                log_print("")
                log_print("[LIVE] total_steps was lowered to {}: finishing the run. / "
                          "total_steps se bajo a {}: terminando la corrida."
                          .format(TOTAL_STEPS, TOTAL_STEPS), flush=True)
                break

            t0 = time.time()

            # Los picos son POR PASO: sin reset, max_memory_allocated() se queda
            # con el maximo historico (normalmente el de la carga NF4) y no dice
            # nada sobre lo que consume realmente un paso de entrenamiento.
            # Per-step peaks: without the reset the counter keeps the historical max
            # (usually from NF4 loading) and says nothing about a training step.
            if torch.cuda.is_available():
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass

            # ----------------------------------------------------------------
            # RNG determinista POR PASO (seguro ante pause/resume).
            #
            # Antes, las semillas se fijaban UNA vez al arrancar. Al reanudar se
            # volvían a fijar al mismo valor, así que el paso 501 repetía exactamente
            # la imagen, el ruido y el sigma del paso 1: cada pausa reiniciaba el
            # stream aleatorio desde el principio y sesgaba el entrenamiento hacia
            # el mismo prefijo de muestras.
            #
            # Derivando la semilla del índice de paso, el paso N usa siempre los
            # mismos datos tanto si viene de una corrida continua como de diez
            # reanudaciones: reanudar pasa a ser EXACTAMENTE equivalente a no parar.
            # ----------------------------------------------------------------
            if SEED > 0:
                # MEZCLA DE LA SEMILLA (splitmix64). Antes la semilla era
                # SEED*1000003 + step, es decir semillas CONSECUTIVAS, y la primera
                # extraccion de cada paso es justo el sigma. En tu log los sigmas de
                # los pasos 3-6 salieron 0.146 / 0.168 / 0.256 / 0.304: monotonos
                # crecientes, que es la firma de semillas correlacionadas. Un sigma
                # que barre en rampa en vez de saltar aleatoriamente sesga el
                # entrenamiento por tramos. splitmix64 rompe esa estructura.
                # Consecutive seeds made the first draw of each step (= sigma) walk
                # monotonically. splitmix64 decorrelates it.
                _z = (int(SEED) * 0x9E3779B97F4A7C15 + int(step) * 0xBF58476D1CE4E5B9)
                _z &= (1 << 64) - 1
                _z ^= (_z >> 30); _z = (_z * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
                _z ^= (_z >> 27); _z = (_z * 0x94D049BB133111EB) & ((1 << 64) - 1)
                _z ^= (_z >> 31)
                _step_seed = _z % (2 ** 31 - 1)
                random.seed(_step_seed)
                np.random.seed(_step_seed)
                torch.manual_seed(_step_seed)

            # ----------------------------------------------------------------
            # MUESTREO DEL DATASET.
            #
            # random.choice() es muestreo CON REEMPLAZO: con 20 imagenes y 2000 pasos
            # la mas vista sale ~30% mas que la menos vista, y en datasets de 10-15
            # fotos eso significa que el modelo ve la cara desde unos angulos el doble
            # que desde otros. Una permutacion barajada por epoca garantiza cobertura
            # exacta y sigue siendo determinista respecto a SEED (reanudar no cambia
            # el orden, porque la epoca se deriva del indice de paso).
            # A shuffled per-epoch permutation guarantees uniform coverage and stays
            # deterministic w.r.t. SEED across pause/resume.
            # ----------------------------------------------------------------
            if DATASET_SAMPLER == "shuffle_epoch" and len(entries) > 1:
                _n = len(entries)
                _epoch = (step - 1) // _n
                _pos = (step - 1) % _n
                _perm = list(range(_n))
                random.Random(int(SEED) * 7919 + int(_epoch)).shuffle(_perm)
                entry = entries[_perm[_pos]]
            else:
                entry = random.choice(entries)

            prompt_embeds, attention_mask = get_prompt_pair(entry["prompt"])

            # ----------------------------------------------------------------
            # CAPTION DROPOUT: se anula el conditioning de texto en una fraccion de
            # los pasos. La identidad tiene que quedar en los pesos, no en la
            # correlacion con la frase exacta del caption.
            # Aviso honesto: lo correcto seria un embedding cacheado del prompt VACIO;
            # aqui se ponen a cero los embeds y se deja 1 token activo, que es lo que
            # hacen la mayoria de trainers cuando no hay embed vacio en la cache.
            # Ideally this would use a cached EMPTY-prompt embedding; zeroing is the
            # common fallback when the cache has none.
            # ----------------------------------------------------------------
            _caption_dropped = False
            if CAPTION_DROPOUT > 0.0 and random.random() < CAPTION_DROPOUT:
                _caption_dropped = True
                prompt_embeds = torch.zeros_like(prompt_embeds)
                if attention_mask is not None:
                    attention_mask = torch.zeros_like(attention_mask)
                    try:
                        attention_mask[..., 0] = 1
                    except Exception:
                        pass

            with torch.no_grad():
                video_clean = entry["video"].to("cuda", dtype=torch.bfloat16, non_blocking=True)
                audio_clean = entry["audio"].to("cuda", dtype=torch.bfloat16, non_blocking=True)
                video_text = prompt_embeds.to("cuda", dtype=torch.bfloat16, non_blocking=True)

                if video_clean.ndim == 4:
                    video_clean = video_clean.unsqueeze(0)

                if audio_clean.ndim == 2:
                    audio_clean = audio_clean.unsqueeze(0)

                if video_text.ndim == 2:
                    video_text = video_text.unsqueeze(0)

                # ------------------------------------------------------------
                # RECORTE DEL PADDING DEL PROMPT.
                #
                # El forward de MiniMax-H3 NO acepta `attention_mask`: el unico canal
                # para expresar padding es `token_tags = -1`, y este script etiqueta
                # todas las filas de texto como 1. Ademas el eje t del RoPE del video
                # arranca en `origin = text_len`, asi que dejar el padding dentro del
                # texto desplaza la geometria posicional del video segun la longitud de
                # cada prompt. Recortar aqui es la solucion limpia: sin filas de padding,
                # `origin` es la longitud real y el modelo se queda con la ruta rapida
                # de atencion sin mascara.
                #
                # The H3 forward takes NO `attention_mask`: the only padding channel is
                # `token_tags = -1`, and this script tags every text row as 1. On top of
                # that the video RoPE t axis starts at `origin = text_len`, so leaving
                # padding inside the text shifts the video's positional geometry per
                # prompt length. Trimming here is the clean fix.
                # ------------------------------------------------------------
                if TRIM_TEXT_PADDING and attention_mask is not None and video_text.ndim == 3:
                    try:
                        _am = attention_mask.detach().reshape(-1)
                        _T_full = int(video_text.shape[1])
                        if _am.numel() >= _T_full:
                            _am = _am[:_T_full]
                            _am_b = (_am > 0)
                            _n_real = int(_am_b.sum().item())
                            # Solo recortamos si la mascara es un PREFIJO limpio
                            # (unos y luego ceros). Cualquier otra forma se deja intacta.
                            # Only trim when the mask is a clean PREFIX (ones then zeros).
                            _is_prefix = (
                                0 < _n_real <= _T_full
                                and bool(_am_b[:_n_real].all())
                                and (_n_real == _T_full or not bool(_am_b[_n_real:].any()))
                            )
                            if _is_prefix and _n_real < _T_full:
                                video_text = video_text[:, :_n_real, :].contiguous()
                                if step <= start_step + 1:
                                    log_print(
                                        "[TEXT] Padding trimmed: {} -> {} tokens (the H3 "
                                        "forward has no attention_mask; padding would "
                                        "shift the video RoPE origin). / Padding "
                                        "recortado: {} -> {} tokens (el forward de H3 no "
                                        "tiene attention_mask; el padding desplazaria el "
                                        "origen del RoPE del video).".format(
                                            _T_full, _n_real, _T_full, _n_real),
                                        flush=True)
                            elif step <= start_step + 1:
                                log_print(
                                    "[TEXT] attention_mask is not a clean prefix "
                                    "({}/{} live tokens); nothing trimmed. / la "
                                    "attention_mask no es un prefijo limpio ({}/{} tokens "
                                    "activos); no se recorta nada.".format(
                                        _n_real, _T_full, _n_real, _T_full), flush=True)
                    except Exception as _e_trim:
                        log_print("[TEXT][WARN] Could not trim padding: {} / No se pudo "
                                  "recortar el padding: {}".format(_e_trim, _e_trim),
                                  flush=True)
                elif step <= start_step + 1 and TRIM_TEXT_PADDING:
                    log_print("[TEXT] No attention_mask in the cache: assuming the prompt "
                              "embeddings carry no padding. / Sin attention_mask en la "
                              "cache: se asume que los embeddings no llevan padding.",
                              flush=True)

                if MAX_TEXT_TOKENS > 0:
                    if video_text.ndim == 3 and video_text.shape[1] > MAX_TEXT_TOKENS:
                        # Esto ya no es padding: se esta tirando conditioning REAL.
                        # Si salta, sube max_text_tokens o acorta los captions.
                        # This is no longer padding: real conditioning is being cut.
                        log_print(
                            "[TEXT][WARN] caption de '{}' recortado {} -> {} tokens REALES; "
                            "sube max_text_tokens o acorta el caption / real conditioning "
                            "truncated.".format(
                                entry.get("name", "?"), video_text.shape[1], MAX_TEXT_TOKENS),
                            flush=True)
                        video_text = video_text[:, :MAX_TEXT_TOKENS, :].contiguous()

                video_clean = align_video_latent_to_patch(
                    video_clean,
                    PATCH_H,
                    PATCH_W,
                    PATCH_T,
                )

                video_tokens = patch_video_latent(
                    video_clean,
                    PATCH_H,
                    PATCH_W,
                    PATCH_T,
                )

                audio_tokens = patch_audio_latent(audio_clean)

                if not USE_AUDIO_LOSS:
                    # Sin loss de audio la fila que se enviaba (token_tag=2,
                    # position_ids=(0,0,0), que colisiona con el primer token de texto)
                    # es una perturbacion que atraviesa los 50 bloques sin aportar nada.
                    # With no audio loss, the row we used to send (token_tag=2, position
                    # (0,0,0), colliding with the first text token) is a perturbation that
                    # crosses all 50 blocks for nothing.
                    if DROP_AUDIO_ROWS_WHEN_UNUSED:
                        audio_tokens = audio_tokens[:, :0, :].contiguous()
                    elif audio_tokens.shape[1] > 1:
                        audio_tokens = audio_tokens[:, :1, :].contiguous()

                B = video_tokens.shape[0]
                video_seq_len = video_tokens.shape[1]
                audio_seq_len = audio_tokens.shape[1]

                num_frames = video_clean.shape[2]
                height = video_clean.shape[3]
                width = video_clean.shape[4]
                audio_num_frames = audio_clean.shape[-1]
                video_latent_shape = tuple(video_clean.shape)

                sigma = sample_sigmas(
                    B,
                    device="cuda",
                    shift=SIGMA_SHIFT,
                    image_tokens=video_seq_len,
                ).to(torch.float32).clamp(1e-4, 1.0 - 1e-4)

                # ------------------------------------------------------------
                # CONSTRUCCION DE noisy / target.
                #
                # En bf16 el dato se construia con bf16(sigma) mientras al modelo se le
                # declaraba bf16(1 - sigma_fp32): dos redondeos distintos, ~0,4% de
                # discrepancia entre el sigma efectivo y el t declarado. Y el `target`,
                # que es el objetivo de la regresion, arrastraba el mismo 0,4% de error
                # de cuantizacion. El tensor son ~25k elementos: hacerlo en fp32 y
                # castear solo al final no cuesta nada medible.
                # In bf16 the data was built with bf16(sigma) while the model was told
                # bf16(1 - sigma_fp32): two different roundings, ~0.4% mismatch between
                # the effective sigma and the declared t, plus the same 0.4% quantization
                # error on the regression target. The tensor is ~25k elements, so doing
                # it in fp32 and casting at the end costs nothing measurable.
                # ------------------------------------------------------------
                _out_dtype = video_tokens.dtype

                if FP32_NOISE_CONSTRUCTION:
                    _work_dtype = torch.float32
                    t_video = sigma.view(B, 1, 1)
                    t_audio = sigma.view(B, 1, 1)
                else:
                    _work_dtype = torch.bfloat16
                    t_video = sigma.view(B, 1, 1).to(torch.bfloat16)
                    t_audio = sigma.view(B, 1, 1).to(torch.bfloat16)

                _video_w = video_tokens.to(_work_dtype)
                noise_video = torch.randn(
                    _video_w.shape, device=_video_w.device, dtype=_work_dtype)
                flow_delta_video = noise_video - _video_w

                noisy_video = (_video_w + t_video * flow_delta_video).to(_out_dtype)

                if FLOW_TARGET_SIGN >= 0:
                    target_video = flow_delta_video.to(_out_dtype)
                else:
                    target_video = (-flow_delta_video).to(_out_dtype)

                del _video_w, noise_video, flow_delta_video

                _audio_w = audio_tokens.to(_work_dtype)
                noise_audio = torch.randn(
                    _audio_w.shape, device=_audio_w.device, dtype=_work_dtype)
                flow_delta_audio = noise_audio - _audio_w
                noisy_audio = (_audio_w + t_audio * flow_delta_audio).to(_out_dtype)

                if USE_AUDIO_LOSS:
                    if FLOW_TARGET_SIGN >= 0:
                        target_audio = flow_delta_audio.to(_out_dtype)
                    else:
                        target_audio = (-flow_delta_audio).to(_out_dtype)
                else:
                    target_audio = None

                del _audio_w, noise_audio, flow_delta_audio

                if cfg_get("timestep_scale_multiplier", None) is not None:
                    multiplier = safe_float(cfg_get("timestep_scale_multiplier", 1.0), 1.0)
                else:
                    multiplier = safe_float(
                        config_get(config, "timestep_scale_multiplier", 1.0),
                        1.0,
                    )

                if DEBUG_TRAINING and step <= 2:
                    log_print(
                        "[DEBUG-TIMESTEP] multiplier={} | sigma_mean={:.6f} | "
                        "timestep=(1-sigma)*mult={:.6f}".format(
                            multiplier,
                            sigma.mean().item(),
                            (1.0 - sigma.mean().item()) * multiplier,
                        ),
                        flush=True,
                    )

                if DEBUG_TRAINING and step <= 2:
                    _debug_tensor_stats("video_tokens", video_tokens)
                    _debug_tensor_stats("target_video", target_video)
                    _debug_tensor_stats("noisy_video", noisy_video)
                    _debug_tensor_stats("sigma", sigma)

                # Convención del timestep. Un modelo bien cableado da coseno pred/target
                # claramente positivo YA en el paso 1 (el base está entrenado para esto).
                # Timestep convention. A correctly wired model gives a clearly positive
                # pred/target cosine at step 1 already (the base model is trained for it).
                if TIMESTEP_CONVENTION == "sigma":
                    _base_t = sigma
                else:
                    _base_t = 1.0 - sigma

                time_value = _base_t * multiplier

                timestep_video = time_value.to(
                    device="cuda",
                    dtype=torch.bfloat16,
                )

                timestep_audio = time_value.to(
                    device="cuda",
                    dtype=torch.bfloat16,
                )

                (
                    timestep_indices,
                    token_tags,
                    position_ids,
                    video_indices,
                    audio_indices,
                    text_indices,
                ) = build_minimax_packed_indices(
                    B=B,
                    text_len=video_text.shape[1],
                    video_len=video_seq_len,
                    audio_len=audio_seq_len,
                    video_latent_shape=video_latent_shape,
                    patch_t=PATCH_T,
                    patch_h=PATCH_H,
                    patch_w=PATCH_W,
                    device=video_tokens.device,
                )

                del video_clean
                del audio_clean
                del video_tokens
                del audio_tokens
                del prompt_embeds
                del attention_mask

                audio_text = video_text

                forward_kwargs = {
                    "hidden_states": noisy_video,
                    "audio_hidden_states": noisy_audio,
                    "encoder_hidden_states": video_text,
                    "audio_encoder_hidden_states": audio_text,
                    "timestep": timestep_video,
                    "audio_timestep": timestep_audio,
                    "timestep_indices": timestep_indices,
                    "token_tags": token_tags,
                    "position_ids": position_ids,
                    "video_indices": video_indices,
                    "audio_indices": audio_indices,
                    "text_indices": text_indices,
                    "sigma": sigma,
                    "audio_sigma": sigma,
                    "num_frames": num_frames,
                    "height": height,
                    "width": width,
                    "fps": FRAME_RATE,
                    "audio_num_frames": audio_num_frames,
                    "return_dict": False,
                }

                _kwargs_before = set(forward_kwargs.keys())
                forward_kwargs = filter_forward_kwargs(forward_kwargs, transformer.forward)
                if step <= start_step + 1:
                    _dropped = sorted(_kwargs_before - set(forward_kwargs.keys()))
                    log_print("[FORWARD-SIG] kwargs DESCARTADOS (no existen en forward real): {}".format(
                        _dropped if _dropped else "ninguno"
                    ), flush=True)

            loss = None
            pred_video = None
            pred_audio = None

            try:
                # El forward de MiniMax-H3 gestiona sus propios dtypes con casts
                # explicitos (.to(self.proj_in.weight.dtype), etc.); autocast los anula
                # y fuerza bf16 en todo nn.Linear, incluidos los _keep_in_fp32_modules.
                # use_autocast=false ejecuta el forward tal y como lo disenaron.
                # The H3 forward manages its own dtypes with explicit casts; autocast
                # overrides them and forces bf16 on every nn.Linear, including the
                # _keep_in_fp32_modules. use_autocast=false runs it as designed.
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=USE_AUTOCAST):
                    if ACTIVATION_OFFLOAD_ACTIVE:
                        try:
                            with _save_on_cpu_ctx(pin_memory=True):
                                output = model(**forward_kwargs)
                        except Exception as e:
                            log_print("[VRAM] save_on_cpu falló: {}. Desactivando.".format(e), flush=True)
                            ACTIVATION_OFFLOAD_ACTIVE = False
                            output = model(**forward_kwargs)
                    else:
                        output = model(**forward_kwargs)

                if isinstance(output, tuple):
                    if len(output) == 0:
                        raise RuntimeError("forward devolvió tupla vacía.")

                    pred_video = output[0]

                    if len(output) > 1 and USE_AUDIO_LOSS:
                        pred_audio = output[1]
                else:
                    pred_video = getattr(output, "video", None)

                    if pred_video is None:
                        pred_video = getattr(output, "sample", None)

                    if USE_AUDIO_LOSS:
                        pred_audio = getattr(output, "audio", None)

                        if pred_audio is None:
                            pred_audio = getattr(output, "audio_sample", None)

                if pred_video is None:
                    raise RuntimeError("No se pudo obtener predicción de video.")

                if DEBUG_TRAINING and step <= 2:
                    if isinstance(output, tuple):
                        log_print(
                            "[DEBUG-OUTPUT] output es tuple con {} elementos".format(
                                len(output)
                            ),
                            flush=True,
                        )
                    else:
                        log_print("[DEBUG-OUTPUT] output tipo {}".format(type(output)), flush=True)

                    _debug_tensor_stats("pred_video", pred_video)

                if pred_video.shape != target_video.shape:
                    raise RuntimeError(
                        "pred_video shape {} != target_video shape {}. "
                        "Posible orden de salida incorrecto o broadcasting silencioso.".format(
                            pred_video.shape,
                            target_video.shape,
                        )
                    )

                # ------------------------------------------------------------
                # SONDA DE AJUSTE: ¿el LoRA MEJORA la prediccion, o solo la cambia?
                #
                # Mismo latente, mismo ruido, mismo sigma, mismo caption: se evalua el
                # forward CON adaptador y SIN adaptador y se comparan las dos losses.
                # Es la unica medida que responde la pregunta de verdad, porque la loss
                # normal mezcla el efecto del LoRA con el ruido del muestreo de sigma
                # (que hace variar la loss entre 0.12 y 0.53 por si solo).
                #
                #   lora < base de forma consistente -> el entrenamiento SI ajusta la
                #       imagen; si aun asi no hay parecido en inferencia, el problema
                #       esta entre el adaptador entrenado y el modelo de ComfyUI.
                #   lora ~= base con ||dW|| grande -> el optimizador mueve mucho peso
                #       sin mejorar nada: el gradiente no esta conectado al contenido
                #       de la imagen. Ningun hiperparametro arregla eso.
                #
                # Same latent/noise/sigma/caption, adapter ON vs OFF. The plain loss
                # cannot answer this because sigma sampling alone swings it 0.12-0.53.
                # ------------------------------------------------------------
                # Una toma de SOLO AUDIO no tiene imagen que aprender: su fila de
                # video es el fotograma negro de relleno. Entrenarla enseñaria
                # exactamente eso, negro, y ese daño no se queda en las tomas de
                # audio -- va a los mismos pesos LoRA que usan las de video.
                # An AUDIO-ONLY take has no picture to learn: its video row is the
                # black filler frame. Training on it would teach exactly that, and
                # the damage does not stay local -- it lands in the same LoRA
                # weights the video takes use.
                solo_audio = str(entry.get("kind", "video")) == "audio"

                # La sonda mide si el LoRA mejora su prediccion sobre la IMAGEN
                # de la propia muestra. En una toma de solo audio la imagen es el
                # fotograma negro de relleno, que a proposito no recibe gradiente:
                # la sonda avisa correctamente de que no mejora, pero de algo que
                # nadie esta entrenando. Ese aviso, repetido cada N pasos en una
                # corrida que va bien, ensena a ignorar los avisos.
                # The probe measures whether the LoRA improves its prediction on
                # the sample's own IMAGE. On an audio-only take the image is the
                # black filler frame, which deliberately gets no gradient: the
                # warning is correct about something nobody is training. Repeated
                # every N steps through a healthy run, it teaches people to ignore
                # warnings.
                if (OVERFIT_PROBE_EVERY > 0 and step % OVERFIT_PROBE_EVERY == 0
                        and not solo_audio):
                    try:
                        _tv = target_video.detach().float()
                        _var = max(float(_tv.var()), 1e-8)
                        _l_lora = float(
                            (pred_video.detach().float() - _tv).pow(2).mean()) / _var

                        with torch.no_grad():
                            with model.disable_adapter():
                                _ob = model(**forward_kwargs)
                            _pb = _ob[0] if isinstance(_ob, tuple) else getattr(
                                _ob, "sample", None)
                            _l_base = float(
                                (_pb.detach().float() - _tv).pow(2).mean()) / _var
                            del _ob, _pb

                        _gain = (_l_base - _l_lora) / max(1e-8, _l_base)
                        log_print(
                            "[FIT-PROBE] step={} | sigma={:.3f} | loss SIN lora={:.4f} | "
                            "loss CON lora={:.4f} | mejora={:+.1%}".format(
                                step, float(sigma.reshape(-1)[0]), _l_base, _l_lora,
                                _gain),
                            flush=True)

                        if _gain < 0.02:
                            log_print(
                                "[FIT-PROBE][WARN] El LoRA no mejora la prediccion sobre "
                                "su PROPIA imagen de entrenamiento. Si ||dW|| es grande, "
                                "el optimizador mueve peso sin ajustar contenido: el "
                                "gradiente no esta conectado a la imagen. / The LoRA does "
                                "not improve prediction on its own training image.",
                                flush=True)
                    except Exception as _e:
                        log_print("[FIT-PROBE] fallo: {}".format(_e), flush=True)

                # El mismo diagnostico para el AUDIO. Es la unica senal que dice,
                # en el primer paso y sin esperar una hora, si la cadena entera
                # esta bien conectada: si el coseno sale cerca de +1 el modelo ya
                # predice la direccion del flujo, y si sale cerca de 0 hay algo
                # roto entre el latente y la loss. Con solo audio el bloque de
                # abajo mira `pred_video`, que es el fotograma negro de relleno, y
                # daria un numero sin significado.
                #
                # The same probe for AUDIO. It is the one signal that says, on the
                # first step and without waiting an hour, whether the whole chain
                # is wired: a cosine near +1 means the model already predicts the
                # flow direction, near 0 means something is broken between latent
                # and loss. On audio-only the block below looks at `pred_video`,
                # which is the black filler frame, and would report a meaningless
                # number.
                if (step <= max(2, FLOW_CONV_DEBUG_STEPS)
                        and pred_audio is not None and target_audio is not None):
                    _pa = pred_audio.detach().float().reshape(1, -1)
                    _ta = target_audio.detach().float().reshape(1, -1)
                    print(
                        "[FLOW-CONV][AUDIO] step={} | cosine(pred,target)={:+.4f} | "
                        "std pred/target={:.2f}/{:.2f} | rows={}".format(
                            step,
                            torch.nn.functional.cosine_similarity(_pa, _ta).item(),
                            float(_pa.std()), float(_ta.std()),
                            int(target_audio.shape[1]) if target_audio.ndim > 1 else -1),
                        flush=True)
                    del _pa, _ta

                if step <= max(2, FLOW_CONV_DEBUG_STEPS) and not solo_audio:
                    pred_flat = pred_video.detach().float().reshape(1, -1)
                    targ_flat = target_video.detach().float().reshape(1, -1)

                    cos = torch.nn.functional.cosine_similarity(
                        pred_flat, targ_flat
                    ).item()

                    target_var = target_video.detach().float().var().item()

                    print(
                        "[FLOW-CONV] step={} conv={} mult={:g} sigma={:.3f} t={:.4f} | "
                        "cosine(pred,target)={:+.4f} | std pred/target={:.2f}/{:.2f} | "
                        "target_var={:.4g}".format(
                            step, TIMESTEP_CONVENTION, multiplier,
                            float(sigma.reshape(-1)[0]),
                            float(time_value.reshape(-1)[0]),
                            cos,
                            float(pred_video.detach().float().std()),
                            float(target_video.detach().float().std()),
                            target_var,
                        ),
                        flush=True,
                    )

                if not USE_AUDIO_LOSS:
                    pred_audio = None

                # Un clip MUDO trae un relleno de silencio de una sola fila, no
                # una pista. Entrenarlo le enseña al modelo que ese video suena a
                # silencio, y con use_audio_loss activo -- que hace falta para las
                # tomas de voz -- eso entraba al 50% del peso de la muestra,
                # trabajando justo contra lo que se quiere aprender. Es lo que dice
                # el docstring de make_audio_latent y no estaba implementado: los
                # trainers de referencia le dan peso CERO.
                #
                # Se detecta por el tamaño, que no admite duda: el relleno es 1
                # fila, y la pista mas corta que la rejilla permite (22 fotogramas,
                # 0,917 s) son 74.
                #
                # A MUTE clip carries a one-row silence placeholder, not a track.
                # Training it teaches the model that this video sounds like
                # silence, and with use_audio_loss on -- needed for the voice takes
                # -- that entered at 50% of the sample's weight, working against
                # the very thing being learned. make_audio_latent's docstring
                # already says the reference trainers give it ZERO weight; it was
                # not implemented. Detected by size, unambiguously: the placeholder
                # is 1 row and the shortest track the grid allows is 74.
                audio_es_relleno = (target_audio is not None
                                    and target_audio.ndim > 1
                                    and int(target_audio.shape[1]) <= 4)

                if solo_audio and USE_AUDIO_LOSS and pred_audio is not None \
                        and target_audio is not None:
                    loss = mse_loss_chunked(pred_audio, target_audio)
                elif solo_audio:
                    # Sin loss de audio, una toma de solo audio no aporta nada:
                    # se salta en vez de entrenar el relleno negro.
                    # With no audio loss, an audio-only take contributes nothing:
                    # skip it rather than train the black filler.
                    raise RuntimeError(
                        "Audio-only sample '{}' with use_audio_loss=False: nothing "
                        "to train. Enable Train Audio. / Muestra de solo audio '{}' "
                        "con use_audio_loss=False: no hay nada que entrenar. "
                        "Activa Train Audio.".format(entry.get("name"), entry.get("name")))
                elif (USE_AUDIO_LOSS and pred_audio is not None
                        and target_audio is not None and not audio_es_relleno):
                    loss_video = mse_loss_chunked(pred_video, target_video)
                    loss_audio = mse_loss_chunked(pred_audio, target_audio)
                    loss = (loss_video + loss_audio) * 0.5
                else:
                    loss = mse_loss_chunked(pred_video, target_video)

                loss = loss / GRAD_ACCUM_STEPS
                current_loss = loss.item() * GRAD_ACCUM_STEPS

                # ------------------------------------------------------------
                # gnorm_grad (norma SOLO del gradiente de este micro-paso) es un
                # diagnostico caro: exige un clon fp32 de TODOS los gradientes vivo
                # durante el backward (~144 MB con rank 8, ~287 MB con rank 16) mas un
                # .item() por tensor, que es una sincronizacion GPU completa. Con ~600
                # tensores entrenables eso eran ~1.800 sincronizaciones por paso, y bajo
                # WDDM cada una cuesta bastante mas que en Linux. Ahora solo se calcula
                # en los dos primeros pasos, y con una unica sincronizacion.
                # gnorm_grad (norm of THIS micro-step's gradient alone) is an expensive
                # diagnostic: an fp32 clone of every gradient held live through backward,
                # plus one .item() per tensor -- a full GPU sync. With ~600 trainable
                # tensors that was ~1800 syncs per step. Now it only runs on the first two
                # steps, with a single sync.
                # ------------------------------------------------------------
                _want_grad_delta = bool(DEBUG_TRAINING and step <= start_step + 2)

                prev_grads = {}
                if _want_grad_delta:
                    for p in trainable:
                        if p.grad is not None:
                            prev_grads[p] = p.grad.detach().float().clone()

                loss.backward()

                def _stacked_norm(tensors):
                    """Una sola sincronizacion en vez de una por tensor.
                    One single sync instead of one per tensor."""
                    if not tensors:
                        return 0.0
                    parts = [torch.linalg.vector_norm(t, ord=2) for t in tensors]
                    return float(torch.linalg.vector_norm(torch.stack(parts), ord=2).item())

                with torch.no_grad():
                    if _want_grad_delta:
                        _deltas = []
                        for p in trainable:
                            if p.grad is None:
                                continue
                            current = p.grad.detach().float()
                            previous = prev_grads.get(p)
                            _deltas.append(current - previous if previous is not None
                                           else current)
                        gnorm_grad = _stacked_norm(_deltas)
                        del _deltas
                    else:
                        gnorm_grad = 0.0

                    gnorm_acc = _stacked_norm(
                        [p.grad.detach().float() for p in trainable if p.grad is not None])

                prev_grads = None

                if DEBUG_TRAINING and step <= 2:
                    vram_stats("Post-backward (swap debe haber liberado bloques CPU)")

                running_loss += current_loss

                if ema_loss is None:
                    ema_loss = current_loss
                else:
                    ema_loss = 0.98 * ema_loss + 0.02 * current_loss

                if DEBUG_TRAINING and step <= 2:
                    target_var = target_video.detach().float().var().item()
                    normalized_loss = current_loss / max(target_var, 1e-8)

                    log_print(
                        "[DEBUG-LOSS] current_loss={:.6g} | ema_loss={:.6g} | target_var={:.6g} | normalized_loss={:.6g}".format(
                            current_loss,
                            ema_loss,
                            target_var,
                            normalized_loss,
                        ),
                        flush=True,
                    )

                if DEBUG_TRAINING and (step <= 2 or step % 10 == 0):
                    n_trainable = len(trainable)
                    n_with_grad = 0
                    n_nonzero = 0

                    with torch.no_grad():
                        _norms = []
                        for p in trainable:
                            if p.grad is not None:
                                n_with_grad += 1
                                _norms.append(torch.linalg.vector_norm(
                                    p.grad.detach().float(), ord=2))
                        if _norms:
                            # Un solo .item() sobre el stack: antes eran ~600.
                            # One single .item() over the stack: it used to be ~600.
                            _nv = torch.stack(_norms)
                            n_nonzero = int((_nv > 0).sum().item())
                            total_norm = float(torch.linalg.vector_norm(_nv, ord=2).item())
                        else:
                            total_norm = 0.0
                        del _norms

                    log_print(
                        "[DEBUG-GRAD] step={} | trainable={} | with_grad={} | nonzero_grad={} | grad_norm={:.6g}".format(
                            step,
                            n_trainable,
                            n_with_grad,
                            n_nonzero,
                            total_norm,
                        ),
                        flush=True,
                    )

                    # nonzero_grad = trainable/2 es NORMAL en los primeros pasos:
                    # lora_B se inicializa a ceros, asi que grad(lora_A) = B^T(...) = 0
                    # hasta que el optimizador mueve B. Si sigue a la mitad pasado el
                    # paso 20, es que B no se esta actualizando (lr efectivo 0, estado
                    # del optimizador roto o pesos LoRA en un dtype que redondea a 0).
                    # Half is NORMAL early on (lora_B starts at zero). Still half after
                    # step 20 means B is not being updated at all.
                    if step > 20 and n_with_grad > 0 and n_nonzero <= n_with_grad // 2:
                        log_print(
                            "[DEBUG-GRAD][WARN] step={}: la mitad de los tensores siguen "
                            "con gradiente CERO. lora_B no se esta moviendo -> el LoRA no "
                            "aprende. Revisa lr, lora_dtype y el estado del optimizador. "
                            "/ half the tensors still have ZERO grad: lora_B is not "
                            "moving.".format(step), flush=True)

                # ------------------------------------------------------------
                # PERFIL DE GRADIENTE POR BLOQUE.
                #
                # El LoRA de referencia reparte ||dW|| de forma casi plana por los 50
                # bloques (11 -> 23 -> 10). El nuestro sale en RAMPA: 1.2 en el bloque 0
                # y 7.7 en el 49, un factor 7 incluso despues de normalizar por escala.
                #
                # Eso NO se explica por gradientes pequenos: Adam es invariante a escala
                # por parametro, un gradiente diminuto pero CONSISTENTE sigue dando un
                # paso de tamano ~lr. Una rampa asi significa que los bloques tempranos
                # reciben gradiente INCONSISTENTE (ruido que se cancela entre pasos),
                # normalmente porque la senal ha perdido precision al retropropagarse
                # por 50 bloques en bf16.
                #
                # Aqui se mide directamente: norma de gradiente agregada por tercios de
                # profundidad. Si el tercio inicial esta ordenes de magnitud por debajo
                # del final, esta confirmado.
                # The reference LoRA spreads ||dW|| almost flat across the 50 blocks;
                # ours ramps 7x. Adam is per-parameter scale invariant, so a ramp means
                # INCONSISTENT gradients in early blocks, not merely small ones.
                # ------------------------------------------------------------
                if (GRAD_PROFILE_EVERY > 0 and step % GRAD_PROFILE_EVERY == 0):
                    try:
                        import re as _re
                        _per_block = {}
                        with torch.no_grad():
                            for _n, _p in model.named_parameters():
                                if not _p.requires_grad or _p.grad is None:
                                    continue
                                _m = _re.search(r"blocks\.(\d+)\.", _n)
                                if not _m:
                                    continue
                                _bi = int(_m.group(1))
                                _g = float(torch.linalg.vector_norm(
                                    _p.grad.detach().float(), ord=2))
                                _per_block[_bi] = _per_block.get(_bi, 0.0) + _g * _g

                        if _per_block:
                            _idx = sorted(_per_block)
                            _vals = [math.sqrt(_per_block[i]) for i in _idx]
                            _n3 = max(1, len(_vals) // 3)
                            _early = sum(_vals[:_n3]) / _n3
                            _late = sum(_vals[-_n3:]) / _n3
                            _ratio = _late / max(1e-12, _early)
                            log_print(
                                "[GRAD-PROFILE] step={} | bloques {}-{} | grad medio "
                                "tercio INICIAL={:.3e} | tercio FINAL={:.3e} | "
                                "final/inicial={:.1f}x".format(
                                    step, _idx[0], _idx[-1], _early, _late, _ratio),
                                flush=True)
                            if _ratio > 5.0:
                                log_print(
                                    "[GRAD-PROFILE][WARN] El gradiente que llega a los "
                                    "bloques tempranos es {:.0f}x menor que el de los "
                                    "finales. El LoRA se esta entrenando casi solo en la "
                                    "segunda mitad de la red. / Early blocks receive "
                                    "{:.0f}x less gradient than late ones.".format(
                                        _ratio, _ratio), flush=True)
                    except Exception as _e:
                        log_print("[GRAD-PROFILE] fallo: {}".format(_e), flush=True)

            finally:
                loss = None
                pred_video = None
                pred_audio = None
                output = None

                free_vram(
                    noisy_video,
                    noisy_audio,
                    target_video,
                    target_audio,
                    sigma,
                    timestep_video,
                    timestep_audio,
                    timestep_indices,
                    token_tags,
                    position_ids,
                    video_indices,
                    audio_indices,
                    text_indices,
                    video_text,
                    audio_text,
                    forward_kwargs,
                    clear_cache=False,
                    # gc.collect() completo por paso costaba 0,5-2 s sobre el heap de un
                    # modelo de 33B. Se deja solo para el ciclo de limpieza de cada 25.
                    # A full gc.collect() per step cost 0.5-2 s on a 33B model's heap.
                    # It is kept only for the every-25-steps cleanup below.
                    collect=False,
                )

                _clear_offload_cpu_cache()

                # ------------------------------------------------------------
                # FRAGMENTACION DEL CACHING ALLOCATOR.
                #
                # El block swap pide y suelta cientos de tensores por paso. PyTorch
                # NO devuelve esa memoria al driver: se queda en el pool (`reserved`),
                # que es justo lo que ve nvidia-smi y lo que satura la tarjeta. En tu
                # log: Alloc 5.48 GB / Reserved 15.98 GB -> 10 GB de pool.
                # Antes esto solo se limpiaba cada 25 pasos. A 20 s/it, hacerlo cada
                # paso cuesta milisegundos y mantiene `reserved` pegado a `alloc`.
                # The swap churns the allocator; PyTorch keeps the freed blocks in its
                # pool (`reserved`), which is what nvidia-smi shows and what fills the
                # card. Cleaning every step costs milliseconds at 20 s/it.
                # ------------------------------------------------------------
                if (torch.cuda.is_available() and VRAM_EMPTY_CACHE_EVERY > 0
                        and step % VRAM_EMPTY_CACHE_EVERY == 0):
                    torch.cuda.empty_cache()

                if torch.cuda.is_available() and (step % 25 == 0):
                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

            if step % GRAD_ACCUM_STEPS == 0:
                grad_norm = clip_grad_norm_mixed_device(trainable, MAX_GRAD_NORM)
                gnorm_acc = float(grad_norm)
                current_lr = lr_at(step)

                for group in optimizer.param_groups:
                    group["lr"] = current_lr

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                # Update aplicado y gradientes limpios: a partir de aqui `step`
                # es un punto de reanudacion consistente.
                # Update applied and grads cleared: `step` is now a consistent
                # resume point.
                last_completed_step = step
            else:
                # gnorm_acc ya se calculo tras el backward con una sola sincronizacion;
                # antes se recalculaba aqui con un .item() por tensor. / gnorm_acc was
                # already computed after backward with a single sync; this used to redo
                # it with one .item() per tensor.
                grad_norm = gnorm_acc
                current_lr = lr_at(step)

            elapsed = time.time() - t0
            avg_time = elapsed if avg_time == 0 else (0.1 * elapsed + 0.9 * avg_time)
            step_time_sum += elapsed
            step_time_count += 1
            mean_time = step_time_sum / step_time_count

            eta_s = (TOTAL_STEPS - step) * avg_time

            eta = "{:02d}:{:02d}:{:02d}".format(
                int(eta_s // 3600),
                int((eta_s % 3600) // 60),
                int(eta_s % 60),
            )

            pct = step / TOTAL_STEPS
            barra = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))

            # running_loss se restaura del checkpoint acumulado desde el paso 1,
            # asi que hay que dividir por `step`, no por los pasos transcurridos
            # desde la reanudacion: con `step - start_step` el primer paso tras
            # reanudar mostraba la suma entera de la corrida dividida por 1.
            # En una corrida continua start_step=0 y las dos formulas coinciden.
            # running_loss is restored cumulative from step 1, so divide by
            # `step`. With `step - start_step` the first step after a resume
            # showed the whole run's sum divided by one.
            avg_loss = running_loss / max(1, step)
            display_loss = ema_loss if ema_loss is not None else avg_loss

            if isinstance(grad_norm, torch.Tensor):
                try:
                    display_grad_norm = grad_norm.detach().float().cpu().item()
                except Exception:
                    display_grad_norm = float(grad_norm)
            else:
                display_grad_norm = float(grad_norm)

            # El lr se mantiene aunque hoy sea constante: en cuanto haya un
            # programa de lr (alto al principio, bajo al final) esta es la unica
            # ventana para ver que lo esta siguiendo.
            #
            # De los tres gnorm que habia aqui solo queda uno. gnorm y gnorm_acc
            # eran literalmente la misma variable impresa dos veces (grad_norm =
            # gnorm_acc unas lineas mas arriba), y gnorm_grad vale 0 salvo que se
            # active su sonda. Ademas iban con {:.17g}, que son 17 cifras
            # significativas de una magnitud que se lee de un vistazo: cuatro
            # decimales dicen lo mismo y dejan sitio en la linea.
            #
            # The lr stays even though it is constant today: the moment there is
            # an lr schedule (high early, low late) this is the only window into
            # whether it is being followed. Of the three gnorms only one remains:
            # gnorm and gnorm_acc were the same variable printed twice
            # (grad_norm = gnorm_acc a few lines up) and gnorm_grad is 0 unless
            # its probe is enabled. They also used {:.17g} -- 17 significant
            # digits of a number read at a glance; four decimals say the same and
            # leave room on the line.
            progress_line = (
                "Step {:4d}/{} [{}] {:5.1f}% | "
                "Loss {:.4f} | lr {:.2e} | {:.2f}s/it (now {:.2f}) | ETA {} | "
                "gnorm {:.4f}".format(
                    step,
                    TOTAL_STEPS,
                    barra,
                    pct * 100,
                    display_loss,
                    current_lr,
                    mean_time,
                    avg_time,
                    eta,
                    display_grad_norm,
                )
            )

            print("\r{}".format(progress_line), end="", flush=True)
            _PROGRESS_LINE["open"] = True

            # ----------------------------------------------------------------
            # SONDA DE PRESION DE MEMORIA.
            #
            #   retries subiendo      -> presion de VRAM / fragmentacion.
            #                            Baja vram_budget_gb.
            #   inact_split alto      -> fragmentacion del asignador (en Windows el
            #                            script desactiva expandable_segments).
            #   rss cerca de la RAM   -> thrashing del archivo de paginacion: el block
            #     fisica                 swap deja de leer de RAM y lee de disco. Es el
            #                            escenario que convierte 2 h en 34 h.
            #
            #   retries climbing      -> VRAM pressure / fragmentation. Lower vram_budget_gb.
            #   inact_split high      -> allocator fragmentation.
            #   rss near physical RAM -> page-file thrashing: the block swap reads from
            #                            disk instead of RAM. This is what turns 2 h into 34 h.
            # ----------------------------------------------------------------
            if ALLOC_PROBE and torch.cuda.is_available() and (step <= 5 or step % 10 == 0):
                try:
                    _ms = torch.cuda.memory_stats()
                    _vm = psutil.virtual_memory()
                    print("", flush=True)
                    print("[ALLOC] step={} retries={} ooms={} | reserved={:.2f}GB "
                          "active={:.2f}GB inact_split={:.2f}GB | rss={:.1f}GB "
                          "ram_used={:.1f}/{:.1f}GB ({:.0f}%)".format(
                              step,
                              _ms.get("num_alloc_retries", 0),
                              _ms.get("num_ooms", 0),
                              _ms.get("reserved_bytes.all.current", 0) / 1e9,
                              _ms.get("active_bytes.all.current", 0) / 1e9,
                              _ms.get("inactive_split_bytes.all.current", 0) / 1e9,
                              psutil.Process(os.getpid()).memory_info().rss / 1e9,
                              (_vm.total - _vm.available) / 1e9,
                              _vm.total / 1e9,
                              _vm.percent),
                          flush=True)
                except Exception as _e_alloc:
                    print("[ALLOC] probe failed / la sonda fallo: {}".format(_e_alloc),
                          flush=True)

            if step <= 5 or step % 10 == 0:
                vram_stats("Step {}".format(step))

            if SAVE_EVERY > 0 and step % SAVE_EVERY == 0:
                save_checkpoint_now(step)

            # ----------------------------------------------------------------
            # PREVISUALIZACION DE PROGRESO.
            #
            # Va DESPUES del checkpoint para que, si la preview reventase, el
            # estado ya este a salvo en disco. El try/except es deliberado y no
            # se debe quitar: una preview fallida (OOM del VAE, PIL ausente,
            # cualquier cosa) no puede tumbar un entrenamiento de horas; se
            # avisa, se desactivan las siguientes y se sigue entrenando.
            #
            # Runs AFTER the checkpoint so a failing preview cannot cost state.
            # The try/except is deliberate: a broken preview must never kill an
            # hours-long run, so it warns, disables itself and training goes on.
            # ----------------------------------------------------------------
            if PREVIEW_EVERY > 0 and step % PREVIEW_EVERY == 0:
                try:
                    _t_prev = time.time()
                    log_print("")
                    log_print("[PREVIEW] Rendering step {} preview / Generando preview "
                              "del paso {} ({} steps/pasos, shift {:g}, CFG {:g}, VAE "
                              "{})".format(step, step, PREVIEW_STEPS, PREVIEW_SHIFT,
                                           PREVIEW_CFG, PREVIEW_VAE_DEVICE), flush=True)
                    _p = run_training_preview(
                        model, entries, step, OUTPUT_DIR, NF4_CACHE_DIR,
                        PATCH_T, PATCH_H, PATCH_W, audio_channels=audio_channels)
                    log_print("[PREVIEW] Saved to / Guardada en: {} ({:.1f}s)"
                              .format(_p, time.time() - _t_prev), flush=True)
                except Exception as _e_prev:
                    PREVIEW_EVERY = 0
                    release_preview_video_vae()
                    log_print("[PREVIEW][WARN] The preview failed and is now DISABLED "
                              "for the rest of the run; training continues. Reason: {} / "
                              "La preview fallo y se DESACTIVA para el resto de la "
                              "corrida; el entrenamiento sigue. Motivo: {}"
                              .format(_e_prev, _e_prev), flush=True)
                    if DEBUG_TRAINING:
                        traceback.print_exc()

    except KeyboardInterrupt:
        # Se guarda el ultimo paso COMPLETADO, no el que estaba en vuelo cuando
        # llego la senal. Reanudar repite ese paso interrumpido desde cero, que
        # es exactamente lo que haria una corrida sin parada.
        # Checkpoint the last COMPLETED step, not the in-flight one. Resuming
        # then re-runs the interrupted step from scratch, which is exactly what
        # an uninterrupted run would have done.
        log_print("[STOP] Interrupted during step {}; checkpointing the last complete "
                  "step: {}. / Interrumpido durante el paso {}; se guarda el ultimo paso "
                  "completo: {}.".format(
                      step if 'step' in locals() else "?", last_completed_step,
                      step if 'step' in locals() else "?", last_completed_step),
                  flush=True)
        save_checkpoint_now(last_completed_step)
        log_print("Total training time / Tiempo total de entrenamiento: {}"
                  .format(format_duration_bilingual(total_elapsed_seconds())), flush=True)
        close_train_log()
        return
    except SystemExit:
        close_train_log()
        return

    log_print()
    log_print()
    log_print("Training completed!", flush=True)
    log_print("=" * 60)
    log_print("Total training time / Tiempo total de entrenamiento: {}"
              .format(format_duration_bilingual(total_elapsed_seconds())), flush=True)
    log_print("=" * 60)

    save_checkpoint_now(TOTAL_STEPS)

    final_path = os.path.join(OUTPUT_DIR, "MiniMaxH3_FINAL_LoRA.safetensors")
    save_lora(model, final_path)

    log_print("Final LoRA saved to: {}".format(final_path), flush=True)

    runtime["completed"] = True
    runtime["last_step"] = TOTAL_STEPS

    close_train_log()

if __name__ == "__main__":
    try:
        train_minimaxh3()
    except Exception:
        log_print()
        log_print("=" * 80)
        log_print("ERROR EN TRAINER MINIMAX-H3")
        log_print("=" * 80)
        # El traceback va a stderr, que tambien esta duplicado al fichero, asi
        # que un fallo queda registrado en train_log.txt antes de cerrarlo.
        # The traceback goes to stderr, which is mirrored too, so a crash is
        # recorded in train_log.txt before it is closed.
        traceback.print_exc()
        close_train_log()
        raise