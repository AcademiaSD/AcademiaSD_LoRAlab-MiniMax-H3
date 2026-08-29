# AcademiaSD LoRAlab-MiniMax-H3 Beta v0.1

![AcademiaSD LoRAlab MiniMax-H3](assets/portada.jpg)

<p align="center">
  <b>Train MiniMax-H3 character & style LoRAs on a consumer GPU — a 33-Billion parameter joint video+audio DiT, on 16 GB of VRAM.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-NVIDIA-green.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/UI-Flask%20%2B%20HTML5-purple.svg" alt="Web UI">
  <img src="https://img.shields.io/badge/Model-MiniMax--H3%2033B-red.svg" alt="MiniMax-H3">
</p>

---

## 🎯 What this is

**MiniMax-H3** is a 33-Billion parameter Diffusion Transformer that generates video **and audio jointly**, from a single packed sequence. The official checkpoint is **498 GB**; the generic partition alone is **135 GB**. Training a LoRA on it normally means datacenter hardware.

**AcademiaSD LoRAlab MiniMax-H3** trains character and style LoRAs from a folder of **images and captions**, on a **16 GB consumer GPU**, in about **two hours**.

> **Verified result:** 8 images at 768×768, 500 steps, LR 2e-4, rank/alpha 16 → excellent likeness, confirmed in ComfyUI video generation. The trainer runs at ~8.4 s/it on an RTX 5080 (16 GB).

This trainer is for **image datasets** (characters, faces, styles). It does not train video clips or audio.

---

## 🔬 Technical Deep-Dive: how a 33B model fits in 16 GB

### 1. 📉 The model is compressed 135 GB → 39 GB

* **4-bit NormalFloat (NF4) quantization**: the 50-block DiT backbone and the Qwen3-VL-32B text encoder are stored as `Linear4bit` NF4 weights. Measured on disk: **135 GB → 39 GB**. Only **86,474,752** parameters are trainable — **0.26 %** of the model's 33,209,467,648.
* **Precision-critical modules stay in float**: the model declares `proj_in`, `audio_proj_in`, `time_embedder`, `proj_out`, `audio_proj_out` and `rope` as `_keep_in_fp32_modules`. These plus the modulation path (`norm_out`, `context_embedder`, `token_refiner`) are reloaded **without** NF4 from a `precision_critical` section of the quantized repo. Quantization error there breaks the final AdaLN cancellation.
* **Adjustable block swap**: transformer blocks are parked in system RAM and streamed to VRAM just in time, with a **hard cap** (`set_per_process_memory_fraction`) so the process physically cannot exceed its budget. This is what lets you *simulate* a smaller card and know whether a run would fit on 12 GB before you own one.
* **Zero VRAM spent on encoders**: during training **neither the text encoder nor the VAE are loaded**. Every embedding and latent is computed once, offline, in the pre-cache stage.
* **fp32 LoRA weights and optimizer state**: deliberately *not* 8-bit. On H3 the per-parameter LoRA gradients are tiny; `AdamW8bit` quantizes `exp_avg_sq` (squared gradients ~1e-6) and destroys exactly the low-magnitude components where facial detail lives. The result is a LoRA that gets pose, hair and framing right and leaves the face soft. fp32 costs ~6 bytes/param and fixes it.

### 2. ⚡ Why it is fast

* **No per-step encoding**: 100 % of GPU compute during training goes to the DiT forward/backward. No VAE, no 32B language model in the loop.
* **(1,2,2) patch packing**: video latents are packed into 2×2 spatial patches, cutting the self-attention sequence length 4×.
* **Reentrant gradient checkpointing**: `bitsandbytes` stashes its quantized weight as a plain `ctx` attribute rather than through `save_for_backward`, so non-reentrant checkpointing does **not** discard it — every executed `Linear4bit` would pin its NF4 weight in VRAM until backward, and the block swap would silently stop working. Reentrant checkpointing runs the forward under `no_grad`, so nothing is pinned.
* **Pinned RAM and non-blocking transfers** for the cached latents and embeddings.

### 3. 🎨 Why the likeness is preserved

* **The exact H3 conditioning**: MiniMax-H3 conditions on the **unnormalized hidden state after decoder layer 50** of Qwen3-VL-32B — not the last layer, not a normalized one. The pre-cache truncates the 64-layer stack to 50 and replaces the final norm with `Identity`, so `hidden_states[50]` is the raw layer-50 output the model expects.
* **The exact VAE convention**: ImageNet-normalized pixels over a `[0,1]` base (not the usual `[-1,1]`), then per-channel `(z − latents_mean) / latents_std`.
* **The exact flow-matching convention**: `noised = (1−σ)·x0 + σ·noise`, timestep `t = 1−σ` in `[0,1]` **unscaled** with `t = 1` meaning *clean*, and a data-ward velocity target of `x0 − noise`. H3 inverts the sign relative to standard flow-match schedulers.
* **The correct sigma schedule**: logit-normal sampling with the resolution-dependent shift `mu = 0.5 + (tokens−256)·(1.15−0.5)/(6400−256)`. The video sampler's `shift = 12.0` is **wrong for training** and ruins likeness.
* **Correct export keys**: the trained adapter is translated from the diffusers module layout to the original checkpoint layout that ComfyUI loads — including the **SwiGLU half swap on `mlp.fc1`** (the reference stores `[gate; value]`, diffusers stores `[value; gate]`) and the QKV fusion. Getting this wrong produces a LoRA that visibly changes the output while learning nothing about your subject.

---

## ✨ Features

### Training
- **🌐 Modern Web GUI** — pre-cache, dataset editing, training, previews, checkpoints and export from a single-page Flask app.
- **🚀 1-Click launch** — `Run_LoRAlab-MiniMaxH3.bat` starts the server and opens `http://127.0.0.1:5000`.
- **🎛️ VRAM profiles** — a dropdown with presets for **32, 24, 16, 12, 10, 8, 6 and 4 GB** cards. Every field stays editable by hand; the dropdown switches to *Custom* the moment you type, so it never advertises a preset that does not match your values.
- **⏱️ Exact-step resume** — stop at any step and continue from it. The checkpoint is written atomically with a strict invariant (weights → optimizer → step file, and the step file only if the first two succeeded), so a kill mid-write can never leave an inconsistent resume point.
- **🎲 Deterministic across pauses** — the RNG is seeded **per step** with a splitmix64 mix of `(seed, step)`, and the dataset sampler is a shuffled per-epoch permutation derived from the same step index. Step N always sees the same image, sigma and noise whether it came from one continuous run or ten resumes.
- **🔁 Live settings** — edit preview settings, `save_every`, `lr`, `max_grad_norm` or lower `total_steps` **while training is running**, press *Save JSON*, and the change lands on the next step. Costs one `getmtime` per step.
- **📈 Total training time** — reported bilingually at the end (`2 Hours 26 Minutes / 2 Horas 26 Minutos`), accumulated across resumes.
- **📝 Full run log on disk** — everything the console shows is mirrored to `<output_dir>/train_log.txt`, flushed on every write, so a crash or a closed browser tab never loses the history.

### Previews
- **🖼️ In-training previews** — every N steps the current LoRA generates an image, shown live in the gallery. `0` disables it completely.
- **🎬 Real joint sampling** — the preview runs the actual packed `[text | audio | video]` sequence with two independent sigma schedules and per-row timesteps, which is the layout the model was trained on.
- **🗣️ Four prompt modes** — *First* (comparable across steps), *Random* (varied), *Rotate* (covers the dataset), *Custom* (a free prompt, encoded by the pre-cache).
- **🎚️ Sampler control** — MiniMax-H3 ships exactly one sampler (rectified-flow Euler, `eta = 0`); its sigma shift is the only real knob and is exposed with sensible presets.
- **💾 VAE on CPU or CUDA** — the VAE decoder is ~2.4 G parameters (4.8 GB in bf16). On CPU it costs ~90 s and touches no VRAM; on a 24 GB+ card, CUDA takes seconds.

### Dataset
- **🖼️ Dataset Inspector & caption editor** — visual grid with caption status badges (🟢 present / 🔴 missing), filename overlays, a lightbox to edit `.txt` captions directly on disk, and a batch tool to inject trigger words.
- **📊 Dataset summary** — image count, epochs, and the video-token grid, reported before training starts so you can plan the next experiment.

### System
- **📊 Real-time telemetry** — RAM, VRAM and GPU temperature with colour coding.
- **🔑 Hugging Face token support** — optional `HF_token.json` for faster downloads with live progress.
- **📂 Automatic project folders** — `./cached_data_minimaxh3_<project>` and `./minimaxh3_lora_output_<project>`.
- **🚀 One-click export** — send the finished `.safetensors` to your ComfyUI / Forge / A1111 `models/loras` folder.
- **🌐 Fully bilingual (English / Español)** — every message is `English / Español` on the same line, English first.

---

## 🖥️ System Requirements

| Requirement | Minimum | Recommended |
| :--- | :--- | :--- |
| **OS** | Windows 10/11 | Windows 11 |
| **GPU** | NVIDIA, **12 GB VRAM** | NVIDIA, **16–24 GB VRAM** |
| **System RAM** | **32 GB** | **64 GB+** (block swap parks the model in RAM) |
| **Disk** | **45 GB** free for the NF4 model | 60 GB+ |
| **Python** | 3.10+ (inside `venv`) | 3.11 / 3.13 |
| **CUDA** | 12.1+ | 12.8 / 13.0 |

### VRAM profiles

| Profile | GPU VRAM | Swap | Headroom | Status |
| :--- | ---: | ---: | ---: | :--- |
| 32 GB | 30.0 | 1.35 | 0.1 | Comfortable |
| 24 GB | 22.0 | 1.35 | 0.1 | Comfortable |
| **16 GB** | **14.0** | **1.35** | **0.1** | **Verified** |
| 12 GB | 10.0 | 1.35 | 0.1 | Expected to work |
| 10 GB | 8.0 | 1.35 | 0.1 | Expected to work |
| 8 GB | 6.0 | 1.35 | 0.1 | Untested |
| 6 GB | 4.0 | 1.35 | 0.1 | Optimistic — may not fit |
| 4 GB | 2.0 | 1.35 | 0.1 | Optimistic — may not fit |

The 6 GB and 4 GB profiles are included for experimentation. The swap budget has to hold one NF4 block **plus** the full bf16 weight that bitsandbytes materializes for each matmul, so very small budgets may not fit a block with its activations. Reports welcome.

Reference timing: **~8.4 s/it** on an RTX 5080 16 GB at 768×768 — about **2 hours for 1000 steps**.

---

## 📦 Installation

1. **Clone or download the repository**:
   ```bash
   git clone https://github.com/AcademiaSD/AcademiaSD_LoRAlab-MiniMaxH3.git
   cd AcademiaSD_LoRAlab-MiniMaxH3
   ```

2. **Install the virtual environment and dependencies**:
   Double-click `Install_LoRAlab-venv-Minimax.bat`.

3. **(Optional) Install Triton & SageAttention 2.2**:
   Double-click `Install_Triton&SageAtten220.bat`.

4. **The model downloads itself.** On the first pre-cache run, the quantized repo **`AcademiaSD/MiniMax-H3-NF4`** (~39 GB) is fetched automatically into `./MiniMax-H3-NF4`. You do **not** need the 498 GB official checkpoint. Everything — training, previews and the VAE decoder — runs from the quantized repo.

---

## ⚡ Usage Guide

### 1. Launch
```cmd
Run_LoRAlab-MiniMaxH3.bat
```
The server starts and your browser opens `http://127.0.0.1:5000`.

### 2. Prepare the dataset

A folder of images with a matching `.txt` caption for each one:

```text
L:\MyDataset\
├── subject01.jpg
├── subject01.txt      ->  "mytrigger, a close-up portrait of a woman ..."
├── subject02.jpg
└── subject02.txt
```

Use the **Dataset Inspector** to check captions, and **+ Trigger All** to inject your trigger word everywhere at once.

### 3. Pre-Cache

1. Enter a **Project Name** and a **Trigger Word**.
2. Pick the dataset folder with **Browse / Explorar**.
3. Set **Resolution** (768×768 recommended) and **Multiple** (32).
4. Click **Start Pre-Cache**.

This loads the Qwen3-VL-32B text encoder once, writes the layer-50 embeddings and VAE latents to disk, and releases everything. It also runs self-tests (RoPE liveness, prompt discrimination, latent statistics) and writes a full `_diagnostics.json`.

Re-running the pre-cache **skips images already cached**, so it is cheap to run again after changing a custom preview prompt.

### 4. Train

Recommended starting point, the configuration that produced the verified result:

| Setting | Value |
| :--- | :--- |
| Total Steps | 1000 |
| Learning Rate | 2e-4 |
| LoRA Rank / Alpha | 16 / 16 |
| Batch Size | 1 |
| Grad Accum | 1 |
| Save Every | 200 |
| Resolution | 768×768 |
| Dataset | 8–20 images |

Click **Start / Resume**. Stop at any time with **Stop Training** — the exact step is saved and resuming continues from it.

First signs of likeness usually appear between steps 400 and 600.

### 5. Previews (optional)

| Setting | Suggested |
| :--- | :--- |
| Preview Every | 100 (0 = off) |
| Caption Mode | First (to compare) or Random (for variety) |
| Preview Steps | 20–30 |
| Preview CFG | 1.0 (the checkpoint is guidance-distilled) |
| Preview Sampler | shift 4.5–6.0 |
| VAE Device | CPU (safe) / CUDA (fast, needs 4.8 GB free) |

For **Custom** prompts: type the prompt, save the **Pre-Cache** JSON and re-run the Pre-Cache once. The trainer has no text encoder by design, so a free prompt must be encoded beforehand.

### 6. Export

Enter a **Final LoRA Filename**, pick your `models/loras` folder, and click **🚀 Send to Models**. The saved file uses the original MiniMax-H3 checkpoint key names and loads directly in ComfyUI.

---

## ⚙️ Settings reference

Most settings live in the `DEFAULTS` dictionary at the top of each script, documented inline in English and Spanish. The GUI exposes the ones you change often. Notable defaults:

| Key | Default | Notes |
| :--- | :--- | :--- |
| `lora_dtype` | `fp32` | fp32 master weights + fp32 Adam state. Do not lower. |
| `optimizer_type` | `adamw` | `adamw8bit` leaves the face soft on H3. |
| `lr_schedule` | `flat` | A cosine decay spends half the movement budget before the LoRA arrives anywhere. |
| `caption_dropout` | `0.05` | Forces identity into the weights, not into caption correlation. |
| `sigma_shift` | `null` | Logit-normal + resolution shift. Do **not** set 12.0 here. |
| `timestep_convention` | `one_minus_sigma` | `t = 1 − σ`, unscaled. |
| `lora_exclude_refiner` | `false` | The token refiner blocks are trained, matching the reference LoRAs. |
| `checkpoint_use_reentrant` | `true` | Required for the block swap to work with `Linear4bit`. |
| `lora_save_raw_copy` | `false` | Diagnostic second file in diffusers key names. Off: it costs 173 MB per save and does nothing in ComfyUI. |

---

## 📁 Project Structure

```text
AcademiaSD_LoRAlab-MiniMaxH3/
├── assets/
│   ├── portada.jpg                 # Web GUI header banner
│   └── logo_128.png                # Browser favicon
├── 1_pre_cache_MiniMaxH3.py        # Text encoder (layer 50) + VAE latent pre-caching
├── 2_train_lora_MiniMaxH3.py       # 33B NF4 LoRA trainer, block swap, previews, export
├── server.py                       # Flask backend
├── trainer_ui.html                 # Web GUI
├── Run_LoRAlab-MiniMaxH3.bat       # 1-click launcher
├── Install_LoRAlab-venv-Minimax.bat
├── Install_Triton&SageAtten220.bat
├── pre_cache_settings.json         # Active pre-cache configuration
├── train_settings.json             # Active training configuration
├── HF_token.json                   # Optional Hugging Face token
├── MiniMax-H3-NF4/                 # Quantized model (auto-downloaded, ~39 GB)
├── cached_data_minimaxh3_<project>/
└── minimaxh3_lora_output_<project>/
    ├── MiniMaxH3_LoRA_step_<N>.safetensors
    ├── MiniMaxH3_FINAL_LoRA.safetensors
    ├── preview_step_<N>.png
    ├── train_log.txt
    └── resume_checkpoint/
```

---

## ⚠️ Beta notes & known limitations

* **Image datasets only.** Video clips and audio training are not implemented. The trainer targets characters and styles.
* **Uses the generic H3 partition.** Not `FL2VA` (first/last frame) or `Ref2VA` (reference-to-video). LoRAs trained here apply to the standard text-to-video path.
* **Previews are not ComfyUI.** The preview sampler is a compact single-frame path; it is a progress indicator, not a quality benchmark. Judge the final LoRA in ComfyUI.
* **The 6 GB and 4 GB profiles are unproven.** See the VRAM table.
* **Windows-focused.** The launchers are `.bat` files; the Python should run elsewhere but is untested.

---

## 💬 Community & Support

- ▶ **YouTube**: [youtube.com/@Academia_SD](https://www.youtube.com/@Academia_SD)
- 𝕏 **X (Twitter)**: [twitter.com/Academia_S_D](https://twitter.com/Academia_S_D)
- 💬 **Discord**: [discord.gg/Syuaduy678](https://discord.gg/Syuaduy678)
- ☕ **Ko-Fi**: [ko-fi.com/academiasd](https://ko-fi.com/academiasd)

---

## 📜 Credits & License

Developed with ❤️ by **AcademiaSD**. Built on PyTorch, Diffusers, PEFT, bitsandbytes and Hugging Face Hub.

Model: **MiniMax-H3** by MiniMaxAI. Quantized weights: **AcademiaSD/MiniMax-H3-NF4**.

Sister projects: [LoRAlab-Krea2](https://github.com/AcademiaSD/AcademiaSD_LoRAlab-Krea2) · [LoRAlab-LTX23](https://github.com/AcademiaSD/AcademiaSD_LoRAlab-LTX23)
