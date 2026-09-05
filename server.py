# -*- coding: utf-8 -*-
"""
server.py — Backend web para AcademiaSD MiniMax-H3 Loralab Trainer
Web backend for AcademiaSD MiniMax-H3 Loralab Trainer
"""
import json
import os
import subprocess
import sys
import threading
import importlib.util
import signal
import shutil
import re
import string
import logging
import webbrowser
from pathlib import Path


# =============================================================================
# DEPENDENCIAS / DEPENDENCIES
# =============================================================================
def ensure_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    if importlib.util.find_spec(import_name) is not None:
        return
    print()
    print("=" * 70)
    print(f"[INFO] Installing missing package / Instalando paquete: '{package_name}'...")
    print(f"[INFO] Python: {sys.executable}")
    print("=" * 70)
    print()
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
    except subprocess.CalledProcessError as exc:
        print(f"\n[ERROR] Failed to install '{package_name}'. Exit code: {exc.returncode}\n")
        raise
    if importlib.util.find_spec(import_name) is None:
        raise RuntimeError(f"Package '{package_name}' installed but import failed.")
    print(f"[OK] '{package_name}' installed successfully / instalado correctamente.")


ensure_package("Flask", "flask")
ensure_package("psutil", "psutil")

from flask import (
    Flask,
    Response,
    jsonify,
    request,
    send_from_directory
)

logging.getLogger('werkzeug').setLevel(logging.ERROR)


# =============================================================================
# CONFIGURACIÓN / CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
UI_FILE = BASE_DIR / "trainer_ui.html"
LOGO_FILE = ASSETS_DIR / "logo.png" if (ASSETS_DIR / "logo.png").exists() else BASE_DIR / "logo.png"
PRECACHE_CONFIG = BASE_DIR / "pre_cache_settings.json"
TRAIN_CONFIG = BASE_DIR / "train_settings.json"
HF_TOKEN_CONFIG = BASE_DIR / "HF_token.json"
CAPTION_SCRIPT = BASE_DIR / "0_caption_MiniMaxH3.py"
PRECACHE_SCRIPT = BASE_DIR / "1_pre_cache_MiniMaxH3.py"
TRAIN_SCRIPT = BASE_DIR / "2_train_lora_MiniMaxH3.py"

app = Flask(__name__)
active_process = None
active_script = None
process_lock = threading.Lock()
output_buffer = []
output_buffer_lock = threading.Lock()


# =============================================================================
# UTILIDADES / UTILS
# =============================================================================
def get_windows_drives():
    drives = []
    if os.name == "nt":
        try:
            from ctypes import windll
            bitmask = windll.kernel32.GetLogicalDrives()
            for letter in string.ascii_uppercase:
                if bitmask & 1:
                    drives.append(f"{letter}:\\")
                bitmask >>= 1
        except Exception:
            pass
    return drives


def read_json_file(path, default=None):
    if default is None:
        default = {}
    try:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else default
    except Exception as exc:
        print(f"[ERROR] Could not read {path}: {exc}")
        return default


def write_json_file(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(temp_path, path)


def resolve_config_path(value, default):
    if value is None or str(value).strip() == "":
        value = default
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()


def get_train_output_dir():
    cfg = read_json_file(TRAIN_CONFIG, {})
    proj = cfg.get("project_name", "").strip()
    if proj:
        return resolve_config_path(f"MiniMaxH3_lora_output_{proj}", "MiniMaxH3_lora_output")
    return resolve_config_path(cfg.get("output_dir"), "MiniMaxH3_lora_output")


# Extensiones del dataset. Se declaran una sola vez: antes cada endpoint tenia
# su propia tupla y bastaba anadir un formato en uno para que los demas lo
# rechazaran sin decir por que.
# Dataset extensions, declared once. Each endpoint used to carry its own tuple,
# so adding a format in one place made the others reject it silently.
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv", ".avi")
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a")
DATASET_EXTS = IMAGE_EXTS + VIDEO_EXTS + AUDIO_EXTS


def kind_of(path):
    """'audio', 'video' o 'image' segun la extension.
    'audio', 'video' or 'image' by extension."""
    ext = path.suffix.lower()
    if ext in AUDIO_EXTS:
        return "audio"
    return "video" if ext in VIDEO_EXTS else "image"


def get_dataset_dir():
    cfg = read_json_file(PRECACHE_CONFIG, {"dataset_path": "./dataset"})
    return resolve_config_path(cfg.get("dataset_path"), "./dataset")


def get_script_for_name(script_name):
    if script_name == "caption":
        return CAPTION_SCRIPT
    if script_name == "precache":
        return PRECACHE_SCRIPT
    if script_name == "train":
        return TRAIN_SCRIPT
    return None


def get_status():
    global active_process
    global active_script
    with process_lock:
        if active_process is None:
            return {"running": False, "script": None, "pid": None}
        if active_process.poll() is not None:
            active_process = None
            active_script = None
            return {"running": False, "script": None, "pid": None}
        return {"running": True, "script": active_script, "pid": active_process.pid}


# =============================================================================
# HUGGINGFACE TOKEN API
# =============================================================================
@app.route("/api/hf-token", methods=["GET"])
def get_hf_token():
    data = read_json_file(HF_TOKEN_CONFIG, {"token": ""})
    return jsonify({"token": data.get("token", "")})


@app.route("/api/save-hf-token", methods=["POST"])
def save_hf_token():
    try:
        req = request.get_json(force=True) or {}
        token = req.get("token", "").strip()
        write_json_file(HF_TOKEN_CONFIG, {"token": token})
        return jsonify({"status": "ok", "file": HF_TOKEN_CONFIG.name})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


# =============================================================================
# REQUESTER NATIVO DE WINDOWS (POPUP SELECCIONAR CARPETA)
# =============================================================================
@app.route("/api/select-folder", methods=["POST"])
def select_folder_native():
    try:
        data = request.get_json(force=True) or {}
        initial_dir = data.get("initial_dir", str(BASE_DIR)).strip()
        if not initial_dir or not os.path.exists(initial_dir):
            initial_dir = str(BASE_DIR)
        selected_path = None
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            chosen = filedialog.askdirectory(
                title="Select Folder / Seleccionar Carpeta",
                initialdir=initial_dir
            )
            root.destroy()
            if chosen:
                selected_path = str(Path(chosen).resolve())
        except Exception:
            pass
        if not selected_path:
            try:
                ps_cmd = (
                    '[System.Reflection.Assembly]::LoadWithPartialName("System.windows.forms") | Out-Null; '
                    '$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; '
                    f'$dialog.SelectedPath = "{initial_dir}"; '
                    'if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $dialog.SelectedPath }'
                )
                creation_flag = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                output = subprocess.check_output(["powershell", "-command", ps_cmd], text=True, errors="ignore", creationflags=creation_flag).strip()
                if output:
                    selected_path = str(Path(output).resolve())
            except Exception:
                pass
        if selected_path:
            return jsonify({"status": "ok", "path": selected_path})
        else:
            return jsonify({"status": "cancelled", "path": None})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


# =============================================================================
# MONITOR VRAM, RAM & TEMPERATURA GPU DE HARDWARE
# =============================================================================
@app.route("/api/system-stats", methods=["GET"])
def get_system_stats():
    import psutil
    ram = psutil.virtual_memory()
    ram_info = {
        "total_gb": round(ram.total / (1024**3), 2),
        "used_gb": round(ram.used / (1024**3), 2),
        "percent": ram.percent
    }
    vram_info = {
        "total_gb": 0.0,
        "used_gb": 0.0,
        "percent": 0.0,
        "temp_c": 0,
        "gpu_name": "N/A"
    }
    try:
        creation_flag = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        cmd = ["nvidia-smi", "--query-gpu=memory.total,memory.used,name,temperature.gpu", "--format=csv,nounits,noheader"]
        output = subprocess.check_output(cmd, text=True, errors="ignore", creationflags=creation_flag).strip().splitlines()[0]
        parts = [p.strip() for p in output.split(",")]
        total_m = float(parts[0])
        used_m = float(parts[1])
        vram_info["gpu_name"] = parts[2]
        vram_info["temp_c"] = int(float(parts[3]))
        vram_info["total_gb"] = round(total_m / 1024.0, 2)
        vram_info["used_gb"] = round(used_m / 1024.0, 2)
        vram_info["percent"] = round((used_m / total_m) * 100, 1) if total_m > 0 else 0.0
    except Exception:
        try:
            import torch
            if torch.cuda.is_available():
                vram_info["gpu_name"] = torch.cuda.get_device_name(0)
                free_b, total_b = torch.cuda.mem_get_info(0)
                used_b = total_b - free_b
                vram_info["total_gb"] = round(total_b / (1024**3), 2)
                vram_info["used_gb"] = round(used_b / (1024**3), 2)
                vram_info["percent"] = round((used_b / total_b) * 100, 1) if total_b > 0 else 0.0
        except Exception:
            pass
    return jsonify({"ram": ram_info, "vram": vram_info})


# =============================================================================
# EXPLORADOR DE DIRECTORIOS (WEB BACKUP)
# =============================================================================
@app.route("/api/browse-dir", methods=["GET"])
def browse_dir():
    requested_path = request.args.get("path", str(BASE_DIR))
    try:
        target = Path(requested_path).resolve()
        if not target.exists() or not target.is_dir():
            target = BASE_DIR
    except Exception:
        target = BASE_DIR
    parent = str(target.parent) if target.parent != target else str(target)
    dirs = []
    try:
        for item in sorted(target.iterdir()):
            if item.is_dir() and not item.name.startswith("."):
                dirs.append({"name": item.name, "path": str(item)})
    except Exception:
        pass
    image_count = 0
    try:
        image_count = sum(1 for f in target.iterdir() if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    except Exception:
        pass
    return jsonify({
        "current_path": str(target),
        "parent_path": parent,
        "directories": dirs,
        "drives": get_windows_drives(),
        "image_count": image_count
    })


# =============================================================================
# EXPORTAR LORA A CARPETA MODELS
# =============================================================================
@app.route("/api/export-lora", methods=["POST"])
def export_lora():
    try:
        data = request.get_json(force=True) or {}
        target_dir_str = data.get("target_dir", "").strip()
        custom_name = data.get("final_name", "").strip()
        if not target_dir_str:
            return jsonify({"status": "error", "error": "Please select a target folder / Por favor selecciona una carpeta de destino."}), 400
        target_dir = Path(target_dir_str).resolve()
        if not target_dir.exists() or not target_dir.is_dir():
            return jsonify({"status": "error", "error": f"Target folder does not exist / Carpeta de destino no existe: {target_dir}"}), 400
        if not custom_name:
            custom_name = "MiniMaxH3_lora.safetensors"
        if not custom_name.lower().endswith(".safetensors"):
            custom_name += ".safetensors"
        output_dir = get_train_output_dir()
        if not output_dir.exists():
            return jsonify({"status": "error", "error": f"Output folder does not exist / Carpeta de salida no existe: {output_dir}"}), 404
        final_file = output_dir / "MiniMaxH3_FINAL_LoRA.safetensors"
        source_file = None
        if final_file.exists():
            source_file = final_file
        else:
            candidates = []
            for f in output_dir.glob("*.safetensors"):
                match = re.search(r"step_(\d+)\.safetensors$", f.name, re.IGNORECASE)
                if match:
                    candidates.append((int(match.group(1)), f))
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                source_file = candidates[0][1]
        if not source_file or not source_file.exists():
            return jsonify({"status": "error", "error": f"No .safetensors files found in / No se encontraron archivos .safetensors en: {output_dir}"}), 404
        dest_file = target_dir / custom_name
        shutil.copy2(source_file, dest_file)
        return jsonify({
            "status": "ok",
            "source": source_file.name,
            "dest": str(dest_file),
            "filename": custom_name
        })
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


# =============================================================================
# UI, ASSETS Y LOGO (FAVICON)
# =============================================================================
@app.route("/")
def index():
    if not UI_FILE.exists():
        return f"File not found / No se encuentra: trainer_ui.html in {BASE_DIR}", 404
    return send_from_directory(str(BASE_DIR), UI_FILE.name)


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    if ASSETS_DIR.exists():
        return send_from_directory(str(ASSETS_DIR), filename)
    return send_from_directory(str(BASE_DIR), filename)


@app.route("/favicon.ico")
@app.route("/logo.png")
def serve_logo():
    if (ASSETS_DIR / "logo.png").exists():
        return send_from_directory(str(ASSETS_DIR), "logo.png")
    if (BASE_DIR / "logo.png").exists():
        return send_from_directory(str(BASE_DIR), "logo.png")
    return "", 404


# =============================================================================
# CONFIGURACIÓN JSON (GUARDADO EN 2 SITIOS)
# =============================================================================
@app.route("/api/settings", methods=["GET"])
def get_settings():
    return jsonify({
        "pre_cache": read_json_file(PRECACHE_CONFIG, {}),
        "train": read_json_file(TRAIN_CONFIG, {}),
        "base_dir": str(BASE_DIR)
    })


@app.route("/api/save-precache", methods=["POST"])
def save_precache():
    try:
        data = request.get_json(force=True)
        if not isinstance(data, dict):
            return jsonify({"status": "error", "error": "JSON object required / Objeto JSON requerido."}), 400
        proj = data.get("project_name", "").strip()
        cache_dir_name = f"cached_data_MiniMaxH3_{proj}" if proj else "cached_data_MiniMaxH3"
        output_dir_name = f"MiniMaxH3_lora_output_{proj}" if proj else "MiniMaxH3_lora_output"
        data["cache_dir"] = f"./{cache_dir_name}"
        write_json_file(PRECACHE_CONFIG, data)
        saved_files = [PRECACHE_CONFIG.name]
        if proj:
            cache_dir_path = resolve_config_path(data["cache_dir"], cache_dir_name)
            cache_dir_path.mkdir(parents=True, exist_ok=True)
            cache_json_file = cache_dir_path / f"pre_cache_settings_{proj}.json"
            write_json_file(cache_json_file, data)
            saved_files.append(f"{cache_dir_name}/{cache_json_file.name}")
        train_cfg = read_json_file(TRAIN_CONFIG, {})
        if "dataset_path" in data:
            train_cfg["dataset_path"] = data["dataset_path"]
        if "project_name" in data:
            train_cfg["project_name"] = data["project_name"]
            train_cfg["cache_dir"] = data["cache_dir"]
            train_cfg["output_dir"] = f"./{output_dir_name}"
        if "trigger_word" in data:
            train_cfg["trigger_word"] = data["trigger_word"]
        write_json_file(TRAIN_CONFIG, train_cfg)
        return jsonify({"status": "ok", "file": saved_files[0], "all_saved": saved_files})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/save-train", methods=["POST"])
def save_train():
    try:
        data = request.get_json(force=True)
        if not isinstance(data, dict):
            return jsonify({"status": "error", "error": "JSON object required / Objeto JSON requerido."}), 400
        proj = data.get("project_name", "").strip()
        cache_dir_name = f"cached_data_MiniMaxH3_{proj}" if proj else "cached_data_MiniMaxH3"
        output_dir_name = f"MiniMaxH3_lora_output_{proj}" if proj else "MiniMaxH3_lora_output"
        data["cache_dir"] = f"./{cache_dir_name}"
        data["output_dir"] = f"./{output_dir_name}"
        write_json_file(TRAIN_CONFIG, data)
        saved_files = [TRAIN_CONFIG.name]
        if proj:
            output_dir_path = resolve_config_path(data["output_dir"], output_dir_name)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            output_json_file = output_dir_path / f"train_settings_{proj}.json"
            write_json_file(output_json_file, data)
            saved_files.append(f"{output_dir_name}/{output_json_file.name}")
        return jsonify({"status": "ok", "file": saved_files[0], "all_saved": saved_files})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


# =============================================================================
# EJECUCIÓN DE SCRIPT & STREAMING
# =============================================================================
@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify(get_status())


@app.route("/api/cache-info", methods=["GET"])
def cache_info():
    """Datos de la cache que necesita el calculo automatico de VRAM.

    El pico de VRAM lo marca la secuencia MAS LARGA del dataset, no la media:
    devuelve el maximo de tokens de texto y la geometria del latente. Si la
    cache no existe todavia, devuelve available=False y la interfaz usa
    max_seq_len como peor caso, que es conservador.

    Data the automatic VRAM calculation needs. The VRAM peak is set by the
    LONGEST sequence in the dataset, so this returns the maximum text-token
    count and the latent geometry. Without a cache it reports available=False
    and the UI falls back to max_seq_len as the worst case.
    """
    cfg = read_json_file(PRECACHE_CONFIG, {})
    proj = str(cfg.get("project_name", "")).strip()
    cache_dir = BASE_DIR / ("cached_data_MiniMaxH3_" + proj if proj else "cached_data_MiniMaxH3")

    out = {"available": False, "cache_dir": str(cache_dir), "images": 0,
           "max_text_tokens": 0, "max_video_tokens": 0}
    if not cache_dir.is_dir():
        return jsonify(out)

    max_text = 0
    images = 0
    for f in cache_dir.glob("*_prompt_structure.json"):
        if f.name.startswith("_"):
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            shape = d.get("items", {}).get("prompt_embeds", {}).get("shape")
            if shape and len(shape) >= 2:
                max_text = max(max_text, int(shape[1]))
                images += 1
        except Exception:
            pass

    # LA RESOLUCION LA MANDA LA CACHE, NO EL FORMULARIO.
    #
    # El entrenamiento usa los latentes ya cacheados, asi que el area que cuenta
    # es la que se uso al construirlos, no la que haya escrita ahora en la
    # interfaz. cache_info.json la guarda; el snapshot pre_cache_settings_*.json
    # que el servidor deja dentro de la carpeta sirve de respaldo.
    # THE CACHE OWNS THE RESOLUTION, not the form: training uses the latents that
    # are already cached, so the area that matters is the one they were built
    # with. cache_info.json records it.
    cached_area = 0
    for candidate in [cache_dir / "cache_info.json"] + sorted(cache_dir.glob("pre_cache_settings_*.json")):
        try:
            d = json.loads(candidate.read_text(encoding="utf-8"))
            area = int(d.get("target_area", 0) or 0)
            if area > 0:
                cached_area = area
                break
        except Exception:
            pass

    # LOS TOKENS SALEN DE latent_shape, NO DE UNA ESTIMACION.
    #
    # Antes se calculaba area/1024, que son los tokens de UN fotograma: valido
    # cuando el proyecto solo entrenaba imagenes, y equivocado por un factor de
    # 37 con un clip de 124. Peor aun en un dataset MIXTO, donde cache_info
    # guarda num_frames=None porque no hay un solo valor, y cualquier formula
    # basada en el area daria el numero de una imagen.
    #
    # Cada muestra guarda su latent_shape [B, C, T, H, W]; con el patch (1,2,2)
    # de H3 los tokens son T x (H/2) x (W/2). Se toma el MAXIMO porque el pico de
    # VRAM lo marca la muestra mas grande, no la media.
    #
    # Tokens come from latent_shape rather than an estimate. The old area/1024 is
    # ONE frame's worth: right when the project only trained images, wrong by a
    # factor of 37 on a 124 frame clip, and worse on a MIXED dataset where
    # cache_info stores num_frames=None because there is no single value. Each
    # sample records its latent_shape [B, C, T, H, W]; with H3's (1,2,2) patch the
    # token count is T x (H/2) x (W/2). The MAXIMUM is taken because the VRAM peak
    # is set by the largest sample, not the average.
    max_video = 0
    max_frames = 0
    max_audio = 0
    for f in cache_dir.glob("*_info.json"):
        if f.name.startswith("_") or f.name == "cache_info.json":
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            shape = d.get("latent_shape")
            if shape and len(shape) >= 5:
                max_video = max(max_video, int(shape[2]) * (int(shape[3]) // 2) * (int(shape[4]) // 2))
            # LAS FILAS DE AUDIO CUENTAN COMO TOKENS.
            #
            # Faltaban aqui, y el error tiraba en la peor direccion posible: en un
            # dataset de solo audio el unico latente de video es el fotograma
            # negro de relleno, que son 1 token, asi que la interfaz veia una
            # secuencia de 105 tokens donde el entrenador contaba 415. Con eso
            # repartia cuatro bloques residentes de mas, escribia un budget de
            # 15,22 GB en una tarjeta de 15,92, y el entrenamiento saturaba la
            # VRAM y se desplomaba de velocidad -- sin ningun mensaje, porque
            # nadie estaba mintiendo: los dos calculos eran correctos sobre datos
            # distintos.
            #
            # AUDIO ROWS COUNT AS TOKENS. They were missing here, and the error
            # pulled the worst way: on an audio-only dataset the only video latent
            # is the black filler frame -- one token -- so the UI saw a 105 token
            # sequence where the trainer counted 415. That handed out four
            # resident blocks too many and wrote a 15.22 GB budget on a 15.92 GB
            # card; training saturated and collapsed in speed, silently, because
            # neither calculation was wrong -- they were right about different
            # data.
            forma_audio = d.get("audio_latent_shape")
            if forma_audio and len(forma_audio) >= 3:
                max_audio = max(max_audio, int(forma_audio[2]))
            nf = int(d.get("num_frames", 0) or 0)
            if nf > max_frames:
                max_frames = nf
        except Exception:
            pass

    # Sin latent_shape (caches antiguas) se vuelve a la estimacion, que al menos
    # no es cero. / Without latent_shape (older caches) fall back to the estimate.
    if max_video == 0 and cached_area > 0:
        max_video = int(cached_area / 1024)

    if images:
        out.update({"available": True, "images": images,
                    "max_text_tokens": max_text, "max_video_tokens": max_video,
                    # Un relleno de silencio es 1 fila; la pista mas corta que la
                    # rejilla admite son 74. Por encima de 4 hay audio de verdad.
                    # A silence placeholder is 1 row; the shortest real track is 74.
                    "max_audio_rows": max_audio,
                    "cached_num_frames": max_frames,
                    "cached_target_area": cached_area,
                    "form_target_area": int(cfg.get("target_area", 0) or 0)})
    return jsonify(out)


@app.route("/api/checkpoint-info", methods=["GET"])
def checkpoint_info():
    output_dir = get_train_output_dir()
    step_file = output_dir / "current_step.txt"
    resume_dir = output_dir / "resume_checkpoint"
    has_checkpoint = step_file.exists() and resume_dir.exists()
    current_step = 0
    if step_file.exists():
        try:
            current_step = int(step_file.read_text(encoding="utf-8").strip())
        except Exception:
            pass
    return jsonify({
        "has_checkpoint": has_checkpoint,
        "current_step": current_step,
        "output_dir": str(output_dir)
    })


@app.route("/api/output", methods=["GET"])
def api_output():
    global output_buffer
    with output_buffer_lock:
        if not output_buffer:
            if active_process is None or active_process.poll() is not None:
                code = active_process.returncode if active_process else 0
                return jsonify({"text": "", "done": True, "code": code})
            return jsonify({"text": "", "done": False})
        text = "".join(output_buffer)
        output_buffer = []
        return jsonify({"text": text, "replace": False, "done": False})


@app.route("/api/run", methods=["POST"])
def run_script():
    global active_process
    global active_script

    try:
        data = request.get_json(force=True) or {}
        script_name = data.get("script")

        script_path = get_script_for_name(script_name)

        if script_path is None or not script_path.exists():
            return jsonify({
                "status": "error",
                "error": f"Script not found / Script no encontrado: {script_name}"
            }), 404

        with process_lock:
            if active_process is not None and active_process.poll() is None:
                return jsonify({
                    "status": "error",
                    "error": f"Process already running / Proceso en ejecucion: {active_script}"
                }), 409

            command = [
                sys.executable,
                "-u",
                str(script_path)
            ]

            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0

            process = subprocess.Popen(
                command,
                cwd=str(BASE_DIR),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creation_flags
            )

            active_process = process
            active_script = script_name

        def stream():
            global active_process
            global active_script

            yield f"data: {json.dumps({'type': 'start', 'script': script_name}, ensure_ascii=False)}\n\n"

            try:
                if process.stdout is not None:
                    buffer = ""

                    while True:
                        char = process.stdout.read(1)

                        if not char:
                            if buffer:
                                yield (
                                    "data: "
                                    + json.dumps(
                                        {
                                            "type": "output",
                                            "text": buffer,
                                            "replace": False
                                        },
                                        ensure_ascii=False
                                    )
                                    + "\n\n"
                                )
                            break

                        if char == "\r":
                            if buffer:
                                yield (
                                    "data: "
                                    + json.dumps(
                                        {
                                            "type": "output",
                                            "text": buffer,
                                            "replace": True
                                        },
                                        ensure_ascii=False
                                    )
                                    + "\n\n"
                                )
                                buffer = ""

                        elif char == "\n":
                            if buffer:
                                yield (
                                    "data: "
                                    + json.dumps(
                                        {
                                            "type": "output",
                                            "text": buffer,
                                            "replace": False
                                        },
                                        ensure_ascii=False
                                    )
                                    + "\n\n"
                                )
                                buffer = ""

                        else:
                            buffer += char

                return_code = process.wait()

                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "done",
                            "script": script_name,
                            "code": return_code
                        },
                        ensure_ascii=False
                    )
                    + "\n\n"
                )

            except GeneratorExit:
                pass

            finally:
                with process_lock:
                    if active_process is process:
                        active_process = None
                        active_script = None

        return Response(
            stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as exc:
        with process_lock:
            active_process = None
            active_script = None

        return jsonify({
            "status": "error",
            "error": str(exc)
        }), 500

@app.route("/api/stop", methods=["POST"])
def stop_script():
    global active_process
    global active_script
    with process_lock:
        process = active_process
        script = active_script
    if process is None or process.poll() is not None:
        with process_lock:
            active_process = None
            active_script = None
        return jsonify({"status": "not_running"})
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGINT)
        return jsonify({"status": "terminating", "script": script})
    except Exception as exc:
        try:
            process.terminate()
        except Exception:
            pass
        return jsonify({"status": "terminated", "script": script})


# =============================================================================
# PREVIEWS & DATASET API
# =============================================================================
@app.route("/api/previews", methods=["GET"])
def get_previews():
    output_dir = get_train_output_dir()
    previews = []
    if output_dir.is_dir():
        for file_path in output_dir.iterdir():
            if file_path.is_file() and file_path.name.startswith("preview_step_") and file_path.suffix.lower() == ".png":
                previews.append(file_path)
    previews.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return jsonify({"output_dir": str(output_dir), "previews": [p.name for p in previews[:50]]})


@app.route("/api/preview/<path:filename>")
def serve_preview(filename):
    output_dir = get_train_output_dir()
    requested = (output_dir / filename).resolve()
    try:
        requested.relative_to(output_dir.resolve())
    except ValueError:
        return "", 403
    if not requested.is_file():
        return "", 404
    return send_from_directory(str(output_dir), requested.name)


@app.route("/api/dataset-info", methods=["GET"])
def dataset_info():
    dataset_dir = get_dataset_dir()
    images = []
    if dataset_dir.is_dir():
        candidatos = [f for f in sorted(dataset_dir.iterdir())
                      if f.is_file() and f.suffix.lower() in DATASET_EXTS]
        # Saber si un clip trae pista de audio cuesta un ffprobe. Aqui habia un
        # limite: por encima de 60 videos se devolvia None y la interfaz no
        # pintaba nada. La intencion era buena -- no tardar diez segundos en
        # listar la carpeta -- pero el efecto era el peor posible: el indicador
        # desaparecia justo en los datasets grandes, que son donde no puedes
        # revisar clip por clip, y su ausencia no se distingue de "todos tienen
        # audio". Trocear una pelicula da 100 clips y de golpe no se veia ninguno.
        #
        # En vez de subir el numero, se abarata la pregunta: el resultado se
        # cachea por (ruta, mtime, tamano), asi que solo se paga por fichero
        # nuevo o modificado, y los que faltan se sondean en paralelo. Un
        # ffprobe son ~40 ms; cien en serie son cuatro segundos y con ocho hilos
        # medio. Los listados siguientes no cuestan nada.
        #
        # There used to be a limit here: past 60 videos this returned None and
        # the UI drew nothing. The intent was right -- not spending ten seconds
        # listing a folder -- but the effect was the worst possible: the badge
        # vanished precisely on the large datasets where you cannot check clip by
        # clip, and its absence is indistinguishable from "they all have audio".
        # Splitting a film gives 100 clips and suddenly none of them showed one.
        # Rather than raising the number, the question is made cheap: results are
        # cached by (path, mtime, size) so only new or changed files cost
        # anything, and the misses are probed in parallel. One ffprobe is ~40 ms;
        # a hundred in series is four seconds, with eight threads half of one.
        pendientes = [f for f in candidatos
                      if kind_of(f) == "video" and _audio_probe_key(f) not in _AUDIO_PROBE_CACHE]
        if pendientes:
            try:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=8) as pool:
                    list(pool.map(_has_audio, pendientes))
            except Exception:
                pass          # si falla, cada uno se sondea abajo de uno en uno
        for file_path in candidatos:
            if True:
                txt_path = file_path.with_suffix(".txt")
                caption = ""
                if txt_path.exists():
                    try:
                        caption = txt_path.read_text(encoding="utf-8").strip()
                    except Exception:
                        pass
                images.append({
                    "file": file_path.name,
                    "kind": kind_of(file_path),
                    "has_audio": (_has_audio(file_path)
                                  if kind_of(file_path) == "video" else None),
                    "has_txt": txt_path.exists(),
                    "caption": caption,
                })
    return jsonify({
        "path": str(dataset_dir),
        "image_count": len(images),
        "video_count": sum(1 for it in images if it["kind"] == "video"),
        "caption_count": sum(1 for img in images if img["has_txt"]),
        "images": images[:500]
    })


@app.route("/api/dataset-image/<path:filename>")
def serve_dataset_image(filename):
    dataset_dir = get_dataset_dir()
    try:
        requested = (dataset_dir / filename).resolve()
        requested.relative_to(dataset_dir.resolve())
    except Exception:
        return "", 403
    if not requested.is_file() or requested.suffix.lower() not in DATASET_EXTS:
        return "", 404
    # conditional=True activa las peticiones por rango (HTTP 206), que es lo que
    # necesita <video> para poder buscar dentro del clip sin descargarlo entero.
    # conditional=True enables range requests (HTTP 206), which <video> needs to
    # seek inside the clip without downloading all of it.
    return send_from_directory(str(dataset_dir), requested.name, conditional=True)


@app.route("/api/save-caption", methods=["POST"])
def save_caption():
    try:
        data = request.get_json(force=True)
        filename = data.get("filename")
        caption = data.get("caption", "").strip()
        dataset_dir = get_dataset_dir()
        img_path = (dataset_dir / filename).resolve()
        img_path.relative_to(dataset_dir.resolve())
        txt_path = img_path.with_suffix(".txt")
        txt_path.write_text(caption, encoding="utf-8")
        return jsonify({"status": "ok", "file": txt_path.name})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/clear-captions", methods=["POST"])
def clear_captions():
    """Borra el .txt de todas las muestras del dataset.

    Se borra el FICHERO en vez de vaciarlo. Un .txt vacio cuenta como caption
    presente: el Dataset Manager lo pintaria con el punto verde y la pre-cache lo
    aceptaria como descripcion valida, entrenando con texto en blanco sin que
    nadie se entere. Borrarlo deja el punto rojo, que es la verdad.

    Deletes each sample's .txt rather than emptying it. An empty .txt counts as a
    caption present: the Dataset Manager would draw the green dot and the
    pre-cache would take it as a valid description, training on blank text
    unnoticed. Deleting leaves the red dot, which is the truth.
    """
    try:
        dataset_dir = get_dataset_dir()
        if not dataset_dir.is_dir():
            return jsonify({"status": "ok", "removed": 0})

        borrados, errores = 0, []
        for file_path in sorted(dataset_dir.iterdir()):
            if not (file_path.is_file() and file_path.suffix.lower() in DATASET_EXTS):
                continue
            txt = file_path.with_suffix(".txt")
            if not txt.exists():
                continue
            try:
                txt.unlink()
                borrados += 1
            except Exception as exc:
                errores.append("{}: {}".format(txt.name, exc))

        return jsonify({"status": "ok" if not errores else "partial",
                        "removed": borrados, "errors": errores})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/batch-caption", methods=["POST"])
def batch_caption():
    """Escribe la palabra trigger, o una descripcion comun, en todos los captions.

    Dos modos en el mismo sitio porque son la misma operacion sobre los mismos
    ficheros. Con `common` se REEMPLAZA el caption entero; sin el, solo se
    antepone el trigger a lo que ya hubiera.

    Reemplazar tiene sentido cuando todas las muestras describen lo mismo, que es
    el caso normal al trocear: 69 segmentos de la misma voz o del mismo martillo
    no necesitan 69 textos distintos, y escribirlos a mano seria la parte mas
    lenta de preparar el dataset. Al captioner de imagen eso no le vale -- cada
    foto es distinta -- pero al audio si, porque Qwen3-VL no oye y no hay
    generador automatico para esas tomas.

    Writes the trigger word, or a common description, into every caption. Both
    modes live here because they are the same operation on the same files: with
    `common` the whole caption is REPLACED, without it the trigger is merely
    prepended to whatever was there. Replacing makes sense when every sample
    describes the same thing, which is the normal case after splitting -- 69
    segments of one voice do not need 69 different texts, and typing them would
    be the slowest part of preparing the dataset. It does not suit the image
    captioner, where every photo differs, but it does suit audio: Qwen3-VL cannot
    hear, so there is no automatic captioner for those takes.
    """
    try:
        data = request.get_json(force=True)
        trigger = data.get("trigger_word", "").strip()
        comun = str(data.get("common", "") or "").strip()
        modo = str(data.get("mode", "append") or "append").strip().lower()
        dataset_dir = get_dataset_dir()
        count = 0

        if not dataset_dir.is_dir() or not (trigger or comun):
            return jsonify({"status": "ok", "updated_count": 0})

        for file_path in sorted(dataset_dir.iterdir()):
            if not (file_path.is_file() and file_path.suffix.lower() in DATASET_EXTS):
                continue
            txt_path = file_path.with_suffix(".txt")

            actual = ""
            if txt_path.exists():
                actual = txt_path.read_text(encoding="utf-8").strip()

            if not comun:
                text = actual
            elif modo == "remove":
                # QUITAR un texto de todos los captions.
                #
                # Anteponer el trigger no tiene deshacer: si se pulsa Apply con la
                # palabra equivocada, queda en los cien ficheros y la unica salida
                # era borrarlos todos y volver a generarlos. Esto lo quita sin
                # tocar el resto del caption.
                #
                # Se limpia tambien la puntuacion que rodeaba al texto, o quedaria
                # " , una coma suelta al principio" o dos espacios en medio.
                #
                # REMOVING a text from every caption. Prepending the trigger has no
                # undo: pressing Apply with the wrong word leaves it in a hundred
                # files, and the only way out was deleting them all and generating
                # again. This takes it out without touching the rest. The
                # punctuation around it is cleaned too, or a stray leading comma or
                # a double space would be left behind.
                text = re.sub(re.escape(comun), "", actual, flags=re.IGNORECASE)
                text = re.sub(r"\s{2,}", " ", text)
                text = re.sub(r"\s+([,.;:!?])", r"\1", text)
                text = re.sub(r"^[\s,;:.]+", "", text)
                text = re.sub(r"[,;:]\s*$", "", text).strip()
            elif modo == "replace":
                text = comun
            elif comun.lower() in actual.lower():
                # Idempotente: pulsar dos veces no debe duplicar la frase. Es la
                # diferencia entre un boton que se puede volver a pulsar sin
                # pensar y uno que hay que usar con cuidado.
                # Idempotent: pressing twice must not duplicate the sentence. That
                # is the difference between a button you can press again without
                # thinking and one you have to be careful with.
                text = actual
            elif actual:
                # Se anade al FINAL. Un caption de video describe lo que se ve; la
                # frase de audio describe lo que se oye, y va detras porque ese es
                # el orden de la secuencia empaquetada [texto | video | audio].
                # Se cierra la frase anterior si no traia puntuacion, o las dos
                # descripciones quedarian pegadas en una sola oracion falsa.
                # Appended at the END. A video caption describes what is seen; the
                # audio sentence describes what is heard, and follows because that
                # is the packed sequence's order. The previous sentence is closed
                # if it carried no punctuation, or the two descriptions would run
                # together into one false sentence.
                base = actual if actual[-1] in ".!?,;:" else actual + "."
                text = "{} {}".format(base, comun)
            else:
                text = comun

            # El trigger va SIEMPRE al principio y solo si falta. Anteponerlo dos
            # veces seria enseñarle al modelo que la palabra se repite.
            # The trigger always goes first and only when missing: prepending it
            # twice would teach the model the word comes in pairs.
            # En modo remove no se antepone nada: se acaba de pedir quitar una
            # palabra, y volver a meter el trigger en la misma pasada dejaria al
            # usuario sin saber que ha pasado.
            # Nothing is prepended in remove mode: a word was just asked to be
            # taken out, and re-inserting the trigger in the same pass would leave
            # the user unsure what happened.
            if modo != "remove" and trigger and trigger.lower() not in text.lower():
                text = "{}, {}".format(trigger, text).strip(", ")

            txt_path.write_text(text, encoding="utf-8")
            count += 1

        return jsonify({"status": "ok", "updated_count": count})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


def _get_precache_dir():
    cfg = read_json_file(PRECACHE_CONFIG, {})
    proj = str(cfg.get("project_name", "")).strip()
    if proj:
        return resolve_config_path("cached_data_MiniMaxH3_{}".format(proj), "./cached_data_MiniMaxH3")
    return resolve_config_path(cfg.get("cache_dir"), "./cached_data_MiniMaxH3")


def _wipe_dir(path):
    """Vacia una carpeta sin borrarla. Devuelve (borrados, errores).

    Se borra el CONTENIDO y no la carpeta en si: las rutas estan guardadas en
    los JSON de configuracion y en la GUI, y una carpeta que desaparece obliga
    al usuario a volver a elegirla.

    Empties a folder without removing it. The CONTENT goes, not the folder
    itself: the paths live in the config JSONs and in the GUI, and a folder that
    vanishes forces the user to pick it again.
    """
    removed, errors = 0, []
    if not path.is_dir():
        return removed, errors
    for entry in path.iterdir():
        try:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
            removed += 1
        except Exception as exc:
            errors.append("{}: {}".format(entry.name, exc))
    return removed, errors


# Geometria temporal de MiniMax-H3, tomada del propio VAE (AutoencoderKLMiniMaxH3):
#
#   "The temporal geometry is fixed by clip_length (17 pixel frames per encoder
#    chunk) and token_drop (3 trailing latent frames dropped per encode):
#    17 * n + 5 pixel frames map to 5 * n + 2 latent frames."
#
# O sea que el numero de fotogramas NO es libre: tiene que ser 17n+5. Un clip de
# 81 no vale (n saldria 4,47) y hay que recortarlo a 73. Las duraciones de audio
# que admite el modelo son exactamente estas mismas cuentas a 24 fps, porque
# audio y video comparten la rejilla dentro de la secuencia empaquetada.
#
# MiniMax-H3's temporal geometry, taken from the VAE itself: the frame count is
# not free, it must be 17n+5. An 81-frame clip is invalid (n would be 4.47) and
# has to be trimmed to 73. The audio durations the model accepts are these same
# counts at 24 fps, because audio and video share the grid inside the packed
# sequence.
H3_FPS = 24.0
H3_CLIP_LENGTH = 17
H3_BASE_FRAMES = 5
H3_SPATIAL_MULTIPLE = 32


def h3_valid_frames(count, target=0):
    """Fotogramas a conservar de un clip de `count`.

    Con `target` se pide un numero concreto (el campo Frames de la pre-cache) y
    se usa si el clip da para tanto; si no, se cae al mayor 17n+5 que quepa. Sin
    `target`, el mayor posible.

    Frames to keep from a clip of `count`. With `target` a specific count is
    requested (the pre-cache's Frames field) and used if the clip is long
    enough; otherwise it falls back to the largest 17n+5 that fits.
    """
    if count < H3_BASE_FRAMES:
        return None
    mayor = H3_CLIP_LENGTH * ((count - H3_BASE_FRAMES) // H3_CLIP_LENGTH) + H3_BASE_FRAMES
    if target and target >= H3_BASE_FRAMES:
        # El objetivo tambien tiene que caer en la rejilla: si llega un valor
        # raro se baja al 17n+5 inmediatamente inferior en vez de aceptarlo.
        # The target must land on the grid too: an odd value is lowered to the
        # nearest 17n+5 below rather than accepted.
        objetivo = H3_CLIP_LENGTH * ((target - H3_BASE_FRAMES) // H3_CLIP_LENGTH) + H3_BASE_FRAMES
        return min(objetivo, mayor)
    return mayor


def _ffbin(name):
    """Localiza ffmpeg/ffprobe. / Locates ffmpeg/ffprobe."""
    found = shutil.which(name) or shutil.which(name + ".exe")
    return found


# Resultado de la sonda, cacheado por (ruta, mtime, tamano). La clave lleva el
# mtime a proposito: si el fichero cambia -- lo vuelves a trocear, le pegas una
# pista -- la clave cambia con el y no hay que invalidar nada a mano.
# Probe results cached by (path, mtime, size). The mtime is in the key on
# purpose: if the file changes -- re-split, audio added -- the key changes with
# it and nothing has to be invalidated by hand.
_AUDIO_PROBE_CACHE = {}


def _audio_probe_key(path):
    try:
        st = os.stat(str(path))
        return (str(path), st.st_mtime_ns, st.st_size)
    except Exception:
        return (str(path), 0, 0)


def _has_audio(path):
    """True si el fichero trae al menos un stream de audio.

    Se pregunta antes de convertir para poder decirlo en el parte: un clip mudo
    entrena solo la imagen, y conviene que eso se vea en vez de descubrirlo
    cuando el audio generado sale en silencio.

    True if the file carries at least one audio stream. Asked before converting
    so the report can say it: a mute clip trains picture only, and that is worth
    seeing up front rather than discovering it when the generated audio is
    silent.
    """
    clave = _audio_probe_key(path)
    if clave in _AUDIO_PROBE_CACHE:
        return _AUDIO_PROBE_CACHE[clave]

    probe = _ffbin("ffprobe")
    if not probe:
        # Sin ffprobe no se sabe, y NO se cachea: instalarlo despues debe bastar
        # para que el indicador aparezca, sin reiniciar el servidor.
        # Without ffprobe there is no answer, and it is NOT cached: installing it
        # afterwards should be enough, with no server restart.
        return False
    try:
        out = subprocess.run(
            [probe, "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        resultado = bool(out)
    except Exception:
        return False

    # Tope tonto pero suficiente: un dataset no llega a 5.000 clips, y si llega,
    # vaciar y volver a llenar cuesta lo mismo que la primera vez.
    # Crude cap: a dataset does not reach 5,000 clips, and if it does, emptying
    # and refilling costs what the first fill did.
    if len(_AUDIO_PROBE_CACHE) > 5000:
        _AUDIO_PROBE_CACHE.clear()
    _AUDIO_PROBE_CACHE[clave] = resultado
    return resultado


def _video_info(path):
    """(fps, frames, ancho, alto) del clip, o None si no se puede leer.

    Se cuentan los fotogramas de verdad (-count_frames) en vez de fiarse de
    nb_frames del contenedor, que en muchos mp4 viene vacio o mal.
    Frames are actually counted instead of trusting the container's nb_frames,
    which is often missing or wrong in mp4.
    """
    probe = _ffbin("ffprobe")
    if not probe:
        return None
    try:
        out = subprocess.run(
            [probe, "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=r_frame_rate,nb_read_frames,width,height",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=120,
        ).stdout.strip()
        w, h, rate, frames = out.split(",")
        num, _, den = rate.partition("/")
        return float(num) / float(den or 1), int(frames), int(w), int(h)
    except Exception:
        return None


def _video_fps(path):
    info = _video_info(path)
    return info[0] if info else None


def h3_nearest_frames(ideal):
    """El 17n+5 mas cercano a `ideal`, nunca por debajo de 5.

    Se redondea al mas cercano y no hacia abajo a proposito: hacia abajo se
    pierde el final del clip, y en un clip de efecto el final es el remate. Un
    2% de camara lenta no se nota; que la cara no termine de abrirse, si.

    The nearest 17n+5 to `ideal`, never below 5. Rounded to nearest rather than
    down on purpose: rounding down loses the end of the clip, and in an effect
    clip the end is the payoff. A 2% slow-motion goes unnoticed; a face that
    never finishes opening does not.
    """
    if ideal <= H3_BASE_FRAMES:
        return H3_BASE_FRAMES
    k = int(round((float(ideal) - H3_BASE_FRAMES) / H3_CLIP_LENGTH))
    return max(H3_BASE_FRAMES, H3_CLIP_LENGTH * k + H3_BASE_FRAMES)


def _audio_silences(path, ffmpeg, sr=32000, hop=320, umbral_db=-38.0, min_ms=120):
    """Centros de las zonas silenciosas, en segundos.

    Cortar una toma de voz por un reloj parte palabras. Cortarla por un silencio
    la parte por donde ya estaba partida: una pausa del habla. Sobre 13 tomas
    reales, 65 de 67 cortes cayeron en un silencio.

    Centres of the silent stretches, in seconds. Cutting a voice take by the
    clock splits words; cutting it at a silence splits it where it was already
    split -- at a pause. On 13 real takes, 65 of 67 cuts landed on a silence.
    """
    import numpy as np
    try:
        raw = subprocess.run(
            [ffmpeg, "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1",
             "-ar", str(sr), "-"], capture_output=True, timeout=600).stdout
        a = np.frombuffer(raw, dtype=np.float32)
    except Exception:
        return []
    if a.size < hop:
        return []

    n = a.size // hop
    e = np.sqrt((a[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-12)
    mudo = 20.0 * np.log10(e + 1e-9) < umbral_db

    zonas, ini = [], None
    for i, m in enumerate(mudo):
        if m and ini is None:
            ini = i
        elif not m and ini is not None:
            if (i - ini) * hop / float(sr) * 1000.0 >= min_ms:
                zonas.append(((ini + i) // 2) * hop / float(sr))
            ini = None
    if ini is not None and (n - ini) * hop / float(sr) * 1000.0 >= min_ms:
        zonas.append(((ini + n) // 2) * hop / float(sr))
    return zonas


def _split_points(duracion, ventana, minimo, silencios):
    """Los cortes [(inicio, fin), ...] en segundos.

    Se elige el silencio MAS TARDIO que cabe en la ventana, no el mas cercano al
    objetivo: asi cada toma se aprovecha al maximo y salen menos trozos. Cuando
    no hay ningun silencio en la ventana se corta en seco y el llamante lo
    contabiliza, porque un corte a media palabra hay que oirlo antes de entrenar
    con el.

    Cut points in seconds. The LATEST silence that fits the window is chosen
    rather than the one nearest the target, so each take is used to the full and
    fewer pieces come out. With no silence in the window it cuts hard and the
    caller counts it, because a cut mid-word has to be heard before training.
    """
    cortes, pos, duros = [], 0.0, 0
    while duracion - pos > ventana + 1e-6:
        dentro = [x for x in silencios if pos + minimo <= x <= pos + ventana]
        if dentro:
            corte = max(dentro)
        else:
            corte = pos + ventana
            duros += 1
        cortes.append((pos, corte))
        pos = corte
    if duracion - pos >= minimo * 0.25:
        cortes.append((pos, duracion))
    return cortes, duros


@app.route("/api/extract-vocals", methods=["POST"])
def extract_vocals():
    """Separa la voz de la musica y los efectos en todas las muestras con audio.

    CUANDO USARLO Y CUANDO NO.
    Un dataset sacado de material real -- una pelicula, una entrevista -- trae
    musica y efectos encima del dialogo, y un LoRA entrenado con eso clona la
    mezcla, no la voz. Ahi separar es imprescindible. Sobre una grabacion que ya
    viene limpia, EMPEORA: la separacion es destructiva, deja resonancias en las
    eses y en los transitorios, y el LoRA aprende tambien ese caracter.
    Tampoco quita reverberacion: separa fuentes, no arregla la sala.

    Va ANTES de Split, no despues. Separado el material entero, los cortes salen
    de una sola pasada del modelo; al reves, cada trozo se separaria por su
    cuenta y los artefactos variarian entre tomas del mismo dataset.

    El modelo (~600 MB) se descarga la primera vez que se pulsa, como hace el
    captioner con Qwen3-VL. Las tres dependencias se comprueban ANTES de
    descargar nada: no tiene sentido bajar 600 MB para fallar despues en un
    import.

    Separates voice from music and effects. Essential on material taken from a
    film or an interview, where a LoRA would otherwise clone the mix; harmful on
    already-clean recordings, since separation is destructive and leaves
    artefacts on sibilants and transients that the LoRA learns too. It does not
    remove reverb: it separates sources, not rooms. It runs BEFORE Split, so the
    cuts come from a single pass of the model rather than one pass per piece
    with artefacts varying between takes. The model (~600 MB) downloads on first
    use, like the captioner's Qwen3-VL; the three dependencies are checked BEFORE
    downloading anything.
    """
    try:
        if get_status().get("running"):
            return jsonify({"status": "error",
                            "error": "A process is running. Stop it first. / "
                                     "Hay un proceso en marcha. Detenlo primero."}), 409

        import sys
        sys.path.insert(0, str(BASE_DIR))
        from melband import separator as sep

        faltan = sep.dependencias_que_faltan()
        if faltan:
            return jsonify({"status": "error",
                            "error": "Missing packages: {}. Install them in the venv with "
                                     "pip install {} / Faltan paquetes: {}. Instalalos en el "
                                     "venv con pip install {}"
                                     .format(", ".join(faltan), " ".join(faltan),
                                             ", ".join(faltan), " ".join(faltan))}), 400

        ffmpeg = _ffbin("ffmpeg")
        if not ffmpeg:
            return jsonify({"status": "error",
                            "error": "ffmpeg not found in PATH. / No se encuentra ffmpeg."}), 400

        dataset_dir = get_dataset_dir()
        if not dataset_dir.is_dir():
            return jsonify({"status": "error", "error": "No dataset folder / No hay dataset"}), 404

        muestras = [f for f in sorted(dataset_dir.iterdir())
                    if f.is_file() and f.suffix.lower() in AUDIO_EXTS]
        if not muestras:
            return jsonify({"status": "ok", "processed": 0,
                            "message": "No audio files / No hay ficheros de audio"})

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        modelo = sep.cargar(str(BASE_DIR / "MelBandRoFormer"), device, log=print)
        if modelo is None:
            return jsonify({"status": "error",
                            "error": "Could not load the separation model. / "
                                     "No se pudo cargar el modelo de separacion."}), 500

        backup = dataset_dir / "_originals"
        hechos, errores = [], []
        for m in muestras:
            try:
                pcm = sep.leer_pcm(m, ffmpeg)
                if pcm.size == 0:
                    errores.append("{}: empty / vacio".format(m.name))
                    continue
                voz = sep.separar_voz(modelo, pcm, device)

                backup.mkdir(exist_ok=True)
                tmp = backup / ("__vocals__" + m.name)
                sep.escribir_pcm(voz, tmp, ffmpeg)
                shutil.move(str(m), str(backup / m.name))
                shutil.move(str(tmp), str(m))
                hechos.append("{} ({:.1f}s)".format(m.name, pcm.shape[1] / sep.SR))
            except Exception as exc:
                errores.append("{}: {}".format(m.name, exc))

        del modelo
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        return jsonify({"status": "ok" if not errores else "partial",
                        "processed": len(hechos), "details": hechos, "errors": errores,
                        "backup": str(backup) if hechos else None})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/split-samples", methods=["POST"])
def split_samples():
    """Trocea las muestras largas en tomas de la duracion configurada.

    UNA GRABACION LARGA NO ES UNA MUESTRA MEJOR, ES UNA MUESTRA CARA.
    Medido sobre 252 s de voz en 13 ficheros: enteros son 13 muestras de hasta
    2.624 filas de audio, 15 bloques residentes y 10,8 s/it. Troceados a 5,167 s
    son 67 muestras de 414 filas, 27 bloques y 3,77 s/it. Casi tres veces mas
    rapido, cinco veces mas muestras, y dentro del rango de duracion que H3
    documenta (0,917 a 5,167 s) en vez de seis veces por encima.

    El audio se corta por silencios; el video por fotograma exacto, arrastrando
    su pista si la tiene. Los originales van a _originals, que ni el pre-cache ni
    el Dataset Manager recorren.

    A long recording is not a better sample, it is an expensive one. Measured on
    252 s of speech in 13 files: whole, that is 13 samples of up to 2,624 audio
    rows, 15 resident blocks and 10.8 s/it; split at 5.167 s it is 67 samples of
    414 rows, 27 blocks and 3.77 s/it -- nearly three times faster, five times
    the samples, and inside the duration range H3 documents rather than six times
    beyond it. Audio is cut at silences, video on the exact frame carrying its
    track along. Originals go to _originals.
    """
    try:
        data = request.get_json(force=True) or {}
        frames = int(data.get("frames", 0) or 0)
        if not frames:
            frames = int(read_json_file(PRECACHE_CONFIG, {}).get("num_frames", 124) or 124)
        frames = max(H3_BASE_FRAMES, frames)
        ventana = frames / 24.0
        minimo = max(H3_BASE_FRAMES, H3_CLIP_LENGTH + H3_BASE_FRAMES) / 24.0

        if get_status().get("running"):
            return jsonify({"status": "error",
                            "error": "A process is running. Stop it first. / "
                                     "Hay un proceso en marcha. Detenlo primero."}), 409

        ffmpeg = _ffbin("ffmpeg")
        probe = _ffbin("ffprobe")
        if not ffmpeg or not probe:
            return jsonify({"status": "error",
                            "error": "ffmpeg/ffprobe not found in PATH. / "
                                     "No se encuentran ffmpeg/ffprobe en el PATH."}), 400

        dataset_dir = get_dataset_dir()
        if not dataset_dir.is_dir():
            return jsonify({"status": "error", "error": "No dataset folder / No hay dataset"}), 404

        muestras = sorted(f for f in dataset_dir.iterdir()
                          if f.is_file() and f.suffix.lower() in DATASET_EXTS
                          and f.suffix.lower() not in IMAGE_EXTS)
        backup = dataset_dir / "_originals"
        troceadas, saltadas, producidas, errores, avisos, detalles = 0, 0, 0, [], [], []

        for m in muestras:
            es_audio = m.suffix.lower() in AUDIO_EXTS
            try:
                dur = float(subprocess.run(
                    [probe, "-v", "error", "-show_entries", "format=duration",
                     "-of", "csv=p=0", str(m)],
                    capture_output=True, text=True, timeout=120).stdout.strip())
            except Exception:
                errores.append("{}: could not read the duration / no se pudo leer la duracion"
                               .format(m.name))
                continue

            if dur <= ventana + 1e-6:
                saltadas += 1
                continue

            sil = _audio_silences(m, ffmpeg) if es_audio else []
            cortes, duros = _split_points(dur, ventana, minimo, sil)
            if duros:
                avisos.append("{}: {} cut(s) fell mid-content, no silence in the window / "
                              "{} corte(s) cayeron a media toma, sin silencio en la ventana"
                              .format(m.name, duros, duros))

            backup.mkdir(exist_ok=True)
            hechos = []
            fallo = None
            for i, (a, b) in enumerate(cortes, start=1):
                destino = dataset_dir / "{}_{:02d}{}".format(m.stem, i, m.suffix)
                if es_audio:
                    cmd = [ffmpeg, "-v", "error", "-y", "-i", str(m),
                           "-ss", "{:.4f}".format(a), "-t", "{:.4f}".format(b - a),
                           "-ar", "32000", "-ac", "2", str(destino)]
                else:
                    cmd = [ffmpeg, "-v", "error", "-y", "-i", str(m),
                           "-ss", "{:.4f}".format(a), "-frames:v", str(frames),
                           "-c:v", "libx264", "-crf", "16", "-preset", "medium",
                           "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "192k", "-ar", "32000", "-ac", "2",
                           str(destino)]
                try:
                    subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=True)
                    hechos.append(destino)
                except subprocess.CalledProcessError as exc:
                    det = (exc.stderr or "").strip().splitlines()
                    fallo = det[-1] if det else "exit {}".format(exc.returncode)
                    break
                except Exception as exc:
                    fallo = str(exc)
                    break

            if fallo:
                # Sin todas las piezas no se toca el original: dejar la mitad de
                # una grabacion troceada y la otra mitad no es peor que no haber
                # empezado. / Without every piece the original is left alone:
                # half a recording split and half not is worse than not starting.
                for h in hechos:
                    try:
                        h.unlink()
                    except Exception:
                        pass
                errores.append("{}: ffmpeg: {}".format(m.name, fallo))
                continue

            # El caption del original se copia a cada trozo: describe la voz o la
            # accion, que no cambian al cortar. Si alguno necesita otro texto, se
            # edita en el Dataset Manager. / The original's caption is copied to
            # every piece: it describes the voice or the action, which cutting
            # does not change.
            txt = m.with_suffix(".txt")
            if txt.exists():
                try:
                    contenido = txt.read_text(encoding="utf-8")
                    for h in hechos:
                        h.with_suffix(".txt").write_text(contenido, encoding="utf-8")
                    shutil.move(str(txt), str(backup / txt.name))
                except Exception:
                    pass

            shutil.move(str(m), str(backup / m.name))
            troceadas += 1
            producidas += len(hechos)
            detalles.append("{} ({:.1f}s -> {} takes / tomas)".format(m.name, dur, len(hechos)))

        return jsonify({"status": "ok" if not errores else "partial",
                        "split": troceadas, "skipped": saltadas, "produced": producidas,
                        "details": detalles, "warnings": avisos, "errors": errores,
                        "backup": str(backup) if troceadas else None})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/prepare-clips", methods=["POST"])
@app.route("/api/convert-fps", methods=["POST"])
def convert_fps():
    """Deja cada clip a `fps` exactos y con un recuento valido 17n+5.

    LO QUE NO HACE, Y POR QUE.

    No recorta por el final. Antes si: se quedaba con los `num_frames` primeros
    fotogramas. Eso, sumado a que la pre-cache tambien lee los N primeros y a que
    el captioner miraba el clip ENTERO, produjo el fallo que motivo esta
    reescritura: un dataset de clips de 161 fotogramas entrenado a 107 aprendio
    solo el 66% de la transformacion, mientras el caption describia el final que
    los pixeles no contenian. El modelo se plantaba a mitad y volvia atras.
    Recortar es trabajo de la pre-cache, que lo hace sin tocar el original;
    hacerlo tambien aqui era destructivo y redundante.

    No reetiqueta con -itsscale. Eso conserva los fotogramas pero cambia la
    VELOCIDAD: 161 fotogramas a 32 fps reetiquetados a 24 pasan de 5,03 a 6,71
    segundos, un 33% mas lentos. H3 razona a 24 fps, asi que ese clip le ensena
    el efecto a camara lenta.

    LO QUE SI HACE. Remuestrea a `fps` conservando la duracion real, y ajusta al
    17n+5 mas cercano estirando o comprimiendo el tiempo lo justo. Para 161
    fotogramas a 32 fps: 5,031 s x 24 = 120,75 fotogramas ideales -> el valido
    mas cercano es 124 -> 5,167 s, un 2,7% mas lento. Ningun fotograma del
    original se pierde y la velocidad practicamente no cambia.

    El precio es una recodificacion (crf 16, casi sin perdida visual). El
    original queda intacto en _originals.

    Leaves every clip at exactly `fps` with a valid 17n+5 count. It no longer
    trims to num_frames: that trimming, combined with the pre-cache also reading
    the first N frames and the captioner looking at the WHOLE clip, is what
    caused the failure behind this rewrite -- 161 frame clips trained at 107
    learned only 66% of the transformation while the caption described an ending
    the pixels did not contain, so generation stalled halfway and reversed.
    Trimming is the pre-cache's job, non-destructively. It no longer uses
    -itsscale either: that keeps the frames but changes the SPEED (161 frames at
    32 fps relabelled to 24 go from 5.03 s to 6.71 s, 33% slower), and H3 reasons
    at 24 fps, so such a clip teaches the effect in slow motion. Instead it
    resamples to `fps` preserving real duration and lands on the nearest 17n+5 by
    stretching time just enough. The cost is a re-encode (crf 16, visually
    lossless); the original stays untouched in _originals.
    """
    try:
        data = request.get_json(force=True) or {}
        target = float(data.get("fps", 24.0))
        # Ya NO se lee num_frames. El recuento lo fija la duracion real del
        # clip, no el ajuste de entrenamiento: si alguien quiere entrenar con
        # menos fotogramas, eso lo decide la pre-cache sobre una copia, no una
        # tijera sobre el fichero de origen.
        # num_frames is NOT read any more. The count is set by the clip's real
        # duration, not by a training setting: training on fewer frames is the
        # pre-cache's decision, made on a copy, not a cut to the source file.
        if target <= 0:
            return jsonify({"status": "error", "error": "fps must be > 0 / fps debe ser > 0"}), 400

        if get_status().get("running"):
            return jsonify({"status": "error",
                            "error": "A process is running. Stop it first. / "
                                     "Hay un proceso en marcha. Detenlo primero."}), 409

        ffmpeg = _ffbin("ffmpeg")
        if not ffmpeg or not _ffbin("ffprobe"):
            return jsonify({"status": "error",
                            "error": "ffmpeg/ffprobe not found in PATH. / "
                                     "No se encuentran ffmpeg/ffprobe en el PATH."}), 400

        dataset_dir = get_dataset_dir()
        if not dataset_dir.is_dir():
            return jsonify({"status": "error", "error": "No dataset folder / No hay dataset"}), 404

        clips = sorted(f for f in dataset_dir.iterdir()
                       if f.is_file() and f.suffix.lower() in VIDEO_EXTS)
        if not clips:
            return jsonify({"status": "ok", "converted": 0, "skipped": 0, "errors": [],
                            "details": [], "warnings": [],
                            "message": "No clips / No hay clips"})

        # Los originales van a una subcarpeta. Ni el pre-cache ni el Dataset
        # Manager recorren subdirectorios, asi que no se entrenaran dos veces.
        # Originals go to a subfolder. Neither the pre-cache nor the Dataset
        # Manager walks subdirectories, so nothing gets trained twice.
        backup = dataset_dir / "_originals"
        converted, skipped, errors, warnings = [], [], [], []

        for clip in clips:
            info = _video_info(clip)
            if info is None:
                errors.append("{}: could not read the clip / no se pudo leer el clip".format(clip.name))
                continue
            fps, frames, width, height = info

            if frames < H3_BASE_FRAMES or fps <= 0:
                errors.append("{}: only {} frames, the minimum is {} / solo {} fotogramas, "
                              "el minimo son {}".format(clip.name, frames, H3_BASE_FRAMES,
                                                        frames, H3_BASE_FRAMES))
                continue

            duracion = frames / float(fps)
            keep = h3_nearest_frames(duracion * target)
            duracion_nueva = keep / float(target)
            # Cuanto hay que estirar (>1) o comprimir (<1) el tiempo para que la
            # duracion caiga justo en `keep` fotogramas a `target` fps.
            # How much time must stretch (>1) or compress (<1) so the duration
            # lands exactly on `keep` frames at `target` fps.
            estiramiento = duracion_nueva / duracion

            desvio = abs(estiramiento - 1.0) * 100.0
            if desvio >= 10.0:
                warnings.append("{}: {:.2f}s does not sit near a valid count, so it becomes "
                                "{:.2f}s ({:+.0f}% speed) / {:.2f}s no cae cerca de un recuento "
                                "valido, asi que pasa a {:.2f}s ({:+.0f}% de velocidad)"
                                .format(clip.name, duracion, duracion_nueva, -desvio,
                                        duracion, duracion_nueva, -desvio))

            # Las dimensiones NO se tocan: el pre-cache reescala al area objetivo
            # respetando `multiple`, y recortar aqui perderia encuadre. Solo se avisa.
            # Dimensions are NOT touched: the pre-cache rescales to the target area
            # honouring `multiple`, and cropping here would lose framing. Warn only.
            if width % H3_SPATIAL_MULTIPLE or height % H3_SPATIAL_MULTIPLE:
                warnings.append("{}: {}x{} is not a multiple of {} (the pre-cache will "
                                "rescale it) / no es multiplo de {} (el pre-cache lo "
                                "reescalara)".format(clip.name, width, height,
                                                     H3_SPATIAL_MULTIPLE, H3_SPATIAL_MULTIPLE))

            tiene_audio = _has_audio(clip)
            necesita_fps = abs(fps - target) >= 0.05
            necesita_recuento = keep != frames
            if not necesita_fps and not necesita_recuento:
                skipped.append("{} (already {:.0f} fps, {} frames, {:.2f}s: left untouched) / "
                               "(ya a {:.0f} fps, {} fotogramas, {:.2f}s: no se toca)"
                               .format(clip.name, fps, frames, duracion,
                                       fps, frames, duracion))
                continue

            backup.mkdir(exist_ok=True)
            tmp = backup / ("__converting__" + clip.name)
            # setpts lleva la duracion a la que hace falta y fps remuestrea a la
            # tasa exacta. El orden importa: fps tiene que ver ya los tiempos
            # corregidos. -frames:v al final acota por si el redondeo interno de
            # ffmpeg deja un fotograma de mas.
            #
            # setpts moves the duration to what is needed and fps resamples to
            # the exact rate; order matters, fps must see the corrected
            # timestamps. The trailing -frames:v guards against ffmpeg's internal
            # rounding leaving one extra frame.
            # EL AUDIO SE CONSERVA Y SE NORMALIZA, NO SE TIRA.
            #
            # Aqui habia un -an, que borraba la pista. En un modelo que genera
            # video Y audio de forma conjunta eso destruye la mitad del material,
            # y ademas en silencio: el clip sigue reproduciendose, solo que mudo,
            # y nada lo delata hasta que alguien va a buscar el audio.
            #
            # atempo estira el audio por el MISMO factor que setpts estira el
            # video, pero invertido: setpts multiplica los TIEMPOS (>1 alarga) y
            # atempo multiplica la VELOCIDAD (<1 alarga). Sin esa inversion la
            # imagen y el sonido se separan un 2-3% a lo largo del clip, que en
            # cinco segundos es un desfase audible en una sincronia labial.
            #
            # atempo solo admite [0,5 , 2,0] por pasada, asi que se encadena. Con
            # los estiramientos de esta funcion (unos pocos por ciento) nunca hace
            # falta, pero un clip a 8 fps llevado a 24 si lo necesitaria.
            #
            # 32 kHz y estereo es lo que pide el VAE de audio de H3 (sampling_rate
            # 32000, empaquetado channel-major a 2 canales), asi que el clip queda
            # ya en ese formato y la pre-cache no tiene que adivinarlo.
            #
            # There was an -an here, which deleted the track. On a model that
            # generates video AND audio jointly that destroys half the material,
            # silently. atempo stretches the audio by the SAME factor setpts
            # stretches the video, inverted: setpts multiplies TIMESTAMPS (>1
            # lengthens) while atempo multiplies SPEED (<1 lengthens). Without the
            # inversion picture and sound drift 2-3% apart across the clip. atempo
            # accepts only [0.5, 2.0] per pass, so it is chained. 32 kHz stereo is
            # what H3's audio VAE expects, so the clip is left in that format.
            velocidad = 1.0 / estiramiento
            pasos_tempo = []
            resto = velocidad
            while resto < 0.5:
                pasos_tempo.append(0.5)
                resto /= 0.5
            while resto > 2.0:
                pasos_tempo.append(2.0)
                resto /= 2.0
            pasos_tempo.append(resto)
            filtro_audio = ",".join("atempo={:.10f}".format(x) for x in pasos_tempo)

            cmd = [ffmpeg, "-v", "error", "-y", "-i", str(clip),
                   "-vf", "setpts=PTS*{:.10f},fps={:.6f}".format(estiramiento, target),
                   "-af", filtro_audio,
                   "-frames:v", str(keep),
                   "-c:v", "libx264", "-crf", "16", "-preset", "medium",
                   "-pix_fmt", "yuv420p",
                   # Sin pista de audio en la entrada estas opciones no generan
                   # nada y ffmpeg no protesta. / With no audio stream in, these
                   # produce nothing and ffmpeg does not complain.
                   "-c:a", "aac", "-b:a", "192k", "-ar", "32000", "-ac", "2",
                   str(tmp)]

            try:
                subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
                shutil.move(str(clip), str(backup / clip.name))
                shutil.move(str(tmp), str(clip))
                cambios = []
                if necesita_fps:
                    cambios.append("{:.0f} -> {:.0f} fps".format(fps, target))
                if necesita_recuento:
                    cambios.append("{} -> {} frames".format(frames, keep))
                cambios.append("{:.2f}s -> {:.2f}s".format(duracion, duracion_nueva))
                cambios.append("audio: {}".format(
                    "32 kHz stereo" if tiene_audio else "none / sin pista"))
                converted.append("{} ({})".format(clip.name, ", ".join(cambios)))
            except subprocess.CalledProcessError as exc:
                # Sin el stderr de ffmpeg, un fallo aqui solo dice un codigo de
                # salida y no hay forma de saber que paso.
                # Without ffmpeg's stderr a failure here is just an exit code.
                detalle = (exc.stderr or "").strip().splitlines()
                errors.append("{}: ffmpeg: {}".format(
                    clip.name, detalle[-1] if detalle else "exit {}".format(exc.returncode)))
            except Exception as exc:
                errors.append("{}: {}".format(clip.name, exc))
            finally:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass

        return jsonify({"status": "ok" if not errors else "partial",
                        "converted": len(converted), "skipped": len(skipped),
                        "details": converted, "skipped_details": skipped,
                        "errors": errors, "warnings": warnings,
                        "backup": str(backup) if converted else None})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/delete-project-data", methods=["POST"])
def delete_project_data():
    """Vacia la pre-cache o la salida de entrenamiento del proyecto actual."""
    try:
        target = str(request.get_json(force=True).get("target", "")).strip().lower()
        if target == "precache":
            path = _get_precache_dir()
        elif target == "training":
            path = get_train_output_dir()
        else:
            return jsonify({"status": "error",
                            "error": "Unknown target / destino desconocido: {}".format(target)}), 400

        # Un proceso en marcha tiene ficheros abiertos: borrar por debajo deja
        # el entrenamiento escribiendo en un checkpoint que ya no existe.
        # A running process holds open files; deleting underneath it leaves the
        # trainer writing into a checkpoint that is no longer there.
        if get_status().get("running"):
            return jsonify({"status": "error",
                            "error": "A process is running. Stop it first. / "
                                     "Hay un proceso en marcha. Detenlo primero."}), 409

        if not path.is_dir():
            return jsonify({"status": "ok", "removed": 0, "path": str(path),
                            "message": "Nothing to delete / No habia nada que borrar"})

        removed, errors = _wipe_dir(path)
        return jsonify({"status": "ok" if not errors else "partial",
                        "removed": removed, "errors": errors, "path": str(path)})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/delete-dataset-image", methods=["POST"])
def delete_dataset_image():
    """Borra una imagen del dataset y su .txt."""
    try:
        filename = str(request.get_json(force=True).get("file", "")).strip()
        if not filename:
            return jsonify({"status": "error", "error": "No file / sin fichero"}), 400

        dataset_dir = get_dataset_dir()
        target = (dataset_dir / filename).resolve()

        # El nombre viene del navegador: hay que confirmar que sigue dentro del
        # dataset antes de borrar nada.
        # The name comes from the browser: confirm it is still inside the dataset
        # before deleting anything.
        if dataset_dir.resolve() not in target.parents:
            return jsonify({"status": "error",
                            "error": "Path outside the dataset / ruta fuera del dataset"}), 400
        if not target.is_file():
            return jsonify({"status": "error",
                            "error": "Not found / no existe: {}".format(filename)}), 404

        removed = []
        target.unlink()
        removed.append(target.name)
        txt = target.with_suffix(".txt")
        if txt.is_file():
            txt.unlink()
            removed.append(txt.name)

        return jsonify({"status": "ok", "removed": removed})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/save-caption-settings", methods=["POST"])
def save_caption_settings():
    """Guarda caption_settings.json.

    Ojo: no confundir con /api/save-caption-text, que escribe el .txt de UNA
    imagen. Este guarda los AJUSTES del auto-captioning.
    Not to be confused with the per-image caption endpoint: this saves the
    auto-captioning SETTINGS.
    """
    try:
        data = request.get_json(force=True)
        path = BASE_DIR / "caption_settings.json"

        # Se FUSIONA, no se reemplaza: la GUI solo manda el prompt y el modo, y
        # un write completo borraria max_new_tokens y max_image_side cada vez
        # que se pulsa el boton, descartando en silencio lo que el usuario haya
        # ajustado a mano en el JSON.
        # MERGED, not replaced: the GUI only sends the prompt and the mode, and a
        # full write would wipe max_new_tokens and max_image_side on every click,
        # silently discarding whatever the user tuned by hand in the JSON.
        merged = read_json_file(path, {})
        merged.update(data)
        write_json_file(path, merged)
        return jsonify({"status": "ok"})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/caption-settings", methods=["GET"])
def get_caption_settings():
    return jsonify(read_json_file(BASE_DIR / "caption_settings.json", {}))


def open_browser():
    try:
        webbrowser.open("http://127.0.0.1:5000")
    except Exception:
        pass


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  ACADEMIASD — MiniMax-H3 LORALAB TRAINER WEB SERVER")
    print("=" * 70)
    print(f"  Base Dir / Carpeta  : {BASE_DIR}")
    print(f"  Python Interpreter  : {sys.executable}")
    print(f"  URL                 : http://127.0.0.1:5000")
    print("=" * 70 + "\n")
    threading.Timer(1.2, open_browser).start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)