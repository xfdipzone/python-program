# coding=utf-8
import os
import subprocess
from pathlib import Path
from PIL import Image

"""
实现真人播报视频生成（基于 MuseTalk v1.5）
"""
# ============================================================
# 全局路径与输入配置
# ============================================================
ROOT = Path.cwd()

# 定义真人照片，音频，音频路径
INPUT_IMAGE = ROOT / "input/person.jpg"
INPUT_AUDIO = ROOT / "input/speech.mp3"

# 定义基础路径
MUSE_ROOT = ROOT / "MuseTalk"
ENV_ROOT = ROOT / "musetalk_env"

PYTHON310 = "/usr/bin/python3.10"
PYTHON = ENV_ROOT / "bin" / "python"

MODEL_ROOT = MUSE_ROOT / "models"
PREPARED_IMAGE = ROOT / "input/person_even.jpg"

WORK_DIR = ROOT / "musetalk_work"
RESULT_DIR = WORK_DIR / "result"
CONFIG = WORK_DIR / "inference.yaml"
LOG_FILE = WORK_DIR / "musetalk.log"

WORK_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper
# ============================================================
def run(cmd, check=True):
    cmd = [str(x) for x in cmd]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    if check and result.returncode != 0:
        print("\n❌ Command failed:")
        print(" ".join(cmd))
        print("\nLast output:")
        print(result.stdout[-3000:])
        raise RuntimeError("\n命令执行失败：\n" + " ".join(cmd))
    return result


def download_file(url, output, min_size_mb=0.01):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Existing file
    if output.exists():
        size_bytes = output.stat().st_size
        size_mb = size_bytes / 1024 / 1024
        if size_bytes > 0 and size_mb >= min_size_mb:
            print(f"  ✓ {output.name} 已存在")
            return
        print(f"  ⚠ {output.name} 异常，重新下载")
        output.unlink()

    # Download
    print(f"  ↓ Downloading {output.name}...")
    result = subprocess.run(
        [
            "wget",
            "-c",
            "--timeout=60",
            "--tries=5",
            url,
            "-O",
            str(output)
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        if output.exists():
            output.unlink()
        print(result.stderr[-3000:])
        raise RuntimeError("\n模型下载失败：\n" + url)

    # Verify
    if not output.exists():
        raise RuntimeError("\n下载命令成功，但是文件不存在：\n" + str(output))

    size_bytes = output.stat().st_size
    size_mb = size_bytes / 1024 / 1024

    if size_bytes == 0:
        output.unlink()
        raise RuntimeError("\n下载文件是空文件：\n" + str(output))

    if min_size_mb > 0 and size_mb < min_size_mb:
        output.unlink()
        raise RuntimeError(
            f"\n下载文件大小异常：\n{output}\n"
            f"Size: {size_mb:.2f} MB\n"
            f"Expected >= {min_size_mb:.2f} MB"
        )

    print(f"  ✓ {output.name} ({size_mb:.1f} MB)")


# ============================================================
# Header
# ============================================================
print("=" * 60)
print("MuseTalk 1.5 | T4 | FP16")
print("=" * 60)


# ============================================================
# [1/16] Python 3.10
# ============================================================
print("\n[1/16] Python 3.10")

if not os.path.exists(PYTHON310):
    print("  Installing Python 3.10...")
    run(["apt-get", "update", "-qq"])
    run([
        "apt-get", "install", "-y", "-qq",
        "python3.10", "python3.10-venv", "python3.10-dev"
    ])
else:
    print("  ✓ Python 3.10")


# ============================================================
# [2/16] Python environment
# ============================================================
print("\n[2/16] Python environment")

if PYTHON.exists():
    print("  ✓ Existing environment")
else:
    print("  Creating environment...")
    venv_check = subprocess.run(
        [PYTHON310, "-c", "import ensurepip, venv; print('venv OK')"],
        capture_output=True,
        text=True
    )

    if venv_check.returncode != 0:
        print("  Installing python3.10-venv...")
        run(["apt-get", "update", "-qq"])
        run(["apt-get", "install", "-y", "-qq", "python3.10-venv"])
        venv_check = subprocess.run(
            [PYTHON310, "-c", "import ensurepip, venv; print('venv OK')"],
            capture_output=True,
            text=True
        )

    if venv_check.returncode != 0:
        print("  Using virtualenv...")
        run([PYTHON310, "-m", "pip", "install", "--upgrade", "virtualenv"])
        run([PYTHON310, "-m", "virtualenv", str(ENV_ROOT)])
    else:
        run([PYTHON310, "-m", "venv", str(ENV_ROOT)])

if not PYTHON.exists():
    raise RuntimeError("Python virtual environment 创建失败")

print("  ✓ Environment ready")


# ============================================================
# [3/16] pip
# ============================================================
print("\n[3/16] pip")

run([
    PYTHON, "-m", "pip", "install", "--upgrade",
    "pip<25", "setuptools<70", "wheel"
])

print("  ✓ pip ready")


# ============================================================
# [4/16] PyTorch
# ============================================================
print("\n[4/16] PyTorch 2.0.1 + CUDA 11.8")

torch_check = subprocess.run(
    [str(PYTHON), "-c", "import torch; print(torch.__version__)"],
    capture_output=True,
    text=True
)

if torch_check.returncode == 0:
    print("  ✓ PyTorch already installed:", torch_check.stdout.strip())
else:
    print("  Installing PyTorch...")
    run([
        PYTHON, "-m", "pip", "install", "--no-cache-dir",
        "torch==2.0.1+cu118", "torchvision==0.15.2+cu118", "torchaudio==2.0.2+cu118",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ])

print("  ✓ PyTorch ready")


# ============================================================
# [5/16] Basic dependencies
# ============================================================
print("\n[5/16] Basic dependencies")

PACKAGES = [
    "numpy==1.23.5",
    "scipy==1.10.1",
    "opencv-python==4.9.0.80",
    "matplotlib==3.7.5",
    "omegaconf",
    "einops",
    "soundfile==0.12.1",
    "librosa==0.10.2",
    "imageio",
    "imageio-ffmpeg",
    "requests",
    "tqdm",
    "munch",
    "munkres",
    "Cython",
    "xtcocotools",
    "regex==2024.11.6",
]

for package in PACKAGES:
    run([PYTHON, "-m", "pip", "install", "--no-cache-dir", package])

print("  ✓ Basic dependencies ready")


# ============================================================
# [6/16] Diffusers / Transformers
# ============================================================
print("\n[6/16] Diffusers / Transformers")

run([PYTHON, "-m", "pip", "install", "--no-cache-dir", "--no-deps", "diffusers==0.30.2"])
run([PYTHON, "-m", "pip", "install", "--no-cache-dir", "--no-deps", "transformers==4.30.2"])
run([PYTHON, "-m", "pip", "install", "--no-cache-dir", "accelerate==0.28.0", "huggingface_hub==0.30.2"])

print("  ✓ Diffusers / Transformers ready")


# ============================================================
# [7/16] OpenMMLab
# ============================================================
print("\n[7/16] OpenMMLab")

run([PYTHON, "-m", "pip", "install", "--no-cache-dir", "mmengine==0.10.7"])
run([
    PYTHON, "-m", "pip", "install", "--no-cache-dir", "mmcv==2.0.1",
    "-f", "https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html"
])
run([PYTHON, "-m", "pip", "install", "--no-cache-dir", "mmdet==3.1.0"])

MMPose_ROOT = ROOT / "mmpose"

if not MMPose_ROOT.exists():
    print("  Cloning MMPose 1.1.0...")
    run([
        "git", "clone", "--depth", "1", "--branch", "v1.1.0",
        "https://github.com/open-mmlab/mmpose.git", str(MMPose_ROOT)
    ])
else:
    print("  ✓ MMPose source exists")

run([PYTHON, "-m", "pip", "install", "--no-cache-dir", "--no-deps", "-e", str(MMPose_ROOT)])

print("  ✓ OpenMMLab ready")


# ============================================================
# [8/16] MuseTalk source
# ============================================================
print("\n[8/16] MuseTalk source")

if not MUSE_ROOT.exists():
    print("  Cloning MuseTalk...")
    run(["git", "clone", "https://github.com/TMElyralab/MuseTalk.git", str(MUSE_ROOT)])
else:
    print("  ✓ MuseTalk source exists")


# ============================================================
# [9/16] MuseTalk PYTHONPATH
# ============================================================
print("\n[9/16] MuseTalk PYTHONPATH")

site_packages = subprocess.check_output(
    [PYTHON, "-c", "import site; print(site.getsitepackages()[0])"],
    text=True
).strip()

pth_file = Path(site_packages) / "musetalk_local.pth"
pth_file.write_text(str(MUSE_ROOT) + "\n", encoding="utf-8")

print("  ✓ PYTHONPATH configured")


# ============================================================
# [10/16] MuseTalk V1.5 models
# ============================================================
print("\n[10/16] MuseTalk V1.5 models")

MODEL_ROOT.mkdir(parents=True, exist_ok=True)

# 1. MuseTalk V1.5 UNet
print("  [1/8] UNet")
download_file(
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/unet.pth",
    MODEL_ROOT / "musetalkV15/unet.pth",
    min_size_mb=3000
)

# 2. MuseTalk V1.5 config
print("  [2/8] MuseTalk config")
download_file(
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/musetalk.json",
    MODEL_ROOT / "musetalkV15/musetalk.json",
    min_size_mb=0
)

# 3. DWPose
print("  [3/8] DWPose")
download_file(
    "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth",
    MODEL_ROOT / "dwpose/dw-ll_ucoco_384.pth",
    min_size_mb=100
)

# 4. SyncNet
print("  [4/8] SyncNet")
download_file(
    "https://huggingface.co/ByteDance/LatentSync/resolve/main/latentsync_syncnet.pt",
    MODEL_ROOT / "syncnet/latentsync_syncnet.pt",
    min_size_mb=100
)

# 5. SD VAE
print("  [5/8] SD VAE")
download_file(
    "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json",
    MODEL_ROOT / "sd-vae/config.json",
    min_size_mb=0
)
download_file(
    "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin",
    MODEL_ROOT / "sd-vae/diffusion_pytorch_model.bin",
    min_size_mb=300
)

# 6. Whisper
print("  [6/8] Whisper")
download_file(
    "https://huggingface.co/openai/whisper-tiny/resolve/main/config.json",
    MODEL_ROOT / "whisper/config.json",
    min_size_mb=0
)
download_file(
    "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin",
    MODEL_ROOT / "whisper/pytorch_model.bin",
    min_size_mb=100
)
download_file(
    "https://huggingface.co/openai/whisper-tiny/resolve/main/preprocessor_config.json",
    MODEL_ROOT / "whisper/preprocessor_config.json",
    min_size_mb=0
)

# 7. Face Parsing
print("  [7/8] Face Parsing")
download_file(
    "https://huggingface.co/camenduru/MuseTalk/resolve/main/face-parse-bisent/79999_iter.pth",
    MODEL_ROOT / "face-parse-bisent/79999_iter.pth",
    min_size_mb=50
)

# 8. ResNet18
print("  [8/8] Face Parsing ResNet18")
download_file(
    "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    MODEL_ROOT / "face-parse-bisent/resnet18-5c106cde.pth",
    min_size_mb=35
)

print("  ✓ All model downloads checked")


# ============================================================
# [11/16] Environment verification
# ============================================================
print("\n[11/16] Environment verification")

VERIFY_CODE = r'''
import sys
import torch

assert torch.__version__.startswith("2.0.1")
assert torch.cuda.is_available()

import mmengine
import mmcv
import mmdet
import mmpose
import musetalk
from mmpose.apis import inference_topdown, init_model
from xtcocotools.coco import COCO

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("OpenMMLab: OK")
print("MuseTalk: OK")
print("MMPose API: OK")
print("xtcocotools: OK")
'''

run([PYTHON, "-c", VERIFY_CODE])

print("  ✓ Environment OK")


# ============================================================
# [12/16] Model verification
# ============================================================
print("\n[12/16] Model verification")

MODELS = {
    "DWPose": MODEL_ROOT / "dwpose/dw-ll_ucoco_384.pth",
    "Face Parsing": MODEL_ROOT / "face-parse-bisent/79999_iter.pth",
    "Face Parsing ResNet": MODEL_ROOT / "face-parse-bisent/resnet18-5c106cde.pth",
    "MuseTalk V1.5 UNet": MODEL_ROOT / "musetalkV15/unet.pth",
    "MuseTalk V1.5 config": MODEL_ROOT / "musetalkV15/musetalk.json",
    "SyncNet": MODEL_ROOT / "syncnet/latentsync_syncnet.pt",
    "SD VAE": MODEL_ROOT / "sd-vae/diffusion_pytorch_model.bin",
    "SD VAE config": MODEL_ROOT / "sd-vae/config.json",
    "Whisper": MODEL_ROOT / "whisper/pytorch_model.bin",
    "Whisper config": MODEL_ROOT / "whisper/config.json",
    "Whisper preprocessor": MODEL_ROOT / "whisper/preprocessor_config.json",
}

missing = []

for name, path in MODELS.items():
    if path.exists():
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}")
        missing.append(name)

if missing:
    print("\n❌ Missing models:")
    for name in missing:
        print("  -", name)
    raise RuntimeError("模型没有全部准备完成")

print("  ✓ All models ready")


# ============================================================
# [13/16] Input verification
# ============================================================
print("\n[13/16] Input")

if not INPUT_IMAGE.exists():
    raise FileNotFoundError(f"找不到人物图片：\n{INPUT_IMAGE}")

if not INPUT_AUDIO.exists():
    raise FileNotFoundError(f"找不到音频：\n{INPUT_AUDIO}")

print("  ✓ Image:", INPUT_IMAGE.name)
print("  ✓ Audio:", INPUT_AUDIO.name)


# ============================================================
# [14/16] Prepare image
# ============================================================
print("\n[14/16] Prepare image")

img = Image.open(INPUT_IMAGE)
width, height = img.size

new_width = width if width % 2 == 0 else width - 1
new_height = height if height % 2 == 0 else height - 1

if (new_width, new_height) != (width, height):
    img = img.crop((0, 0, new_width, new_height))

if img.mode != "RGB":
    img = img.convert("RGB")

img.save(PREPARED_IMAGE, quality=95)

print(f"  ✓ Image ready: {img.size[0]}x{img.size[1]}")


# ============================================================
# [15/16] Create inference config
# ============================================================
print("\n[15/16] Inference config")

CONFIG.write_text(
    f"""task_0:
  video_path: "{PREPARED_IMAGE}"
  audio_path: "{INPUT_AUDIO}"
""",
    encoding="utf-8"
)

print("  ✓ Config ready")


# ============================================================
# [16/16] MuseTalk inference
# ============================================================
print("\n[16/16] MuseTalk V1.5 inference")
print("  🚀 T4 / FP16 inference starting...\n")

UNET = MODEL_ROOT / "musetalkV15/unet.pth"
UNET_CONFIG = MODEL_ROOT / "musetalkV15/musetalk.json"
WHISPER_DIR = MODEL_ROOT / "whisper"

cmd = [
    str(PYTHON),
    str(MUSE_ROOT / "scripts/inference.py"),
    "--inference_config", str(CONFIG),
    "--result_dir", str(RESULT_DIR),
    "--version", "v15",
    "--unet_model_path", str(UNET),
    "--unet_config", str(UNET_CONFIG),
    "--whisper_dir", str(WHISPER_DIR),
    "--fps", "25",
    "--use_float16",
]

print("  🚀 Starting MuseTalk subprocess...")
print("  📄 Log:", LOG_FILE)

with open(LOG_FILE, "w", encoding="utf-8") as log:
    process = subprocess.Popen(
        cmd,
        cwd=str(MUSE_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()

    process.wait()

print(f"\n  MuseTalk exit code: {process.returncode}")


# ============================================================
# Result
# ============================================================
if process.returncode != 0:
    print("\n❌ MuseTalk FAILED")
    print("\nLast 80 lines:\n")
    try:
        lines = LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
        print("\n".join(lines[-80:]))
    except Exception as e:
        print("读取日志失败:", e)
    raise RuntimeError(f"\nMuseTalk 推理失败。\n\n完整日志：\n{LOG_FILE}\n")

print("\n  ✓ MuseTalk inference finished")


# ============================================================
# Find output
# ============================================================
output_videos = []

for p in RESULT_DIR.rglob("*"):
    if not p.is_file():
        continue
    if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        output_videos.append(p)

if not output_videos:
    raise RuntimeError(
        f"\nMuseTalk 执行成功，但没有找到输出视频。\n\nResult:\n{RESULT_DIR}\n\nLog:\n{LOG_FILE}\n"
    )

output_videos.sort(key=lambda p: p.stat().st_mtime)
output_video = output_videos[-1]
size_mb = output_video.stat().st_size / 1024 / 1024


# ============================================================
# Final
# ============================================================
print("\n" + "=" * 60)
print("🎉 MuseTalk FINISHED")
print("=" * 60)

print(f"Output: {output_video}")
print(f"Size: {size_mb:.2f} MB")
print(f"Log: {LOG_FILE}")

print("\n✓ ALL DONE")
