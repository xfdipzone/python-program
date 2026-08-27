# coding=utf-8
import json
import os
import shutil
import subprocess
from pathlib import Path

"""
实现真人播报视频生成（基于 MuseTalk v1.5）
"""
# ============================================================
# 全局路径与输入配置
# ============================================================
ROOT = Path.cwd()

# 定义真人照片，音频，音频与文本对齐数据文件路径
INPUT_IMAGE = ROOT / "input/person.jpg"
INPUT_AUDIO = ROOT / "input/speech.mp3"
SUBTITLE_JSON = ROOT / "input/subtitles.json"

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

"""
通用 Shell 命令执行函数。

参数:
    cmd (list): 要执行的命令列表，如 ['ls', '-l']
    check (bool): 是否在命令返回码非 0 时抛出异常，默认为 True

返回:
    subprocess.CompletedProcess: 包含命令执行结果的对象
"""
def run(cmd, check=True):
    cmd = [str(x) for x in cmd]

    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    if check and result.returncode != 0:

        print("\n❌ Command failed:")
        print(" ".join(cmd))

        print("\nLast output:")
        print(result.stdout[-3000:])

        raise RuntimeError("\n命令执行失败：\n" + " ".join(cmd))

    return result


"""
使用 wget 下载文件，包含文件缓存校验与完整性检查。

参数:
    url (str): 文件下载地址
    output (str | Path): 本地保存路径
    min_size_mb (float): 允许的最小文件大小（单位：MB），默认为 0.01 MB
"""
def download_file(url, output, min_size_mb=0.01):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Existing file
    # --------------------------------------------------------

    if output.exists():

        size_bytes = output.stat().st_size
        size_mb = size_bytes / 1024 / 1024

        if size_bytes > 0 and size_mb >= min_size_mb:

            print(f"  ✓ {output.name} 已存在")

            return

        print(f"  ⚠ {output.name} 异常，重新下载")

        output.unlink()

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    print(f"  ↓ Downloading {output.name}...")

    result = subprocess.run(
        [
            "wget",
            "-c",
            "--timeout=60",
            "--tries=5",
            url,
            "-O",
            str(output),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:

        if output.exists():
            output.unlink()

        print(result.stderr[-3000:])

        raise RuntimeError("\n模型下载失败：\n" + url)

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

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
            f"\n下载文件大小异常：\n"
            f"{output}\n"
            f"Size: {size_mb:.2f} MB\n"
            f"Expected >= {min_size_mb:.2f} MB"
        )

    print(f"  ✓ {output.name} " f"({size_mb:.1f} MB)")


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

    run(
        [
            "apt-get",
            "install",
            "-y",
            "-qq",
            "python3.10",
            "python3.10-venv",
            "python3.10-dev",
        ]
    )

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
        text=True,
    )

    if venv_check.returncode != 0:
        print("  Installing python3.10-venv...")
        run(["apt-get", "update", "-qq"])
        run(["apt-get", "install", "-y", "-qq", "python3.10-venv"])

        venv_check = subprocess.run(
            [PYTHON310, "-c", "import ensurepip, venv; print('venv OK')"],
            capture_output=True,
            text=True,
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

run(
    [
        PYTHON,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip<25",
        "setuptools<70",
        "wheel",
    ]
)

print("  ✓ pip ready")


# ============================================================
# [4/16] PyTorch
# ============================================================

print("\n[4/16] PyTorch 2.0.1 + CUDA 11.8")

torch_check = subprocess.run(
    [str(PYTHON), "-c", "import torch; print(torch.__version__)"],
    capture_output=True,
    text=True,
)

if torch_check.returncode == 0:
    print("  ✓ PyTorch already installed:", torch_check.stdout.strip())

else:
    print("  Installing PyTorch...")

    run(
        [
            PYTHON,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "torch==2.0.1+cu118",
            "torchvision==0.15.2+cu118",
            "torchaudio==2.0.2+cu118",
            "--index-url",
            "https://download.pytorch.org/whl/cu118",
        ]
    )

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

run(
    [
        PYTHON,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--no-deps",
        "diffusers==0.30.2",
    ]
)

run(
    [
        PYTHON,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--no-deps",
        "transformers==4.30.2",
    ]
)

run(
    [
        PYTHON,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "accelerate==0.28.0",
        "huggingface_hub==0.30.2",
    ]
)

print("  ✓ Diffusers / Transformers ready")


# ============================================================
# [7/16] OpenMMLab
# ============================================================

print("\n[7/16] OpenMMLab")

run([PYTHON, "-m", "pip", "install", "--no-cache-dir", "mmengine==0.10.7"])

run(
    [
        PYTHON,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "mmcv==2.0.1",
        "-f",
        "https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html",
    ]
)

run([PYTHON, "-m", "pip", "install", "--no-cache-dir", "mmdet==3.1.0"])


MMPose_ROOT = ROOT / "mmpose"

if not MMPose_ROOT.exists():

    print("  Cloning MMPose 1.1.0...")

    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            "v1.1.0",
            "https://github.com/open-mmlab/mmpose.git",
            str(MMPose_ROOT),
        ]
    )

else:

    print("  ✓ MMPose source exists")


run(
    [
        PYTHON,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--no-deps",
        "-e",
        str(MMPose_ROOT),
    ]
)

print("  ✓ OpenMMLab ready")


# ============================================================
# [8/16] MuseTalk source
# ============================================================

print("\n[8/16] MuseTalk source")

if not MUSE_ROOT.exists():
    print("  Cloning MuseTalk...")

    run(
        [
            "git",
            "clone",
            "https://github.com/TMElyralab/MuseTalk.git",
            str(MUSE_ROOT),
        ]
    )

else:
    print("  ✓ MuseTalk source exists")


# ============================================================
# [9/16] MuseTalk PYTHONPATH
# ============================================================

print("\n[9/16] MuseTalk PYTHONPATH")

site_packages = subprocess.check_output(
    [PYTHON, "-c", "import site; print(site.getsitepackages()[0])"], text=True
).strip()

pth_file = Path(site_packages) / "musetalk_local.pth"

pth_file.write_text(str(MUSE_ROOT) + "\n", encoding="utf-8")

print("  ✓ PYTHONPATH configured")


# ============================================================
# [10/16] MuseTalk V1.5 models
# ============================================================

print("\n[10/16] MuseTalk V1.5 models")

MODEL_ROOT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 1. MuseTalk V1.5 UNet
# ------------------------------------------------------------

print("  [1/8] UNet")

download_file(
    "https://huggingface.co/"
    "TMElyralab/MuseTalk/"
    "resolve/main/"
    "musetalkV15/unet.pth",
    MODEL_ROOT / "musetalkV15/unet.pth",
    min_size_mb=3000,
)


# ------------------------------------------------------------
# 2. MuseTalk V1.5 config
# ------------------------------------------------------------

print("  [2/8] MuseTalk config")

download_file(
    "https://huggingface.co/"
    "TMElyralab/MuseTalk/"
    "resolve/main/"
    "musetalkV15/musetalk.json",
    MODEL_ROOT / "musetalkV15/musetalk.json",
    min_size_mb=0,
)


# ------------------------------------------------------------
# 3. DWPose
# ------------------------------------------------------------

print("  [3/8] DWPose")

download_file(
    "https://huggingface.co/"
    "yzd-v/DWPose/"
    "resolve/main/"
    "dw-ll_ucoco_384.pth",
    MODEL_ROOT / "dwpose/dw-ll_ucoco_384.pth",
    min_size_mb=100,
)


# ------------------------------------------------------------
# 4. SyncNet
# ------------------------------------------------------------

print("  [4/8] SyncNet")

download_file(
    "https://huggingface.co/"
    "ByteDance/LatentSync/"
    "resolve/main/"
    "latentsync_syncnet.pt",
    MODEL_ROOT / "syncnet/latentsync_syncnet.pt",
    min_size_mb=100,
)


# ------------------------------------------------------------
# 5. SD VAE
# ------------------------------------------------------------

print("  [5/8] SD VAE")

download_file(
    "https://huggingface.co/"
    "stabilityai/sd-vae-ft-mse/"
    "resolve/main/"
    "config.json",
    MODEL_ROOT / "sd-vae/config.json",
    min_size_mb=0,
)

download_file(
    "https://huggingface.co/"
    "stabilityai/sd-vae-ft-mse/"
    "resolve/main/"
    "diffusion_pytorch_model.bin",
    MODEL_ROOT / "sd-vae/diffusion_pytorch_model.bin",
    min_size_mb=300,
)


# ------------------------------------------------------------
# 6. Whisper
# ------------------------------------------------------------

print("  [6/8] Whisper")

download_file(
    "https://huggingface.co/"
    "openai/whisper-tiny/"
    "resolve/main/"
    "config.json",
    MODEL_ROOT / "whisper/config.json",
    min_size_mb=0,
)

download_file(
    "https://huggingface.co/"
    "openai/whisper-tiny/"
    "resolve/main/"
    "pytorch_model.bin",
    MODEL_ROOT / "whisper/pytorch_model.bin",
    min_size_mb=100,
)

download_file(
    "https://huggingface.co/"
    "openai/whisper-tiny/"
    "resolve/main/"
    "preprocessor_config.json",
    MODEL_ROOT / "whisper/preprocessor_config.json",
    min_size_mb=0,
)


# ------------------------------------------------------------
# 7. Face Parsing
# ------------------------------------------------------------

print("  [7/8] Face Parsing")

download_file(
    "https://huggingface.co/"
    "camenduru/MuseTalk/"
    "resolve/main/"
    "face-parse-bisent/79999_iter.pth",
    MODEL_ROOT / "face-parse-bisent/79999_iter.pth",
    min_size_mb=50,
)


# ------------------------------------------------------------
# 8. ResNet18
# ------------------------------------------------------------

print("  [8/8] Face Parsing ResNet18")

download_file(
    "https://download.pytorch.org/" "models/" "resnet18-5c106cde.pth",
    MODEL_ROOT / "face-parse-bisent/resnet18-5c106cde.pth",
    min_size_mb=35,
)

print("  ✓ All model downloads checked")


# ============================================================
# [11/16] Environment verification
# ============================================================

print("\n[11/16] Environment verification")

VERIFY_CODE = r"""
import sys
import torch

assert torch.__version__.startswith("2.0.1")

assert torch.cuda.is_available()

import mmengine
import mmcv
import mmdet
import mmpose

import musetalk

from mmpose.apis import (
    inference_topdown,
    init_model
)

from xtcocotools.coco import COCO

print(
    "Python:",
    sys.version.split()[0]
)

print(
    "PyTorch:",
    torch.__version__
)

print(
    "CUDA:",
    torch.version.cuda
)

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print(
    "OpenMMLab: OK"
)

print(
    "MuseTalk: OK"
)

print(
    "MMPose API: OK"
)

print(
    "xtcocotools: OK"
)
"""

run([PYTHON, "-c", VERIFY_CODE])

print("  ✓ Environment OK")


# ============================================================
# [12/16] Model verification
# ============================================================

print("\n[12/16] Model verification")

MODELS = {
    "DWPose": MODEL_ROOT / "dwpose/dw-ll_ucoco_384.pth",
    "Face Parsing": MODEL_ROOT / "face-parse-bisent/79999_iter.pth",
    "Face Parsing ResNet": MODEL_ROOT
    / "face-parse-bisent/resnet18-5c106cde.pth",
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

from PIL import Image

img = Image.open(INPUT_IMAGE)

width, height = img.size

new_width = width if width % 2 == 0 else width - 1

new_height = height if height % 2 == 0 else height - 1

if (new_width, new_height) != (width, height):

    img = img.crop((0, 0, new_width, new_height))

if img.mode != "RGB":

    img = img.convert("RGB")

img.save(PREPARED_IMAGE, quality=95)

print(f"  ✓ Image ready: " f"{img.size[0]}x{img.size[1]}")


# ============================================================
# [15/16] Create inference config
# ============================================================

print("\n[15/16] Inference config")

CONFIG.write_text(
    f"""task_0:
  video_path: "{PREPARED_IMAGE}"
  audio_path: "{INPUT_AUDIO}"
""",
    encoding="utf-8",
)

print("  ✓ Config ready")


# ============================================================
# [16/16] MuseTalk inference
# ============================================================

print("\n[16/16] MuseTalk V1.5 inference")
print("  🚀 T4 / FP16 inference starting...")
print()


UNET = MODEL_ROOT / "musetalkV15/unet.pth"
UNET_CONFIG = MODEL_ROOT / "musetalkV15/musetalk.json"
WHISPER_DIR = MODEL_ROOT / "whisper"


cmd = [
    str(PYTHON),

    str(
        MUSE_ROOT /
        "scripts/inference.py"
    ),

    "--inference_config",
    str(CONFIG),

    "--result_dir",
    str(RESULT_DIR),

    "--version",
    "v15",

    "--unet_model_path",
    str(UNET),

    "--unet_config",
    str(UNET_CONFIG),

    "--whisper_dir",
    str(WHISPER_DIR),

    "--fps",
    "25",

    "--batch_size",
    "4",

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
        lines = LOG_FILE.read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines()

        print("\n".join(lines[-80:]))

    except Exception as e:

        print("读取日志失败:", e)

    raise RuntimeError(
        f"""
MuseTalk 推理失败。

完整日志：
{LOG_FILE}
"""
    )

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
        f"""
MuseTalk 执行成功，
但没有找到输出视频。

Result:
{RESULT_DIR}

Log:
{LOG_FILE}
"""
    )

output_videos.sort(key=lambda p: p.stat().st_mtime)
output_video = output_videos[-1]
size_mb = output_video.stat().st_size / 1024 / 1024


# ============================================================
# [17/19] Chinese font
# ============================================================

print("\n[17/19] Chinese font")

# ASS/libass 在 Colab 默认环境里可能没有中文字体。
# 调试结果证明：安装 Noto CJK 后，ASS 可以正常显示中文。

font_check = subprocess.run(
    ["fc-match", "Noto Sans CJK SC"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

if font_check.returncode != 0 or "NotoSansCJK" not in font_check.stdout:
    print("  Installing fonts-noto-cjk...")
    run(["apt-get", "update", "-qq"])
    run(["apt-get", "install", "-y", "-qq", "fonts-noto-cjk"])
    run(["fc-cache", "-f"])

else:
    print("  ✓ Noto Sans CJK SC already installed")

font_check = subprocess.run(
    ["fc-match", "Noto Sans CJK SC"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

if font_check.returncode != 0 or "NotoSansCJK" not in font_check.stdout:

    raise RuntimeError("中文字体 Noto Sans CJK SC 安装失败。")


print("  ✓ Chinese font ready:")
print("    ", font_check.stdout.strip())


# ============================================================
# [18/19] subtitles
# ============================================================

print("\n[18/19] subtitles")

ASS_FILE = WORK_DIR / "subtitles.ass"
RAW_VIDEO = RESULT_DIR / "raw_musetalk.mp4"
FINAL_VIDEO = RESULT_DIR / "final_musetalk_subtitle.mp4"

if not SUBTITLE_JSON.exists():
    raise FileNotFoundError(f"找不到字幕文件：\n{SUBTITLE_JSON}")

with open(SUBTITLE_JSON, "r", encoding="utf-8") as f:
    WORDS = json.load(f)

if not isinstance(WORDS, list) or not WORDS:
    raise RuntimeError("subtitles.json 必须是非空数组。")

for index, item in enumerate(WORDS):
    if not all(key in item for key in ("text", "start", "end")):
        raise RuntimeError(
            f"subtitles.json 第 {index + 1} 项缺少 text/start/end。"
        )

print(f"  ✓ Loaded {len(WORDS)} subtitle items")


# ------------------------------------------------------------
# ASS helpers
# ------------------------------------------------------------

"""
将秒数转换为 ASS 字幕格式的时间戳格式 (H:MM:SS.cs)。

参数:
    seconds (float/int): 需要转换的总秒数

返回:
    str: ASS 格式的时间字符串，例如 "0:01:23.45"
    """
def ass_time(seconds):
    seconds = max(0.0, float(seconds))

    total_cs = int(round(seconds * 100))

    hours = total_cs // 360000
    total_cs %= 360000

    minutes = total_cs // 6000
    total_cs %= 6000

    secs = total_cs // 100
    cs = total_cs % 100

    return f"{hours}:" f"{minutes:02d}:" f"{secs:02d}." f"{cs:02d}"


"""
对文本进行转义，防止其中包含的特殊字符影响 ASS 字幕样式的解析。

参数:
    text (str): 待转义的原文本

返回:
    str: 转义处理后的文本，安全用于 ASS 特效/字幕内容
"""
def ass_escape(text):
    return (
        str(text)
        .replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


# ------------------------------------------------------------
# 每 10 个字一行
# ------------------------------------------------------------

MAX_CHARS_PER_LINE = 10

sentences = []
current_sentence = []
current_char_count = 0

for item in WORDS:
    text = str(item["text"])
    char_count = len(text)

    if (
        current_sentence
        and current_char_count + char_count > MAX_CHARS_PER_LINE
    ):

        sentences.append(current_sentence)
        current_sentence = []
        current_char_count = 0

    current_sentence.append(item)
    current_char_count += char_count

if current_sentence:
    sentences.append(current_sentence)

print(f"  ✓ Created {len(sentences)} subtitle lines")


# ------------------------------------------------------------
# ASS
#
# Layer 0:
#    白色整行字幕
#
# Layer 1:
#    黄色字幕高亮
#    SecondaryColour 使用透明色，
#    PrimaryColour 使用黄色，
#    \kf 按每个字的 start/end 推进。
# ------------------------------------------------------------

ass_header = """[Script Info]
Title: MuseTalk KTV Subtitle
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: None

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: KTV,Noto Sans CJK SC,52,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,2,2,80,80,70,1
Style: KTVHighlight,Noto Sans CJK SC,52,&H0000FFFF,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,3,2,2,80,80,70,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
ass_lines = [ass_header]

for group in sentences:
    if not group:
        continue

    sentence_start = float(group[0]["start"])
    sentence_end = float(group[-1]["end"])

    if sentence_end <= sentence_start:
        sentence_end = sentence_start + 0.05

    # --------------------------------------------------------
    # 整行文字
    # --------------------------------------------------------

    full_text = "".join(ass_escape(item["text"]) for item in group)

    start_str = ass_time(sentence_start)
    end_str = ass_time(sentence_end + 0.05)

    # Layer 0：白色底字
    ass_lines.append(
        f"Dialogue: 0,{start_str},{end_str},KTV,,0,0,0,,{full_text}\n"
    )

    # --------------------------------------------------------
    # Layer 1：黄色逐字高亮
    # --------------------------------------------------------

    ktv_parts = []
    current_time = sentence_start

    for item in group:
        word_start = float(item["start"])
        word_end = float(item["end"])
        text = ass_escape(item["text"])

        if word_start < current_time:
            word_start = current_time

        if word_end < word_start:
            word_end = word_start

        duration = word_end - word_start

        # JSON 中允许 start == end。
        # ASS 不接受 0 厘秒，因此给最小 0.01 秒。
        if duration <= 0:
            duration = 0.01

        duration_cs = max(1, int(round(duration * 100)))

        gap = word_start - current_time

        if gap > 0:
            gap_cs = max(1, int(round(gap * 100)))
            ktv_parts.append(f"{{\\k{gap_cs}}}")

        ktv_parts.append(f"{{\\kf{duration_cs}}}{text}")
        current_time = max(current_time, word_end)

    ktv_text = "".join(ktv_parts)

    ass_lines.append(
        f"Dialogue: 1,{start_str},{end_str},KTVHighlight,,0,0,0,,{ktv_text}\n"
    )

with open(ASS_FILE, "w", encoding="utf-8") as f:
    f.write("".join(ass_lines))

print(f"  ✓ ASS generated: {ASS_FILE}")


# ============================================================
# [19/19] FFmpeg burn subtitle
# ============================================================

print("\n[19/19] Burn subtitles")


# ------------------------------------------------------------
# 保存原始 MuseTalk 视频
# ------------------------------------------------------------

# 当前 RESULT_DIR 中可能已经存在历史输出。
# 排除 raw/final 后，只选择本次最新生成的视频。

output_videos = []

for p in RESULT_DIR.rglob("*"):
    if not p.is_file():
        continue

    if p.name in {
        RAW_VIDEO.name,
        FINAL_VIDEO.name,
    }:
        continue

    if p.suffix.lower() in {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
    }:
        output_videos.append(p)

if not output_videos:
    raise RuntimeError("MuseTalk 输出视频不存在。")

output_videos.sort(key=lambda p: p.stat().st_mtime)
output_video = output_videos[-1]

if RAW_VIDEO.exists():
    RAW_VIDEO.unlink()

output_video.rename(RAW_VIDEO)
raw_size_mb = RAW_VIDEO.stat().st_size / 1024 / 1024

print(f"  ✓ Raw video: {RAW_VIDEO}")
print(f"  ✓ Raw size: {raw_size_mb:.2f} MB")


# ------------------------------------------------------------
# FFmpeg subtitles filter
# ------------------------------------------------------------

if not shutil.which("ffmpeg"):
    raise RuntimeError("找不到 ffmpeg。")

ffmpeg_filters = subprocess.run(
    ["ffmpeg", "-hide_banner", "-filters"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

if (
    ffmpeg_filters.returncode != 0
    or "subtitles" not in ffmpeg_filters.stdout
):

    raise RuntimeError("当前 FFmpeg 没有 subtitles filter。")


# ------------------------------------------------------------
# 字体验证
# ------------------------------------------------------------

font_check = subprocess.run(
    ["fc-match", "Noto Sans CJK SC"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

if font_check.returncode != 0 or "NotoSansCJK" not in font_check.stdout:
    raise RuntimeError("FFmpeg/libass 所需的 Noto Sans CJK SC 不可用。")


# ------------------------------------------------------------
# ASS path
# ------------------------------------------------------------

abs_ass_path = str(ASS_FILE.resolve()).replace("\\", "/").replace(":", r"\:")
subtitle_filter = f"subtitles='{abs_ass_path}'"

if FINAL_VIDEO.exists():
    FINAL_VIDEO.unlink()

ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-hide_banner",
    "-loglevel",
    "error",

    "-i",
    str(RAW_VIDEO),

    "-vf",
    subtitle_filter,

    "-c:v",
    "libx264",

    "-preset",
    "medium",

    "-crf",
    "20",

    "-pix_fmt",
    "yuv420p",

    "-c:a",
    "aac",

    "-b:a",
    "128k",

    "-movflags",
    "+faststart",

    str(FINAL_VIDEO),
]

print("  🚀 Rendering video...")

ffmpeg_result = subprocess.run(
    ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
)

if ffmpeg_result.returncode != 0:
    print("\n❌ FFmpeg FAILED")
    print(ffmpeg_result.stderr[-5000:])
    raise RuntimeError("字幕视频生成失败。")

if not FINAL_VIDEO.exists():
    raise RuntimeError("最终视频不存在。")

final_size_mb = FINAL_VIDEO.stat().st_size / 1024 / 1024


# ============================================================
# Final
# ============================================================

print("\n" + "=" * 70)
print("🎉 MuseTalk 1.5 + FINISHED")
print("=" * 70)

print(f"\nRaw video:\n  {RAW_VIDEO}")
print(f"\nSubtitle video:\n  {FINAL_VIDEO}")
print(f"\nSubtitle Video Size: {final_size_mb:.2f} MB")
print(f"\nASS subtitle:\n  {ASS_FILE}")
print(f"\nMuseTalk log:\n  {LOG_FILE}")
print("\n✓ ALL DONE")

"""
============================================================
MuseTalk 1.5 | T4 | FP16
============================================================

[1/16] Python 3.10
  ✓ Python 3.10

[2/16] Python environment
  Creating environment...
  Installing python3.10-venv...
  ✓ Environment ready

[3/16] pip
  ✓ pip ready

[4/16] PyTorch 2.0.1 + CUDA 11.8
  Installing PyTorch...
  ✓ PyTorch ready

[5/16] Basic dependencies
  ✓ Basic dependencies ready

[6/16] Diffusers / Transformers
  ✓ Diffusers / Transformers ready

[7/16] OpenMMLab
  Cloning MMPose 1.1.0...
  ✓ OpenMMLab ready

[8/16] MuseTalk source
  Cloning MuseTalk...

[9/16] MuseTalk PYTHONPATH
  ✓ PYTHONPATH configured

[10/16] MuseTalk V1.5 models
  [1/8] UNet
  ↓ Downloading unet.pth...
  ✓ unet.pth (3242.6 MB)
  [2/8] MuseTalk config
  ↓ Downloading musetalk.json...
  ✓ musetalk.json (0.0 MB)
  [3/8] DWPose
  ↓ Downloading dw-ll_ucoco_384.pth...
  ✓ dw-ll_ucoco_384.pth (388.0 MB)
  [4/8] SyncNet
  ↓ Downloading latentsync_syncnet.pt...
  ✓ latentsync_syncnet.pt (1419.1 MB)
  [5/8] SD VAE
  ↓ Downloading config.json...
  ✓ config.json (0.0 MB)
  ↓ Downloading diffusion_pytorch_model.bin...
  ✓ diffusion_pytorch_model.bin (319.2 MB)
  [6/8] Whisper
  ↓ Downloading config.json...
  ✓ config.json (0.0 MB)
  ↓ Downloading pytorch_model.bin...
  ✓ pytorch_model.bin (144.1 MB)
  ↓ Downloading preprocessor_config.json...
  ✓ preprocessor_config.json (0.2 MB)
  [7/8] Face Parsing
  ↓ Downloading 79999_iter.pth...
  ✓ 79999_iter.pth (50.8 MB)
  [8/8] Face Parsing ResNet18
  ↓ Downloading resnet18-5c106cde.pth...
  ✓ resnet18-5c106cde.pth (44.7 MB)
  ✓ All model downloads checked

[11/16] Environment verification
  ✓ Environment OK

[12/16] Model verification
  ✓ DWPose
  ✓ Face Parsing
  ✓ Face Parsing ResNet
  ✓ MuseTalk V1.5 UNet
  ✓ MuseTalk V1.5 config
  ✓ SyncNet
  ✓ SD VAE
  ✓ SD VAE config
  ✓ Whisper
  ✓ Whisper config
  ✓ Whisper preprocessor
  ✓ All models ready

[13/16] Input
  ✓ Image: person.jpg
  ✓ Audio: speech.mp3

[14/16] Prepare image
  ✓ Image ready: 1280x1706

[15/16] Inference config
  ✓ Config ready

[16/16] MuseTalk V1.5 inference
  🚀 T4 / FP16 inference starting...
  ✓ MuseTalk inference finished

[17/19] Chinese font
  Installing fonts-noto-cjk...
  ✓ Chinese font ready:
     NotoSansCJK-Regular.ttc: "Noto Sans CJK SC" "Regular"

[18/19] subtitles
  ✓ Loaded 26 subtitle items
  ✓ Created 3 subtitle lines
  ✓ ASS generated: /content/musetalk_work/subtitles.ass

[19/19] Burn subtitles
  ✓ Raw video: /content/musetalk_work/result/raw_musetalk.mp4
  ✓ Raw size: 0.26 MB
  🚀 Rendering video...

======================================================================
🎉 MuseTalk 1.5 + FINISHED
======================================================================

Raw video:
  /content/musetalk_work/result/raw_musetalk.mp4

Subtitle video:
  /content/musetalk_work/result/final_musetalk_subtitle.mp4

Subtitle Video Size: 0.39 MB

ASS subtitle:
  /content/musetalk_work/subtitles.ass

MuseTalk log:
  /content/musetalk_work/musetalk.log

✓ ALL DONE
"""
