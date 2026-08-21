# coding=utf-8
import os
import sys
import json
import subprocess
import shutil
import site

"""
实现真人播报视频生成（基于 SadTalker）
"""
# ============================================================
# 全局路径与输入配置
# ============================================================

# 定义真人照片，音频，音频与文本对齐数据文件路径（支持相对路径或绝对路径）
IMAGE_PATH = "input/person.jpg"
AUDIO_PATH = "input/speech.mp3"
JSON_PATH = "input/subtitles.json"

WORK_DIR = "talking_head"
OUTPUT_DIR = "output"

# 衍生路径计算
SADTALKER_DIR = os.path.abspath("SadTalker")
WAV_PATH = os.path.join(WORK_DIR, "audio.wav")
RAW_DIR = os.path.join(WORK_DIR, "raw")
RAW_VIDEO = os.path.join(OUTPUT_DIR, "talking_head_512.mp4")
FINAL_VIDEO = os.path.join(OUTPUT_DIR, "talking_head_512_subtitle.mp4")

print("=" * 60)
print(">>> 开始执行播报视频生成全流程脚本...")
print("=" * 60)


# ============================================================
# 1. 环境初始化与依赖安装
# ============================================================
print("\n[1/7] 正在初始化系统环境与安装 Python 依赖包...")

subprocess.run(
    "apt-get update -qq && apt-get install -y -qq ffmpeg fonts-noto-cjk",
    shell=True,
    check=True
)

if os.path.isdir(SADTALKER_DIR):
    shutil.rmtree(SADTALKER_DIR)

print(" -> 正在拉取 SadTalker 仓库...")
subprocess.run(
    [
        "git",
        "clone",
        "--depth",
        "1",
        "https://github.com/OpenTalker/SadTalker.git",
        SADTALKER_DIR
    ],
    check=True
)

print(" -> 正在升级基础科学计算库 (numpy/scipy)...")
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--upgrade",
        "numpy==2.1.3",
        "scipy==1.14.1"
    ],
    check=True
)

packages = [
    "kornia==0.8.3",
    "imageio",
    "imageio-ffmpeg",
    "librosa",
    "scikit-image",
    "opencv-python",
    "tqdm",
    "pyyaml",
    "yacs",
    "einops",
    "safetensors",
    "omegaconf",
    "facexlib",
    "basicsr==1.4.2",
    "gfpgan==1.3.8",
]

print(" -> 正在安装图像与深度学习依赖库...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", *packages],
    check=True
)
print(" ✓ 阶段 1/7 完成：环境与依赖准备就绪。")


# ============================================================
# 2. 代码兼容性 Patch 修复（必须在 import 依赖库前完成）
# ============================================================
print("\n[2/7] 正在应用代码兼容性 Patch 修复...")

AWING_FILE = os.path.join(SADTALKER_DIR, "src/face3d/util/my_awing_arch.py")
if os.path.isfile(AWING_FILE):
    with open(AWING_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("np.float", "float")
    with open(AWING_FILE, "w", encoding="utf-8") as f:
        f.write(content)

PREPROCESS_FILE = os.path.join(SADTALKER_DIR, "src/face3d/util/preprocess.py")
if os.path.isfile(PREPROCESS_FILE):
    with open(PREPROCESS_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("np.VisibleDeprecationWarning", "DeprecationWarning")
    content = content.replace(
        "trans_params = np.array([w0, h0, s, t[0], t[1]])",
        "trans_params = np.array([w0, h0, s, t[0], t[1]], dtype=object)"
    )
    with open(PREPROCESS_FILE, "w", encoding="utf-8") as f:
        f.write(content)

# 动态定位并修补 BasicSR API 变更问题
python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
BASICSR_FILE = f"/usr/local/lib/{python_version}/dist-packages/basicsr/data/degradations.py"

if not os.path.isfile(BASICSR_FILE):
    for path in sys.path:
        target = os.path.join(path, "basicsr/data/degradations.py")
        if os.path.isfile(target):
            BASICSR_FILE = target
            break

if os.path.isfile(BASICSR_FILE):
    with open(BASICSR_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace(
        "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
        "from torchvision.transforms.functional import rgb_to_grayscale"
    )
    with open(BASICSR_FILE, "w", encoding="utf-8") as f:
        f.write(content)

print(" ✓ 阶段 2/7 完成：代码兼容性修复已覆盖。")


# ============================================================
# 3. 环境与依赖导入（在 Patch 修复后执行）
# ============================================================
print("\n[3/7] 正在加载 PyTorch 与核心算法模块...")

site.main()

import numpy as np
import scipy
from scipy import special
import torch
import torchvision
import kornia
from torchvision.transforms.functional import rgb_to_grayscale
from gfpgan import GFPGANer

if not torch.cuda.is_available():
    raise RuntimeError("CUDA 不可用，请确认 Runtime 使用 GPU。")

print(f" -> GPU 设备检查正常: {torch.cuda.get_device_name(0)}")
print(" ✓ 阶段 3/7 完成：算法核心模块加载成功。")


# ============================================================
# 4. 模型下载与完整性校验
# ============================================================
print("\n[4/7] 正在检查与下载预训练模型...")

CHECKPOINT_DIR = os.path.join(SADTALKER_DIR, "checkpoints")
GFPGAN_DIR = os.path.join(SADTALKER_DIR, "gfpgan/weights")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(GFPGAN_DIR, exist_ok=True)

def download_file(url, output):
    if os.path.isfile(output):
        return
    print(f" -> 正在下载: {os.path.basename(output)} ...")
    subprocess.run(
        ["wget", "-q", "--show-progress", url, "-O", output],
        check=True
    )
    if not os.path.isfile(output):
        raise RuntimeError(f"Download failed: {output}")

download_file(
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors",
    os.path.join(CHECKPOINT_DIR, "SadTalker_V0.0.2_512.safetensors")
)
download_file(
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar",
    os.path.join(CHECKPOINT_DIR, "mapping_00109-model.pth.tar")
)
download_file(
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar",
    os.path.join(CHECKPOINT_DIR, "mapping_00229-model.pth.tar")
)
download_file(
    "https://huggingface.co/vinthony/SadTalker/resolve/main/epoch_20.pth",
    os.path.join(CHECKPOINT_DIR, "epoch_20.pth")
)

download_file(
    "https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth",
    os.path.join(GFPGAN_DIR, "alignment_WFLW_4HG.pth")
)
download_file(
    "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    os.path.join(GFPGAN_DIR, "detection_Resnet50_Final.pth")
)
download_file(
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    os.path.join(GFPGAN_DIR, "GFPGANv1.4.pth")
)
download_file(
    "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
    os.path.join(GFPGAN_DIR, "parsing_parsenet.pth")
)

required_models = [
    os.path.join(CHECKPOINT_DIR, "SadTalker_V0.0.2_512.safetensors"),
    os.path.join(CHECKPOINT_DIR, "mapping_00109-model.pth.tar"),
    os.path.join(CHECKPOINT_DIR, "mapping_00229-model.pth.tar"),
    os.path.join(CHECKPOINT_DIR, "epoch_20.pth"),
    os.path.join(GFPGAN_DIR, "alignment_WFLW_4HG.pth"),
    os.path.join(GFPGAN_DIR, "detection_Resnet50_Final.pth"),
    os.path.join(GFPGAN_DIR, "GFPGANv1.4.pth"),
    os.path.join(GFPGAN_DIR, "parsing_parsenet.pth"),
]

for path in required_models:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing model: {path}")

print(" ✓ 阶段 4/7 完成：模型文件完整性校验通过。")


# ============================================================
# 5. 输入数据与音频预处理
# ============================================================
print("\n[5/7] 正在校验输入素材与读取 JSON 时间戳...")

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.isfile(IMAGE_PATH):
    raise FileNotFoundError(f"找不到人物图片: {IMAGE_PATH}")

if not os.path.isfile(AUDIO_PATH):
    raise FileNotFoundError(f"找不到语音: {AUDIO_PATH}")

if not os.path.isfile(JSON_PATH):
    raise FileNotFoundError(f"找不到字幕 JSON 文件: {JSON_PATH}")

# 从 JSON 文件加载时间戳
with open(JSON_PATH, "r", encoding="utf-8") as f:
    WORDS = json.load(f)

if not WORDS:
    raise ValueError("WORDS 为空")

for i, item in enumerate(WORDS):
    if "text" not in item or "start" not in item or "end" not in item:
        raise ValueError(f"第 {i} 个字幕字段缺失")
    if item["start"] < 0 or item["end"] < item["start"]:
        raise ValueError(f"第 {i} 个字幕时间戳异常")

print(f" -> 成功读取 {len(WORDS)} 条字幕时间戳数据")
print(" -> 正在使用 FFmpeg 转换语音为 16kHz 单声道 WAV 格式...")

subprocess.run(
    ["ffmpeg", "-y", "-loglevel", "error", "-i", AUDIO_PATH, "-ar", "16000", "-ac", "1", WAV_PATH],
    check=True,
)
print(" ✓ 阶段 5/7 完成：素材与语音采样就绪。")


# ============================================================
# 6. SadTalker 驱动生成视频
# ============================================================
print("\n[6/7] 正在运行 SadTalker 神经网络驱动生成说话人脸视频（耗时较长，请稍候）...")

if os.path.isdir(RAW_DIR):
    shutil.rmtree(RAW_DIR)
os.makedirs(RAW_DIR, exist_ok=True)

cmd = [
    sys.executable,
    "inference.py",
    "--driven_audio", os.path.abspath(WAV_PATH),
    "--source_image", os.path.abspath(IMAGE_PATH),
    "--result_dir", os.path.abspath(RAW_DIR),
    "--size", "512",
    "--preprocess", "crop",
    "--enhancer", "gfpgan",
    "--expression_scale", "1.0",
]

LOG_FILE = os.path.join(WORK_DIR, "sadtalker.log")
with open(LOG_FILE, "w", encoding="utf-8") as log:
    process = subprocess.run(
        cmd,
        cwd=SADTALKER_DIR,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )

if process.returncode != 0:
    raise RuntimeError(f"SadTalker 运行失败，日志路径: {LOG_FILE}")

videos = []
for root, dirs, files in os.walk(RAW_DIR):
    for file in files:
        if file.lower().endswith(".mp4"):
            videos.append(os.path.join(root, file))

if not videos:
    raise FileNotFoundError("SadTalker 没有生成 MP4")

raw_video = max(videos, key=os.path.getmtime)
shutil.copy2(raw_video, RAW_VIDEO)

print(f" ✓ 阶段 6/7 完成：基础数字人视频生成完成 -> {RAW_VIDEO}")


# ============================================================
# 7. 字幕生成与视频压制
# ============================================================
print("\n[7/7] 正在生成按字数限制分行的 KTV 变色字幕 (ASS) 并进行压制...")

ASS_FILE = os.path.join(WORK_DIR, "ktv_subtitles.ass")

# 最多显示字数控制变量
MAX_CHARS_PER_LINE = 10

# 1. 定义卡拉 OK ASS 样式
ass_header = """[Script Info]
ScriptType: v4.00+
PlayResX: 512
PlayResY: 512
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: KTV,Noto Sans CJK SC,30,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,1,2,20,20,35,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

def ass_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    centiseconds = int(round((secs - int(secs)) * 100))
    whole_seconds = int(secs)
    if centiseconds >= 100:
        centiseconds = 0
        whole_seconds += 1
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"

def ass_escape(text):
    return str(text).replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")

# 2. 结合“停顿时间”和“单行最大字数”进行分行
sentences = []
if WORDS:
    current_sentence = [WORDS[0]]
    current_char_count = len(WORDS[0]["text"])

    for item in WORDS[1:]:
        word_len = len(item["text"])
        time_gap = item["start"] - current_sentence[-1]["end"]

        # 触发切行条件：1. 停顿超过 0.8s； 2. 当前行字数加上新词超过 MAX_CHARS_PER_LINE
        if time_gap > 0.8 or (current_char_count + word_len > MAX_CHARS_PER_LINE):
            sentences.append(current_sentence)
            current_sentence = [item]
            current_char_count = word_len
        else:
            current_sentence.append(item)
            current_char_count += word_len

    if current_sentence:
        sentences.append(current_sentence)

# 3. 构造卡拉 OK ASS 语句
ass_lines = [ass_header]

for group in sentences:
    sentence_start = group[0]["start"]
    sentence_end = group[-1]["end"]

    ktv_text_parts = []
    current_time = sentence_start

    for item in group:
        word_start = item["start"]
        word_end = item["end"]
        text = ass_escape(item["text"])

        # 处理字词间的微小停顿
        gap_dur = int(round((word_start - current_time) * 100))
        if gap_dur > 0:
            ktv_text_parts.append(f"{{\\k{gap_dur}}}")

        # 计算高亮持续时间（厘秒）
        duration_cs = max(1, int(round((word_end - word_start) * 100)))

        # \kf 为平滑变色（扫过），如需整字直接跳转变色可换成 \k
        ktv_text_parts.append(f"{{\\kf{duration_cs}}}{text}")
        current_time = word_end

    full_ktv_text = "".join(ktv_text_parts)

    start_str = ass_time(sentence_start)
    end_str = ass_time(sentence_end + 0.1)  # 留 0.1s 缓冲余量
    ass_lines.append(f"Dialogue: 0,{start_str},{end_str},KTV,,0,0,0,,{full_ktv_text}\n")

with open(ASS_FILE, "w", encoding="utf-8") as f:
    f.write("".join(ass_lines))

# 4. FFmpeg 字幕硬压制
abs_ass_path = os.path.abspath(ASS_FILE).replace("\\", "/").replace(":", r"\:")
subtitle_filter = f"subtitles='{abs_ass_path}'"

ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-loglevel",
    "error",
    "-i", RAW_VIDEO,
    "-vf", subtitle_filter,
    "-c:v", "libx264",
    "-preset", "medium",
    "-crf", "20",
    "-c:a", "aac",
    "-b:a", "128k",
    "-movflags", "+faststart",
    FINAL_VIDEO,
]

subprocess.run(ffmpeg_cmd, check=True)

if not os.path.isfile(FINAL_VIDEO):
    raise FileNotFoundError("最终视频生成失败")

print(" ✓ 阶段 7/7 完成：带字幕的最终视频压制完成。")
print("\n" + "=" * 60)
print(f"🎉 全部流程顺利执行结束！")
print(f"原视频保存路径: {RAW_VIDEO}")
print(f"最终字幕视频路径: {FINAL_VIDEO}")
print("=" * 60)
