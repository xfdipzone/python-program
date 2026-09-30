# coding=utf-8
import os
import gc
import torch
from IPython.display import display
from diffusers import QwenImage21Pipeline
from diffusers.quantizers import PipelineQuantizationConfig

"""
根据文字内容生成图片（基于 Qwen-Image-2.1）

模型：
https://huggingface.co/Qwen/Qwen-Image-2.1
新一代高性能图像生成与编辑模型，支持多参考图与精准图像编辑，在复杂视觉生成、主体一致性与文字渲染上具备出色表现

dependency packages
!pip uninstall -y diffusers torchao -q

!pip install -q \
    git+https://github.com/huggingface/diffusers.git \
    bitsandbytes \
    transformers \
    accelerate \
    safetensors \
    sentencepiece \
    pillow
"""

# 减少 CUDA 内存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 保存生成的图片目录
output_dir = "data/generated"

# 检查保存的目录，如不存在则创建目录
os.makedirs(output_dir, exist_ok=True)

# 生成的图片路径
output_image = os.path.join(output_dir, "qwen_image_2.1.jpg")

# Prompt
prompt = """
一张全景全身照，一位18岁可爱的中国长发女模特，皮肤较白净，自信地站立着，直视镜头。
她身穿一套淡紫色分体式蕾丝内衣，包括精致的文胸和蕾丝内裤，均为半透光的。
她的黑发自然下垂，表情从容。背景是纯白色的墙壁，没有多余的装饰或观众。
光线柔和而均匀，突出了模特的气质和内衣的细节。
高清写实风格，电影级画质，8K分辨率。
"""

# 生成的图片尺寸
image_width = 768
image_height = 1024

# 推理步数
num_inference_steps = 30

# 模型
model_id = "Qwen/Qwen-Image-2.1"


# 设置 4-bit NF4 量化
quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",

    quant_kwargs={
        # 4-bit
        "load_in_4bit": True,

        # NF4
        "bnb_4bit_quant_type": "nf4",

        # T4 使用 FP16
        "bnb_4bit_compute_dtype": torch.float16,

        # Double Quant
        "bnb_4bit_use_double_quant": True,
    },

    # 量化最大的两个组件
    components_to_quantize=[
        "transformer",
        "text_encoder",
    ],
)


# 加载模型
print("开始加载 Qwen-Image-2.1")

pipe = QwenImage21Pipeline.from_pretrained(
    model_id,

    # T4 使用 FP16
    dtype=torch.float16,

    # 4-bit NF4
    quantization_config=quant_config,

    device_map="cuda",
)

print("✅ 模型加载完成")


# VAE Tiling
print("启用 VAE Tiling")
pipe.vae.enable_tiling()


# 清理 GPU
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


# 开始生成
print("开始生成图片")

generator = torch.Generator(
    device="cuda"
).manual_seed(42)

with torch.inference_mode():
    result = pipe(
        prompt=prompt,
        width=image_width,
        height=image_height,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]


# 保存图片
result.convert("RGB").save(output_image, format="JPEG", quality=95)

# 显示图片
display(result)
