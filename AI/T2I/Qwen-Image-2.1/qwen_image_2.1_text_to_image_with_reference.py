# coding=utf-8
import os
import gc
import torch
from PIL import Image
from IPython.display import display
from diffusers import QwenImage21Pipeline
from diffusers.quantizers import PipelineQuantizationConfig

"""
根据参考图与文字内容生成图片（基于 Qwen-Image-2.1）

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
output_image = os.path.join(output_dir, "qwen_image_2.1_with_reference.jpg")

# 参考图
reference_image_path = 'data/reference.jpg'
reference_image = Image.open(reference_image_path).convert("RGB")

# Prompt
prompt = """
将人物改为穿粉红色的旗袍。

保持人物的脸部、发型、身体、姿势、背景、街道环境、
建筑物、构图和摄影角度尽可能不变。

只修改人物的服装。

真实摄影风格，
自然的人物姿态，
真实的皮肤质感，
自然光照，
高细节。
"""

# 生成的图片尺寸
image_width = reference_image.width
image_height = reference_image.height

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
        image=reference_image,
        width=image_width,
        height=image_height,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]


# 保存图片
result.convert("RGB").save(output_image, format="JPEG", quality=95)

# 显示参考图
print("参考图\n")

display(reference_image)

# 显示图片
print("生成的图片\n")

display(result)
