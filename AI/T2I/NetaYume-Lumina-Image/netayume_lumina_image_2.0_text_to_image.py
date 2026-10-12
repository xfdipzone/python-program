# coding=utf-8
import os
import torch
from diffusers import Lumina2Pipeline
from IPython.display import display

"""
根据自然语言文本内容生成动漫图片（基于 NetaYume-Lumina-Image-2.0-Diffusers）

模型：
https://huggingface.co/duongve/NetaYume-Lumina-Image-2.0-Diffusers
基于 Lumina Image 2.0 微调的二次元动漫图像生成模型，支持 Danbooru 标签与自然语言提示词控制
在日系动漫插画、角色立绘、丰富场景描绘与复杂提示词遵循方面具有良好的表现，兼顾角色细节刻画、色彩表现与多元素构图生成

dependency packages
!pip install -q -U diffusers transformers accelerate sentencepiece

!pip uninstall -y torchao
"""
# 保存生成的图片目录
output_dir = "data/generated"

# 检查保存的目录，如不存在则创建目录
os.makedirs(output_dir, exist_ok=True)

# 生成的图片路径
output_image = os.path.join(
    output_dir, "netayume_lumina_image_2.0_diffusers.jpg")

# Prompt
prompt = """
一位银色长发、蓝色眼睛的动漫少女，穿着白色连衣裙，站在盛开的樱花树下，柔和的春日阳光，精致的二次元插画。
"""

# System Prompt
system_prompt = """
你是一位擅长根据中文提示词生成高质量动漫插画的助手。
请准确理解人物外观、服装、场景和光影描述。
"""

# 生成的图片尺寸
image_width = 720
image_height = 1280

# 推理步数
num_inference_steps = 25

# 模型
model_id = "duongve/NetaYume-Lumina-Image-2.0-Diffusers"


# 加载模型
print("开始加载 NetaYume-Lumina-Image-2.0-Diffusers")

pipe = Lumina2Pipeline.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
)

pipe.enable_model_cpu_offload()

print("✨ 模型加载成功！")


# 开始生成
print("开始生成图片")

image = pipe(
    prompt=prompt,
    width=image_width,
    height=image_height,
    guidance_scale=4.0,
    num_inference_steps=num_inference_steps,
    cfg_trunc_ratio=6,
    cfg_normalization=False,
    system_prompt=system_prompt,
).images[0]

# 保存图片
image.convert("RGB").save(output_image, format="JPEG", quality=95)

# 显示图片
display(image)
