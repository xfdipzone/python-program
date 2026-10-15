# coding=utf-8
import os
import torch
from diffusers import Lumina2Pipeline
from IPython.display import display

"""
根据自然语言文本内容生成动漫图片（基于 NetaYume-Lumina-Image-2.0-Diffusers-v40）

模型：
https://huggingface.co/duongve/NetaYume-Lumina-Image-2.0-Diffusers-v40
基于 Lumina Image 2.0 与 Neta-Lumina 微调的 v40 二次元动漫图像生成模型，支持 Danbooru 标签、自然语言描述及结构化提示词控制
在日系动漫插画、角色立绘、精细人物描绘与电影感场景构建方面具有良好的表现
擅长结合丰富的角色特征、服饰细节、鲜明配色、柔和光影与景深效果，呈现富有层次感和氛围感的动漫画面
兼顾复杂提示词理解、多元素构图组织与画面细节表达，适用于角色设计、唯美插画及高质量二次元场景创作

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
    output_dir, "netayume_lumina_image_2.0_diffusers_v40.jpg")

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
model_id = "duongve/NetaYume-Lumina-Image-2.0-Diffusers-v40"


# 加载模型
print("开始加载 NetaYume-Lumina-Image-2.0-Diffusers-v40")

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
