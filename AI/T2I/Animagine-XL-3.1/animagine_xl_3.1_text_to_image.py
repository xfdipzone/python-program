# coding=utf-8
import os
import torch
import translators as ts
from IPython.display import display
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

"""
根据自然语言文本内容生成动漫图片（基于 Animagine XL 3.1）

模型：
https://huggingface.co/cagliostrolab/animagine-xl-3.1
旗舰级的 6.9B 二次元动漫图像生成模型，支持精准标签控制与角色高度还原，在日系插画质感、复杂提示词遵循与多细节角色立绘生成上具备出色表现

dependency packages
!pip install -q diffusers transformers accelerate invisible-watermark OmegaConf translators
"""
# 保存生成的图片目录
output_dir = "data/generated"

# 检查保存的目录，如不存在则创建目录
os.makedirs(output_dir, exist_ok=True)

# 生成的图片路径
output_image = os.path.join(output_dir, "animagine-xl-3.1.jpg")

# Prompt
prompt = """
一位银色长发、蓝色眼睛的动漫少女，穿着白色连衣裙，站在盛开的樱花树下，柔和的春日阳光，精致的二次元插画。
"""

# 固定好二次元常用的反向提示词
negative_prompt = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digits, worst quality, low quality, blurry, cropped, close-up, upper body, half body, cut off head, cut off legs, cut off feet, out of frame, incomplete body"

# 生成的图片尺寸
image_width = 720
image_height = 1280

# 推理步数
num_inference_steps = 28

# 模型
model_id = "cagliostrolab/animagine-xl-3.1"


# 加载模型
print("开始加载 Animagine XL 3.1")

pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    dtype=torch.float16,
    use_safetensors=True
).to("cuda")

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipe.scheduler.config)
pipe.enable_attention_slicing()

print("✨ 模型加载成功！")


# 翻译中文提示词
try:
    english_prompt = ts.translate_text(
        prompt, from_language='zh', to_language='en', translator='google')

    # 强制在开头加上二次元最核心的质量标签
    final_prompt = f"masterpiece, best quality, {english_prompt}"

except Exception as e:
    print("翻译失败，降级使用默认英文提示词。")
    raise e


# 开始生成
print("开始生成图片")

with torch.inference_mode():
    image = pipe(
        prompt=final_prompt,
        negative_prompt=negative_prompt,
        width=image_width,
        height=image_height,
        num_inference_steps=num_inference_steps,
        guidance_scale=7.0
    ).images[0]

# 保存图片
image.convert("RGB").save(output_image, format="JPEG", quality=95)

# 显示图片
display(image)
