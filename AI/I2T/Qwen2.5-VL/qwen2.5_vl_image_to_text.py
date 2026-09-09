# coding=utf-8
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info

"""
自动识别图片内容为文本（基于 Qwen2.5-VL）

模型：
https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
轻量高效的 7B 级多模态模型，在复杂视觉与高难文档解析上具备极高准确率，性能达同体量领先水平

dependency packages
!pip install -U transformers accelerate bitsandbytes qwen-vl-utils
"""
# 模型配置
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

# 图片路径
image_path = "data/image.jpg"

# 设置提示词
prompt = """
请用中文详细分析这张照片。

请从以下几个方面描述：

1. 照片整体内容和场景
2. 照片中出现的人物、动物、物体
3. 人物正在进行的活动（如果有）
4. 环境、建筑、自然景观等
5. 时间、天气、光线等可以从照片中观察到的信息
6. 照片可能拍摄的地点或场景类型（如果可以合理判断）
7. 照片整体的氛围和特点

请只描述能够从照片中观察或合理推断出的内容。
对于无法确定的信息，请明确说明“不确定”，不要凭空编造。
"""


# 检查 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 50)
print(f"当前使用的设备: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU 显存: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print("=" * 50)


# 4-bit 量化
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)


# 加载 Qwen2.5-VL-7B
print("\n正在加载 Qwen2.5-VL-7B-Instruct，请稍候...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
)

print("✅ 模型加载完成")


# Processor
# min_pixels / max_pixels 用于控制图片处理后的视觉 token 数量
processor = AutoProcessor.from_pretrained(
    model_id,
    min_pixels=256 * 28 * 28,
    max_pixels=1280 * 28 * 28
)


# 构建 Qwen2.5-VL 对话
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
    }
]


# 处理图片和文本
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
)


# 移动到 GPU
inputs = inputs.to(device)


# 开始推理
print("\n正在分析图片，请稍候...\n")

generated_ids = model.generate(
    **inputs,
    max_new_tokens=512
)


# 去掉输入部分，只保留模型生成的内容
generated_ids_trimmed = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(
        inputs.input_ids,
        generated_ids
    )
]


# 解码
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)


# 输出结果
print("=" * 50)
print("📷 图片分析结果")
print("=" * 50)
print(output_text[0])
print("=" * 50)
