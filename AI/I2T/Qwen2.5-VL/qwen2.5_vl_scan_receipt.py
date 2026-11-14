# coding=utf-8
import torch
import os

from IPython.display import display
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info

"""
识别小票内容并结构化输出（基于 Qwen2.5-VL）

模型：
https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
轻量高效的 7B 级多模态模型，在复杂视觉与高难文档解析上具备极高准确率，性能达同体量领先水平

dependency packages
!pip install -U transformers accelerate bitsandbytes qwen-vl-utils
!pip install pillow ipython
"""
# 模型配置
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

# 小票路径
image_path = "data/receipt/receipt.jpg"

# 提示词文档
prompt_file = "data/receipt/prompt.txt"

# 读取文件
def readfile(file):
    try:
        if not os.path.exists(file):
            raise FileNotFoundError(f"文件不存在: {file}")

        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()

        return prompt
    except Exception as e:
        raise e


# 读取提示词
prompt = readfile(prompt_file)

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

# 推理模式
model.eval()

print("✅ 模型加载完成")


# Processor
# shortest_edge / longest_edge 用于控制图片处理后的视觉 token 数量
processor = AutoProcessor.from_pretrained(
    model_id,
    size={
        "shortest_edge": 256 * 28 * 28,
        "longest_edge": 1280 * 28 * 28
    }
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
inputs = inputs.to(model.device)


# 开始推理
print("\n正在分析小票，请稍候...\n")

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=4096,
        do_sample=False
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


# 显示原始小票
image = Image.open(image_path)

display(
    image.resize(
        (500, int(image.height * 500 / image.width))
    )
)

# 输出结果
print("\n")
print("🧾 小票识别结果")
print("=" * 50)
print(output_text[0])

"""
🧾 小票识别结果
==================================================
{
  "receipt_number": "192",
  "merchant": {
    "name": "McDonald's",
    "address": "16725 Stony Plain Rd Edmonton AB T5P 4A9",
    "phone": "780-414-8362",
    "tax_id": null
  },
  "receipt_type": "retail",
  "currency": "CAD",
  "transaction_date": "2018-03-10",
  "transaction_time": "19:39:12",
  "items": [
    {
      "name": "Delivery",
      "category": "Food & Groceries",
      "quantity": 1,
      "unit": "pcs",
      "unit_price": 0.00,
      "subtotal": 0.00,
      "discount": 0.00,
      "total": 0.00
    },
    {
      "name": "10 McNuggets EVM",
      "category": "Food & Groceries",
      "quantity": 1,
      "unit": "pcs",
      "unit_price": 10.29,
      "subtotal": 10.29,
      "discount": 0.00,
      "total": 10.29
    },
    {
      "name": "Barbeque Sauce",
      "category": "Food & Groceries",
      "quantity": 1,
      "unit": "pcs",
      "unit_price": 0.40,
      "subtotal": 0.40,
      "discount": 0.00,
      "total": 0.40
    },
    {
      "name": "L Coke",
      "category": "Beverages",
      "quantity": 1,
      "unit": "L",
      "unit_price": 0.00,
      "subtotal": 0.00,
      "discount": 0.00,
      "total": 0.00
    },
    {
      "name": "M French Fries",
      "category": "Food & Groceries",
      "quantity": 1,
      "unit": "pcs",
      "unit_price": 2.99,
      "subtotal": 2.99,
      "discount": 0.00,
      "total": 2.99
    },
    {
      "name": "HM GrChs S-Fry Yog",
      "category": "Food & Groceries",
      "quantity": 1,
      "unit": "pcs",
      "unit_price": 3.99,
      "subtotal": 3.99,
      "discount": 0.00,
      "total": 3.99
    },
    {
      "name": "Snoopy",
      "category": "Food & Groceries",
      "quantity": 1,
      "unit": "pcs",
      "unit_price": 0.00,
      "subtotal": 0.00,
      "discount": 0.00,
      "total": 0.00
    },
    {
      "name": "HM Apple Juice",
      "category": "Food & Groceries",
      "quantity": 1,
      "unit": "pcs",
      "unit_price": 0.00,
      "subtotal": 0.00,
      "discount": 0.00,
      "total": 0.00
    },
    {
      "name": "Cookies",
      "category": "Food & Groceries",
      "quantity": 6,
      "unit": "pcs",
      "unit_price": 0.00,
      "subtotal": 2.89,
      "discount": 0.00,
      "total": 2.89
    },
    {
      "name": "Choc Chip Cookie",
      "category": "Food & Groceries",
      "quantity": 6,
      "unit": "pcs",
      "unit_price": 0.00,
      "subtotal": 2.89,
      "discount": 0.00,
      "total": 2.89
    },
    {
      "name": "Baked Apple Pie",
      "category": "Food & Groceries",
      "quantity": 1,
      "unit": "pcs",
      "unit_price": 1.19,
      "subtotal": 1.19,
      "discount": 0.00,
      "total": 1.19
    },
    {
      "name": "French Fries",
      "category": "Food & Groceries",
      "quantity": 1,
      "unit": "L",
      "unit_price": 3.29,
      "subtotal": 3.29,
      "discount": 0.00,
      "total": 3.29
    },
    {
      "name": "Iced Tea",
      "category": "Food & Groceries",
      "quantity": 1,
      "unit": "L",
      "unit_price": 2.99,
      "subtotal": 2.99,
      "discount": 0.00,
      "total": 2.99
    }
  ],
  "summary": {
    "subtotal": 25.04,
    "discount": 1.11,
    "tax": 1.11,
    "service_charge": null,
    "tip": null,
    "total": 26.15,
    "amount_paid": 26.15,
    "change": 0.00
  },
  "warnings": [],
  "confidence": 0.8
}
"""