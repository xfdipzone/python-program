# coding=utf-8
import torch
import os

from IPython.display import display
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)

"""
识别小票内容并结构化输出（基于 InternVL3）

模型：
https://huggingface.co/OpenGVLab/InternVL3-8B-hf
高性能 8B 级多模态旗舰模型，在跨模态感知、复杂视觉推理与多页长文档理解上表现卓越，综合实力处于同量级前列

dependency packages
!pip install -U transformers accelerate bitsandbytes sentencepiece
!pip install pillow ipython
"""
# 模型配置
model_id = "OpenGVLab/InternVL3-8B-hf"

# 小票路径
image_path = "data/receipt/receipt.jpg"

# 提示词文档
prompt_file = "data/receipt/prompt.txt"

# 读取文件
def readfile(file):
    try:
        if not os.path.exists(file):
            raise FileNotFoundError(f"文件不存在: {file}")

        with open(file, "r", encoding="utf-8") as f:
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


# 加载 InternVL3-8B-hf
print("\n正在加载 InternVL3-8B-hf，请稍候...")

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
)

# 推理模式
model.eval()

print("✅ 模型加载完成")


# Processor
processor = AutoProcessor.from_pretrained(
    model_id
)


# 加载图片
image = Image.open(image_path).convert("RGB")


# 构建 InternVL3 对话
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
    }
]


# 处理图片和文本
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)


# 移动到 GPU
inputs = inputs.to(model.device)


# 开始推理
print("\n正在分析小票，请稍候...\n")

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
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
  "raw_text": [
    "*** DUPLICATE ***",
    "*** BILLA ***",
    "Stony",
    "16725 Stony Plain Rd",
    "Edmonton AB T5P 4A9",
    "Store#: 3859 Tel#: 780-414-6362",
    "Welcome to all day breakfast Mc Donald's",
    "192",
    "03/10/2018 07:39:12 PM",
    "KS# 1",
    "QTY ITEM",
    "1 Delivery",
    "1 10 McNuggets EVM",
    "1 Barbeque Sauce",
    "1 Barbeque Sauce",
    "1 L Coke",
    "1 M French Fries",
    "1 HM Grchs S-Fry Yog",
    "1 Snoopy",
    "1 HM Apple Juice",
    "1 6 Cookies",
    "6 Choc Chip Cookie",
    "1 Baked Apple Pie",
    "1 L French Fries",
    "1 L Iced Tea",
    "Subtotal",
    "25.04",
    "GST",
    "1.11",
    "Take-Out Total",
    "26.15",
    "CREDIT SALES - McDelivery",
    "26.15",
    "Change",
    "0.00",
    "GST #: 120907092",
    "SALE #04kdfw9532"
  ],
  "fields": [
    {"label": "Store#", "value": "3859"},
    {"label": "Tel#", "value": "780-414-6362"},
    {"label": "GST #", "value": "120907092"},
    {"label": "SALE #", "value": "04kdfw9532"}
  ],
  "warnings": []
}
"""