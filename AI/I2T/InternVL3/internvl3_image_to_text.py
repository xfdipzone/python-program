# coding=utf-8
import torch

from IPython.display import display
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)

"""
自动识别图像内容为自然语言文本（基于 InternVL3）

模型：
https://huggingface.co/OpenGVLab/InternVL3-8B-hf
高性能 8B 级多模态旗舰模型，在跨模态感知、复杂视觉推理与多页长文档理解上表现卓越，综合实力处于同量级前列

dependency packages
!pip install -U transformers accelerate bitsandbytes sentencepiece
!pip install pillow ipython
"""
# 模型配置
model_id = "OpenGVLab/InternVL3-8B-hf"

# 图片路径
image_path = "data/image.jpg"

# 设置提示词
prompt = """
请用中文详细描述这张图片。

请综合分析图片中的视觉信息，生成自然、连贯、准确、客观的中文描述。

重点描述图片中实际可以观察到的主要内容，包括人物、动物、物体、动作、场景、环境，以及它们之间的关系。
根据图片内容，可以适当描述光线、天气、时间、建筑、自然景观、场景类型以及整体氛围等信息。

请根据图片实际内容自行决定描述重点，不需要强行覆盖所有类型的信息。

输出要求：

只输出自然语言文本。
不要使用 Markdown 格式。
不要使用标题、数字编号、项目符号、列表或分类标签。
不要重复描述相同的信息。
不要添加与图片无关的信息。

请将相关内容自然地组织成多个短段落。
不同主题之间适当换行，每个段落之间空一行。
不要把所有内容连续输出成一个很长的段落。
也不要为了换行而机械地把每一句话单独换行。

请以图片中实际观察到的信息为基础进行描述。
对于无法确定的信息，不要凭空编造，可以使用“不确定”“看起来像”“可能是”等谨慎的表达。

描述应尽可能准确、自然，并保持客观。
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
print("\n正在分析图片，请稍候...\n")

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


# 显示原始图片
display(
    image.resize(
        (800, int(image.height * 800 / image.width))
    )
)

# 输出结果
print("\n")
print("📷 图片分析结果")
print("=" * 50)
print(output_text[0])
