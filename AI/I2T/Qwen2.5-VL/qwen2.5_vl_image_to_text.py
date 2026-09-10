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
print("📷 图片分析结果")
print("=" * 50)
print(output_text[0])

"""
📷 图片分析结果
==================================================
一位女性坐在泳池边的瓷砖地板上。她穿着一件粉红色的比基尼，露出健康的肤色和纤细的四肢。她的头发扎成一个高高的发髻，显得非常整洁。她面带微笑，表情愉悦，似乎在享受阳光和游泳池带来的乐趣。

背景中可以看到一个清澈的蓝色泳池，泳池周围是绿色植物和一些大型盆栽。这些盆栽整齐地排列着，为整个场景增添了一丝生机。泳池旁边有一堵浅色的墙壁，上面有一些绿色植物点缀其中。墙后方还有一排黑色的金属栅栏，可能用于保护隐私或安全。

地面是由深色的瓷砖铺成，反射出周围的光线，显得格外明亮。整体环境给人一种宁静而舒适的氛围，适合放松和享受夏日时光。
"""
