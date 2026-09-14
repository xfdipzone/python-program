# coding=utf-8
import torch

from IPython.display import display
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info
from pathlib import Path

"""
自动识别多图像内容综合分析为自然语言文本（基于 Qwen2.5-VL）

模型：
https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
轻量高效的 7B 级多模态模型，在复杂视觉与高难文档解析上具备极高准确率，性能达同体量领先水平

dependency packages
!pip install -U transformers accelerate bitsandbytes qwen-vl-utils
!pip install pillow ipython
"""
# 模型配置
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

# 图片目录
image_dir = Path("data")

image_paths = sorted([
    str(path)
    for path in image_dir.iterdir()
    if path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
])

# 设置提示词
prompt = """
请用中文详细分析这些图片。

这是多张图片，请将所有图片视为一个整体，综合分析所有图片中的视觉信息，生成自然、连贯、准确、客观的中文描述。

请注意：
这些图片可能展示的是同一个人物、同一个物体、同一个场景或相关场景的不同角度、不同时间或不同状态。

请综合所有图片进行判断，不要简单地逐张重复描述。

重点描述图片中实际可以观察到的主要内容，包括人物、动物、物体、动作、场景、环境，以及它们之间的关系。

如果多张图片中出现相同的人物、物体或场景，可以结合多张图片的信息进行描述，避免重复。

如果不同图片展示了不同的内容，也请分别说明，并自然地组织在整体描述中。

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

请以所有图片中实际观察到的信息为基础进行描述。
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
            *[
                {
                    "type": "image",
                    "image": path
                }
                for path in image_paths
            ],
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
print("\n正在分析多组图片，请稍候...\n")

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


# 显示原始图片集合
for image_path in image_paths:
    image = Image.open(image_path)

    display(
        image.resize(
            (300, int(image.height * 300 / image.width))
        )
    )

# 输出结果
print("\n")
print("📷 多图片综合分析结果")
print("=" * 50)
print(output_text[0])

"""
📷 多图片综合分析结果
==================================================
第一张图片展示了一位女性坐在沙滩上。她穿着一件蓝色的连衣裙，戴着一顶编织的草帽，面带微笑。她的右手在沙地上画了一个心形图案。背景是波光粼粼的海水和细腻的沙滩，给人一种宁静而愉悦的感觉。

第二张图片中，一位女性坐在一个巨大的玫瑰熊旁边。这个玫瑰熊是由红色玫瑰制成的，胸前有一个粉色的心形装饰。她穿着白色上衣和灰色裙子，脚上穿着白色的鞋子。背景是一个室内场景，有玻璃墙和灯光效果，营造出一种浪漫温馨的氛围。

第三张图片显示了一位女性抱着一只猫。她穿着白色T恤，头上戴着一个毛绒兔耳朵发饰。她的表情显得很温柔，似乎在享受与猫咪相处的时光。背景是一个室内环境，墙壁是浅绿色的，旁边有一些装饰品。猫咪有着灰白相间的毛色，正依偎在她的怀里，显得非常可爱。

这三张图片虽然拍摄地点和场景各不相同，但都传递出一种轻松愉快的氛围。第一张和第三张图片中的人物都展现出愉悦的心情，而第二张图片则充满了浪漫的气息。
"""
