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
批量独立自动识别图像内容为自然语言文本（基于 Qwen2.5-VL）

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


# 循环分析每一张图片
for index, image_path in enumerate(image_paths, 1):

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

    # 显示原始图片
    image = Image.open(image_path)

    display(
        image.resize(
            (400, int(image.height * 400 / image.width))
        )
    )

    # 输出结果
    print("\n")
    print(f"📷 {Path(image_path).name} 分析结果")
    print("=" * 50)
    print(output_text[0])

"""
📷 image1.jpg 分析结果
==================================================
在一片宁静的海滩上，一位年轻女子正坐在沙滩上。她穿着一件蓝色渐变色的连衣裙，裙子轻盈飘逸，颜色从浅蓝过渡到深蓝，给人一种清新自然的感觉。她的头发披散下来，披肩长发显得柔顺而自然。头上戴着一顶编织而成的宽边草帽，帽子边缘有精致的装饰，为整体造型增添了几分优雅。

女子的右手轻轻触碰着沙地上的线条，这些线条勾勒出一个心形图案。她的表情平静而愉悦，嘴角微微上扬，似乎在享受这片刻的宁静时光。背景中，海浪轻轻拍打着岸边，波光粼粼，海水呈现出淡绿色调，与天空的颜色相呼应。沙滩细腻柔软，金黄色的沙粒在阳光下闪烁着温暖的光芒。

整个场景充满了夏日的气息，阳光柔和地洒在沙滩上，营造出一种温馨和谐的氛围。远处的海面波澜不惊，给人一种平静和放松的感觉。这个画面不仅展示了自然美景，也传递出一种轻松愉快的心情。

📷 image2.jpg 分析结果
==================================================
在一张照片中，一位年轻女性坐在一个巨大的红色玫瑰熊旁边。这个玫瑰熊是由无数朵红色玫瑰精心制作而成，熊的胸前有一个粉色的心形图案，上面还系着一条白色丝带，丝带上写着“Moe”的字样。玫瑰熊放置在一个圆形的底座上，底座边缘有彩色灯光装饰，显得非常梦幻。

这位女性穿着一件白色的长袖上衣和一条灰色的百褶裙，脚上穿着一双浅色的玛丽珍鞋，搭配透明的丝袜。她的头发是深色的，披散在肩上。她坐在玫瑰熊的前面，双手放在膝盖上，面带微笑，似乎在享受这个温馨的时刻。

背景中可以看到一扇大窗户，窗外似乎是一个城市的景象。窗户后面有一堵由许多透明瓶子组成的墙，这些瓶子排列整齐，反射出柔和的光线。整个场景给人一种浪漫和梦幻的感觉，可能是在一个室内展览或者特别的主题活动中拍摄的。

📷 image3.jpg 分析结果
==================================================
这张照片中，一位年轻女性坐在室内环境中。她穿着一件白色T恤，头发是深色的，披散在肩上。她的头上戴着一顶毛绒兔子耳朵发饰，显得非常可爱。她正抱着一只灰色和白色相间的猫咪，猫咪的表情有些无奈，但似乎也习惯了被抱着的状态。女性的手臂上有一个小贴纸，上面有卡通图案。

背景是一间卧室，墙壁是浅绿色的，窗帘是米色的。床上铺着白色的床单和枕头，旁边放着一个粉色的小包。房间内的光线柔和，营造出一种温馨舒适的氛围。整体感觉像是一个安静的午后时光，女性和猫咪享受着彼此的陪伴。
"""
