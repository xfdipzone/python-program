# coding=utf-8
import torch

from IPython.display import display
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)
from pathlib import Path

"""
批量独立自动识别图像内容为自然语言文本（基于 InternVL3）

模型：
https://huggingface.co/OpenGVLab/InternVL3-8B-hf
高性能 8B 级多模态旗舰模型，在跨模态感知、复杂视觉推理与多页长文档理解上表现卓越，综合实力处于同量级前列

dependency packages
!pip install -U transformers accelerate bitsandbytes sentencepiece
!pip install pillow ipython
"""
# 模型配置
model_id = "OpenGVLab/InternVL3-8B-hf"

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


# 加载 InternVL3-8B-hf
print("\n正在加载 InternVL3-8B-hf，请稍候...")

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
)

model.config.tie_word_embeddings = False

# 推理模式
model.eval()

print("✅ 模型加载完成")


# Processor
processor = AutoProcessor.from_pretrained(
    model_id
)


# 循环分析每一张图片
for index, image_path in enumerate(image_paths, 1):

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
这张照片拍摄于一个海滩上，背景是轻柔的海浪拍打着沙滩。海水呈现出淡淡的绿色，波浪轻轻拍打在岸边，泛起白色的泡沫。沙滩上的沙子细腻，颜色偏浅，阳光洒在沙滩上，显得温暖而柔和。

照片中的人物坐在沙滩上，身穿一条蓝白相间的连衣裙，裙子轻盈飘逸，带有水彩般的图案，显得非常清新。她戴着一顶宽边草帽，帽子上有一些装饰，看起来既时尚又适合海边的环境。她的长发自然垂落，发色深棕，与整体装扮相得益彰。

人物面带微笑，眼神温柔，看向镜头，给人一种愉悦和放松的感觉。她用一只手轻轻触碰着沙地，另一只手支撑在沙滩上，身体微微前倾，似乎在与镜头互动。在她面前的沙地上，用手指画出了一个心形的图案，心形的线条清晰可见，为画面增添了一份浪漫和温馨的氛围。

整体来看，这张照片充满了夏日海滩的轻松与惬意，阳光、海浪、沙滩和人物的装扮与动作，共同营造出一种悠闲、愉快的氛围。

📷 image2.jpg 分析结果
==================================================
这张图片展示了一位年轻女性坐在一个巨大的红色玫瑰花熊雕塑前。玫瑰熊雕塑非常显眼，由大量红色玫瑰花精心编织而成，熊的脖子上系着一条白色丝带，丝带上用粉色字体写着“Meow”。熊的胸前还有一个粉色的心形装饰，看起来非常可爱和浪漫。

女性坐在玫瑰熊雕塑的底座上，她穿着白色长袖上衣和灰色百褶裙，搭配白色袜子和厚底鞋。她的长发披散在肩上，面带微笑，显得非常温柔和自然。

背景中可以看到一个现代化的室内环境，地面是大理石材质，反射出柔和的光线。玫瑰熊雕塑的底座周围有彩色的灯光效果，增添了一种梦幻的氛围。背景的墙上装饰着透明的水晶吊饰，进一步提升了环境的精致感。

窗外可以看到一些绿色植物和建筑结构，表明这是一个开放且明亮的空间，可能是商场或展览馆的一部分。整体氛围显得非常时尚和浪漫，适合拍照留念。

📷 image3.jpg 分析结果
==================================================
这张图片展示了一位年轻女性坐在床上，抱着一只猫。她穿着一件白色的长袖T恤，T恤上有一个小老虎的图案。她的头上戴着一顶毛茸茸的兔子耳朵发箍，显得非常可爱。她的头发是深色的，扎成一个低马尾，几缕头发自然地垂在脸旁。

她坐在一张浅色的床上，床上有白色的枕头和被子。床边放着一个浅粉色的包，包的带子随意地搭在床上。背景墙是淡绿色的，墙上挂着一些装饰物，包括一串垂下来的白色流苏和一个蓝色的小物件。

她怀里的猫是黑白相间的，猫的眼睛大而圆，正直视镜头，看起来有些严肃。她用双手轻轻抱着猫，显得非常温柔和爱护。

整个场景看起来非常温馨和舒适，光线柔和，给人一种宁静的感觉。房间布置简洁，色调柔和，营造出一种放松和舒适的氛围。
"""
