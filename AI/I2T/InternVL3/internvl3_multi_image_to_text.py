# coding=utf-8
import torch
import base64

from IPython.display import display, HTML
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)
from pathlib import Path

"""
自动识别多图像内容综合分析为自然语言文本（基于 InternVL3）

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
请用中文分析下面的多张图片。

这些图片必须作为一个整体进行理解。不要将每张图片完全独立地进行描述，而是需要结合所有图片中的视觉信息进行综合判断。

在分析过程中，请先识别各张图片中的主要视觉内容，包括人物、动物、物体、动作、环境和场景等。

然后比较不同图片之间的内容，判断图片中是否存在相同的人物、物体、场景或事件，并结合不同图片中的信息分析它们之间可能存在的关系、共同点、差异和变化。

最后，请基于所有图片的信息给出跨图片的综合分析，说明这些图片整体展示了什么，以及不同图片之间可能存在的关联。

最终结果必须包含对所有图片的整体理解和综合判断，不能只进行逐张图片描述，也不能在完成逐张描述后直接结束。

如果多张图片展示的是同一个人物、物体、场景或事件，请结合不同图片中的信息进行描述，避免重复。

如果不同图片展示的是不同内容，也请将这些内容自然地组织起来，并说明它们之间是否存在可以观察到的联系。

请只根据图片中实际可以观察到的信息进行判断。

对于无法确定的信息，不要凭空编造，可以使用“不确定”“看起来像”“可能是”等谨慎的表达。

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


# 加载所有图片
images = [
    Image.open(path).convert("RGB")
    for path in image_paths
]


# 构建 InternVL3 对话
messages = [
    {
        "role": "user",
        "content": [
            *[
                {
                    "type": "image",
                    "image": image
                }
                for image in images
            ],
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
    return_tensors="pt",
    processor_kwargs={
        "crop_to_patches": True,
        "min_patches": 2,
        "max_patches": 4,
    }
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


# 显示原始图片集合方法（HTML）
def display_images(
        image_paths,
        images_per_row=3,
        image_width=400
):
    html = "<table><tr>"

    for index, image_path in enumerate(image_paths):
        image_name = Path(image_path).name
        suffix = Path(image_path).suffix.lower()

        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(suffix, "image/jpeg")

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        html += f"""
            <td style="text-align:center; padding:10px;">
                <img
                    src="data:{mime_type};base64,{image_data}"
                    width="{image_width}"
                >
                <br>
                {image_name}
            </td>
        """

        if (index + 1) % images_per_row == 0:
            html += "</tr><tr>"

    html += "</tr></table>"

    display(HTML(html))


# 显示原始图片集合
display_images(
    image_paths=image_paths,
    images_per_row=3,
    image_width=400
)

# 输出结果
print("\n")
print("📷 多图片综合分析结果")
print("=" * 50)
print(output_text[0])

"""
📷 多图片综合分析结果
==================================================
这些图片展示了几位女性在不同场景中的活动和装扮，整体上呈现了她们在不同环境中的生活片段。

第一张图片中，一位女性坐在沙滩上，背景是海浪。她戴着一顶草帽，穿着蓝色的连衣裙，面带微笑，似乎在享受海边的时光。她用手指在沙地上画了一个心形，这表明她可能在表达对某人的爱意或纪念某个特别的时刻。沙滩和海浪的环境营造出一种轻松、愉悦的氛围。

第二张图片中，同一位女性坐在一个巨大的红色毛绒熊前。她穿着白色上衣和灰色短裙，脚上穿着白色的鞋子。毛绒熊上有一个粉色的心形装饰，背景是玻璃装饰墙，环境显得非常现代和时尚。这张图片可能是在某个特别的活动或展览中拍摄的，女性的装扮和背景的装饰都显得非常精致和有设计感。

第三张图片中，这位女性坐在室内，背景是绿色的墙壁和一些装饰物。她戴着一个毛茸茸的兔耳朵发饰，穿着白色T恤，怀里抱着一只黑白相间的猫。她看起来非常温柔和放松，环境显得温馨舒适。这张图片可能是在家中或某个舒适的室内场所拍摄的，展示了她与宠物的亲密时刻。

综合来看，这三张图片展示了一位女性在不同场景中的生活状态。从海边到现代展览，再到温馨的室内，她展现了自己多样的生活面貌和情感表达。这些场景不仅反映了她对自然、时尚和宠物的喜爱，也展示了她在不同环境中的轻松和愉悦。这些图片共同描绘了一个充满爱意和温暖的生活画面。
"""
