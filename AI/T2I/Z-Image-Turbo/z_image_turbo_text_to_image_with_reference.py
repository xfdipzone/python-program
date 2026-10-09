# coding=utf-8
import os
import ctypes

# Preload CUDA 12 runtime for Nunchaku
ctypes.CDLL(
    "/usr/local/lib/python3.13/dist-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
)

import torch
from PIL import Image
from IPython.display import display
from transformers import AutoTokenizer, AutoModel

from diffusers import (
    ZImageImg2ImgPipeline,
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)

from nunchaku import NunchakuZImageTransformer2DModel

"""
根据参考图与自然语言文本内容生成图片（基于 Z-Image-Turbo）

模型：
https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
高效极速的 6B 级图像生成蒸馏模型，支持 8 步（Sub-second）毫秒级超快渲染，在高质量摄影写实、中英双语文字渲染与指令遵循上具备出色表现

dependency packages
!pip install -q \
    "https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu12.8torch2.11-cp313-cp313-linux_x86_64.whl" \
    --no-deps

!pip install -q "torchao>=0.17.0"
"""
# 保存生成的图片目录
output_dir = "data/generated"

# 检查保存的目录，如不存在则创建目录
os.makedirs(output_dir, exist_ok=True)

# 生成的图片路径
output_image = os.path.join(output_dir, "z-image-turbo_with_reference.jpg")

# 参考图
reference_image_path = 'data/reference.jpg'
reference_image = Image.open(reference_image_path).convert("RGB")

# Prompt
prompt = """
将人物改为穿粉红色的旗袍。

保持人物的脸部、发型、身体、姿势、背景、街道环境、
建筑物、构图和摄影角度尽可能不变。

只修改人物的服装。

真实摄影风格，
自然的人物姿态，
真实的皮肤质感，
自然光照，
高细节。
"""

# 生成的图片尺寸
image_width = reference_image.width
image_height = reference_image.height

# 推理步数
num_inference_steps = 8

# 模型
model_id = "Tongyi-MAI/Z-Image-Turbo"


# 加载模型
print("开始加载 Z-Image-Turbo")

nunchaku_model = (
    "nunchaku-ai/nunchaku-z-image-turbo/"
    "svdq-int4_r32-z-image-turbo.safetensors"
)

dtype = torch.float16


# 将 Prompt 转为 Token (Tokenizer)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    subfolder="tokenizer",
)


# 文本编码器 (Text Encoder)
text_encoder = AutoModel.from_pretrained(
    model_id,
    subfolder="text_encoder",
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
)

# 直接将文本编码器移动到 GPU (Move Text Encoder directly to GPU)
text_encoder = text_encoder.to("cuda", dtype=dtype)


# Nunchaku 四位整数量化 Transformer (Nunchaku INT4 Transformer)
transformer = NunchakuZImageTransformer2DModel.from_pretrained(
    nunchaku_model,
    torch_dtype=dtype,
)

# 将 Transformer 移动到 GPU (Move Transformer to GPU)
transformer = transformer.to("cuda")


# 变分自编码器 (VAE)
vae = AutoencoderKL.from_pretrained(
    model_id,
    subfolder="vae",
    torch_dtype=dtype,
)

# 将 VAE 移动到 GPU (Move VAE to GPU)
vae = vae.to("cuda")


# 调度器 (Scheduler)
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
)


# Nunchaku / Diffusers 兼容性补丁 (Nunchaku / Diffusers compatibility patch)
# Nunchaku 和 Diffusers 当前版本之间存在接口不兼容的问题，手动修改 Transformer 的 forward() 调用方式，让两个库能够正常配合
def nunchaku_zimage_forward_compat(
    self,
    x,
    t,
    cap_feats,
    patch_size=2,
    f_patch_size=1,
    return_dict=True,
):
    """
    Compatibility patch for:

        Nunchaku 1.2.1
        +
        Diffusers 0.41.0.dev0

    Diffusers changed the Z-Image transformer forward()
    argument order.

    Nunchaku 1.2.1 passes these arguments positionally,
    which causes patch_size / f_patch_size to be interpreted
    as other arguments.

    Use keyword arguments when calling the Diffusers
    ZImageTransformer2DModel.forward().
    """

    from diffusers.models.transformers.transformer_z_image import (
        ZImageTransformer2DModel,
    )

    from nunchaku.models.transformers.transformer_zimage import (
        NunchakuZImageRopeHook,
    )

    rope_hook = NunchakuZImageRopeHook()

    self.register_rope_hook(rope_hook)

    try:
        return ZImageTransformer2DModel.forward(
            self,
            x,
            t,
            cap_feats,
            return_dict=return_dict,
            patch_size=patch_size,
            f_patch_size=f_patch_size,
        )
    finally:
        self.unregister_rope_hook()
        del rope_hook


transformer.forward = nunchaku_zimage_forward_compat.__get__(
    transformer,
    type(transformer),
)


# 手动构建 Pipeline (Build Pipeline manually)
pipe = ZImageImg2ImgPipeline(
    scheduler=scheduler,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    transformer=transformer,
)

print("✅ 模型加载完成")

# 开始生成
print("开始生成图片")

generator = torch.Generator(
    device="cuda"
).manual_seed(42)

result = pipe(
    prompt=prompt,
    image=reference_image,
    strength=0.6,
    width=image_width,
    height=image_height,
    num_inference_steps=num_inference_steps,
    guidance_scale=0.0,
    generator=generator,
).images[0]

# 保存图片
result.convert("RGB").save(output_image, format="JPEG", quality=95)

# 显示参考图
print("参考图\n")

display(reference_image)

# 显示图片
print("\n生成的图片\n")

display(result)
