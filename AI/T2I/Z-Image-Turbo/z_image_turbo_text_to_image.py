# coding=utf-8
import os
import ctypes

# Preload CUDA 12 runtime for Nunchaku
ctypes.CDLL(
    "/usr/local/lib/python3.13/dist-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
)

import torch
from IPython.display import display
from transformers import AutoTokenizer, AutoModel

from diffusers import (
    ZImagePipeline,
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)

from nunchaku import NunchakuZImageTransformer2DModel

"""
根据自然语言文本内容生成图片（基于 Z-Image-Turbo）

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
output_image = os.path.join(output_dir, "z-image-turbo.jpg")

# Prompt
prompt = """
全身人像摄影，人物从头顶到脚底完整出现在画面中，头部、双肩、双臂、双手、双腿和双脚全部清晰可见。
人物身体任何部分都没有被画面边缘裁切。
一位18岁可爱的中国长发女模特，皮肤较白净，自信地站立着，直视镜头。
她身穿一套淡紫色分体式蕾丝内衣，包括精致的文胸和蕾丝内裤，均为半透光的。
她的黑发自然下垂，表情从容。背景是纯白色的墙壁，没有多余的装饰或观众。
光线柔和而均匀，突出了模特的气质和内衣的细节。
高清写实风格，电影级画质，8K分辨率。
"""

# 生成的图片尺寸
image_width = 768
image_height = 1024

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
pipe = ZImagePipeline(
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
    width=image_width,
    height=image_height,
    num_inference_steps=num_inference_steps,
    guidance_scale=0.0,
    generator=generator,
).images[0]

# 保存图片
result.convert("RGB").save(output_image, format="JPEG", quality=95)

# 显示图片
display(result)
