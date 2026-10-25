# coding=utf-8
from pathlib import Path
import sys
import asyncio

# 获取 AI 项目根目录
AI_ROOT = Path(__file__).resolve().parent.parent.parent

# 将 AI 项目根目录加入 Python 导入路径
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))


# 导入 AIClient
from packages.ai_client import AIClient

async def write_love_letter():
    ai = AIClient()

    try:
        result = await ai.chat_async(
            message="请用中文写一封真挚、浪漫的情书。",
            system_prompt="你是一位擅长中文情感表达的写作助手。",
        )
        print(result)
    finally:
        await ai.close()


if __name__ == "__main__":
    asyncio.run(write_love_letter())
