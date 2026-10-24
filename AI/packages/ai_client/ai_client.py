# coding=utf-8
import os
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

# 加载 .env 文件中的变量到系统的环境变量中
load_dotenv()

class AIClient:
    """封装 AI 模型的同步、异步及流式调用。"""

    def __init__(self, model=os.environ.get("AI_DEFAULT_MODEL")):
        self.model = model

        api_key = os.environ.get("KIMI_API_KEY")

        if not api_key:
            raise ValueError("未找到 KIMI_API_KEY，请检查环境变量配置。")

        self.client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("AI_BASE_URL"),
        )

        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=os.environ.get("AI_BASE_URL"),
        )

    def get_client(self):
        """获取 client"""
        return self.client

    def get_model(self):
        """获取 model"""
        return self.model

    def _build_messages(self, message, system_prompt=None):
        """创建 messages"""
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        messages.append({
            "role": "user",
            "content": message,
        })

        return messages

    def chat(self, message, system_prompt=None):
        """同步调用。"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(message, system_prompt),
        )

        return response.choices[0].message.content

    def chat_stream(self, message, system_prompt=None):
        """流式调用：逐段返回生成的文本。"""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(message, system_prompt),
            stream=True,
        )

        for chunk in stream:
            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content

            if content:
                yield content

    async def chat_async(self, message, system_prompt=None):
        """异步调用：等待完整结果。"""
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(message, system_prompt),
        )

        return response.choices[0].message.content

    async def chat_stream_async(self, message, system_prompt=None):
        """异步流式调用：逐段产生文本。"""
        stream = await self.async_client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(message, system_prompt),
            stream=True,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content

            if content:
                yield content

    async def close(self):
        """关闭异步客户端，释放网络资源。"""
        await self.async_client.close()
        self.client.close()
