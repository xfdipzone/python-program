# coding=utf-8
import edge_tts
import nest_asyncio

"""
获取 Microsoft Edge TTS 所有中文声音列表

dependency packages
pip install edge-tts
pip install nest_asyncio
"""
nest_asyncio.apply()

# 获取所有中文声音列表
async def list_chinese_voices_detailed():
    voices = await edge_tts.VoicesManager.create()

    # 筛选出所有中文声音（指定地区可用参数 Local=zh-CN）
    chinese_voices = voices.find(Language="zh")

    print(f"✨ 正在打印 {len(chinese_voices)} 种声音列表...\n")
    print(f"{'Local':<15} | {'ShortName':<29} | {'Gender':<6}")
    print("-" * 60)

    # 按照语言地区（Locale）排序后打印
    sorted_voices = sorted(chinese_voices, key=lambda x: x['Locale'])

    for v in sorted_voices:
        print(f"{v['Locale']:<15} | {v['ShortName']:<29} | {v['Gender']:<6}")

# 获取列表
await list_chinese_voices_detailed()
