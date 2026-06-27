# coding=utf-8
import edge_tts
import nest_asyncio
import pandas as pd

"""
获取 Microsoft Edge TTS 所有中文声音列表

dependency packages
pip install edge-tts
pip install nest_asyncio
pip install pandas
"""
nest_asyncio.apply()

# 获取所有中文声音列表（直接打印）
async def print_chinese_voices_detailed():
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

    print("\n")

# 获取所有中文声音列表（输出到 Data Frame）
async def chinese_voices_detailed_data_frame():
    voices = await edge_tts.VoicesManager.create()

    # 筛选出所有中文声音（指定地区可用参数 Local=zh-CN）
    chinese_voices = voices.find(Language="zh")

    # 将数据保存到 Data Frame
    df = pd.DataFrame(chinese_voices)
    df = df.sort_values(by='Locale').reset_index(drop=True)
    df = df[['Locale', 'ShortName', 'Gender']]
    df.columns = ['地区/语言区域', '声音名称/播音员', '性别']

    display(df)


# 获取列表（直接打印）
await print_chinese_voices_detailed()

# 获取列表（输出到 Data Frame）
await chinese_voices_detailed_data_frame()

"""
✨ 正在打印 14 种声音列表...

Local           | ShortName                     | Gender
----------------------------------------------------------
zh-CN           | zh-CN-XiaoxiaoNeural          | Female
zh-CN           | zh-CN-XiaoyiNeural            | Female
zh-CN           | zh-CN-YunjianNeural           | Male
zh-CN           | zh-CN-YunxiNeural             | Male
zh-CN           | zh-CN-YunxiaNeural            | Male
zh-CN           | zh-CN-YunyangNeural           | Male
zh-CN-liaoning  | zh-CN-liaoning-XiaobeiNeural  | Female
zh-CN-shaanxi   | zh-CN-shaanxi-XiaoniNeural    | Female
zh-HK           | zh-HK-HiuGaaiNeural           | Female
zh-HK           | zh-HK-HiuMaanNeural           | Female
zh-HK           | zh-HK-WanLungNeural           | Male
zh-TW           | zh-TW-HsiaoChenNeural         | Female
zh-TW           | zh-TW-YunJheNeural            | Male
zh-TW           | zh-TW-HsiaoYuNeural           | Female
"""
