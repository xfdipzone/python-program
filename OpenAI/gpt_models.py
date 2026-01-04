# coding=utf-8
from openai import OpenAI
from google.colab import userdata
import pandas as pd

"""
获取 OpenAI GPT 支持的模型列表

dependency packages
pip install openai
"""
client = OpenAI(api_key=userdata.get("OPENAI_API_KEY"))

# 获取 OpenAI GPT 可用模型列表，默认按 id 升序排序
def models(orderby='id'):
    try:
        models = client.models.list()
        data = [[model.id, model.created, model.object, model.owned_by]
                for model in models.data]
        df = pd.DataFrame(
            data, columns=['id', 'created', 'object', 'owned_by'])

        # 按 created 小到大排序
        if orderby == 'created':
            df = df.sort_values(by='created', ascending=True)

        # 将 created 时间戳转为 datetime 格式，单位秒
        df['created'] = pd.to_datetime(df['created'], unit='s')

        display(df[['id', 'created', 'object', 'owned_by']])
    except Exception as e:
        print(e)


# 按 id 排序
print("Models List(order by id)\n")
models()

# 按 created 排序
print("\nModels List(order by created)\n")
models('created')
