# coding=utf-8
from google import genai
from google.colab import userdata
import pandas as pd

"""
设置全局 pd option
可选项列表 pd.describe_option()
设置可选项 pd.set_option(key, value)
重置可选项 pd.reset_option(key)
重置所有可选项 pd.reset_option('all')

设置临时可选项（只对当前的 display 有效）
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(df[['id', 'created', 'object', 'owned_by']])
"""
# 显示所有行
pd.set_option('display.max_rows', None)

# 显示所有列
pd.set_option('display.max_columns', None)

# 单元格内容完整显示，不截断
pd.set_option('display.max_colwidth', None)

# 提高换行阈值，防止自动换行
pd.set_option('display.width', 1000)

"""
获取 Google Gemini 支持的模型列表

dependency packages
pip install google-genai
pip install pandas
"""
client = genai.Client(api_key=userdata.get("GOOGLE_API_KEY"))

# 获取 Google Gemini 可用模型列表，默认按 id 升序排序
def models(orderby='id'):
    try:
        models = client.models.list()
        data = [[model.name, model.version, model.supported_actions, model.description]
                for model in models]

        df = pd.DataFrame(
            data, columns=['name', 'version', 'supported actions', 'description'])

        # 按 name 升序排序
        if orderby == 'name':
            df = df.sort_values(by='name', ascending=True)

        display(df[['name', 'version', 'supported actions', 'description']])
    except Exception as e:
        print(e)


# 按 id 排序
print("Models List(order by id)\n")
models()

# 按 name 排序
print("\nModels List(order by name)\n")
models('name')
