# coding=utf-8
from openai import OpenAI
from google.colab import userdata
import pandas as pd
import numpy as np

# 显示所有行
pd.set_option('display.max_rows', None)

# 显示所有列
pd.set_option('display.max_columns', None)

# 单元格内容完整显示，不截断
pd.set_option('display.max_colwidth', None)

# 提高换行阈值，防止自动换行
pd.set_option('display.width', 1000)

"""
AI 产品数据生成器

dependency packages
pip install openai
pip install pandas
pip install numpy
"""
client = OpenAI(
    api_key=userdata.get("KIMI_API_KEY"),
    base_url="https://api.moonshot.cn/v1"
)

COMPLETION_MODEL = "moonshot-v1-8k"

"""
产品数据生成器
"""
class ProductDataGenerator:
    def __init__(self):
        self.model = COMPLETION_MODEL

    def generate(self, prompt):
        try:
            completions = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个熟悉中国电商平台（淘宝、京东等）的运营专家，擅长撰写吸引消费者的商品标题，了解各品类的促销话术和用户心理。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                n=1,
                stop=None,
                temperature=0.5
            )
        except Exception as e:
            print(e)
            return e

        message = completions.choices[0].message
        return message.content


"""
清洗生成的产品数据
"""
def format_product_data(data):
    # 将数据放入 data frame
    product_names = data.strip().split('\n')
    df = pd.DataFrame({'product_name': product_names})

    # 去除返回结果的标号
    df.product_name = df.product_name.apply(lambda x: x.split('.')[1].strip())
    return df


# 提示词
prompts = [
    """请你生成50条淘宝网里的商品的标题，每条在30个字左右，品类是3C数码产品，标题里往往也会有一些促销类的信息，每行一条。""",
    """请你生成50条淘宝网里的商品的标题，每条在30个字左右，品类是女性的服饰箱包等等，标题里往往也会有一些促销类的信息，每行一条。"""
]

# Data Frames
total_df = None

# 生成产品数据
generator = ProductDataGenerator()

for prompt in prompts:
    product_data = generator.generate(prompt)

    df = format_product_data(product_data)
    if total_df is None:
        total_df = df
    else:
        total_df = pd.concat([total_df, df], axis=0)

# 解决索引重复
total_df = total_df.reset_index(drop=True)

# 索引从 1 开始
total_df.index = np.arange(1, len(total_df) + 1)

# 显示数据集
display(total_df)
