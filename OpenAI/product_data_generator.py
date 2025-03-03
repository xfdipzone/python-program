# coding=utf-8
from openai import OpenAI
import os
import pandas as pd

client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))

"""
产品数据生成器
"""
class ProductDataGenerator:
    def __init__(self):
        self.model = "gpt-4o-mini"

    def generate(self, prompt):
        try:
            completions = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
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
        total_df = total_df.reset_index(drop=True)

# 显示数据集
total_df.index.name = 'product_id'
display(total_df)
