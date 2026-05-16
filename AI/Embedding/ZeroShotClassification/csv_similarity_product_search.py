# coding=utf-8
from sentence_transformers import SentenceTransformer
from google.colab import userdata
from huggingface_hub import login
from scipy.spatial.distance import cosine
import pandas as pd
import numpy as np

"""
搜索相似产品

dependency packages
pip install sentence-transformers
pip install scipy
pip install pandas
pip install numpy
"""
# Login HuggingFace Hub
login(token=userdata.get("HF_TOKEN"))

# 使用 SentenceTransformer 加载计算文本向量模型
embed_model = SentenceTransformer("google/embeddinggemma-300M")

# 计算文本的向量 (embedding)
def get_embedding(text):
    return embed_model.encode(text)


# 计算向量的余弦相似度
def cosine_similarity(vector_a, vector_b):
    # 注意：scipy 计算的是距离，相似度 = 1 - 距离
    return 1 - cosine(vector_a, vector_b)


# 将 csv 中 Embedding 字符串转为数组（Embeddings 使用空格分隔）
def parse_embedding(embeddings_string):
    return np.fromstring(embeddings_string.strip('[]'), sep=' ', dtype=np.float32)


# 搜索产品
def search_product(df, query, n=3, threshold=0.5):
    # 要搜索的产品名称的 Embedding
    product_embedding = get_embedding(query)

    # 计算搜索的产品与每一个产品 Embedding 的余弦相似度
    df["similarity"] = df.embedding.apply(
        lambda x: cosine_similarity(x, product_embedding))

    # 过滤相似度，只保留大于 threshold 的结果
    filtered_df = df[df["similarity"] > threshold]

    results = (
        filtered_df.sort_values("similarity", ascending=False)
        .head(n)
    )

    return results


# 打印结果
def print_results(results):
    if results.empty:
        print(f"没有找到相似的产品")
    else:
        print(f"找到以下产品：")
        for idx, row in results.iterrows():
            name = row["product_name"]
            score = row["similarity"]
            print(f"标号：{idx:2}｜相似度：{score:.4f}｜产品名称：{name}")
    print("-" * 80)


# 读取 csv 文件到 Data Frame
csv_datafile_path = "data/product_data.csv"
df = pd.read_csv(csv_datafile_path)

df["embedding"] = df["embedding"].apply(parse_embedding)

# 相似度阈值
threshold_value = 0.5

query = "优雅自然女背包"
results = search_product(df, query, n=3, threshold=threshold_value)
print(f"搜索：{query}")
print_results(results)

query = "高颜值高性能手机"
results = search_product(df, query, n=3, threshold=threshold_value)
print(f"搜索：{query}")
print_results(results)

"""
搜索：优雅自然女背包
找到以下产品：
标号：55｜相似度：0.5790｜产品名称：【特卖专场】精美刺绣手提包 优雅女性首选 全场满减
标号：81｜相似度：0.5165｜产品名称：【新品首发】运动风背包 时尚休闲 买就送小礼品
标号：87｜相似度：0.5040｜产品名称：【限时抢购】流苏装饰斜挎包 时尚新潮 限时优惠
--------------------------------------------------------------------------------
搜索：高颜值高性能手机
找到以下产品：
标号： 6｜相似度：0.5706｜产品名称：【抢先体验】Realme GT2 Pro 12G+256G 5G高性能手机
标号： 1｜相似度：0.5633｜产品名称：【疯狂抢购】华为Mate 40 Pro 8G+256G 5G旗舰手机
标号：11｜相似度：0.5506｜产品名称：【特价促销】三星Note20 Ultra 12G+512G 5G手机
--------------------------------------------------------------------------------
"""
