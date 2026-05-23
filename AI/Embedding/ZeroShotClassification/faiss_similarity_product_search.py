# coding=utf-8
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from google.colab import userdata
from huggingface_hub import login
import pandas as pd
import numpy as np
import faiss

"""
基于 Faiss (Facebook AI Similarity Search) 搜索相似产品

dependency packages
pip install sentence-transformers
pip install pandas
pip install numpy
pip install pyarrow
pip install fastparquet
pip install faiss-gpu-cu12
"""
# Login HuggingFace Hub
login(token=userdata.get("HF_TOKEN"))

# 使用 SentenceTransformer 加载计算文本向量模型
embed_model = SentenceTransformer("google/embeddinggemma-300M")

# 计算文本的向量 (embedding)
def get_embedding(text):
    return embed_model.encode(text)


# 将产品数据保存到 Faiss
def load_embeddings_to_faiss(df):
    # 将 embedding 列转为 numpy 矩阵
    embeddings = np.stack(df["embedding"].values).astype("float32")

    # Faiss 关键点：余弦相似度必须归一化
    # Faiss 的 Inner Product(IP) 索引在向量归一化后等于同于余弦相似度
    faiss.normalize_L2(embeddings)

    # 创建内积索引
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


# 搜索产品
def search_product(index, df, query, n=3, threshold=0.5):
    # 要搜索的产品名称的 Embedding，并归一化处理
    query_vector = np.array(get_embedding(query)).astype(
        "float32").reshape(1, -1)
    faiss.normalize_L2(query_vector)

    # 搜索，找出最接近的 n 个产品
    scores, indexes = index.search(query_vector, n)

    # 构建结果
    results = pd.DataFrame(
        {
            "product_name": df.iloc[indexes[0]]["product_name"].values,
            "similarity": scores[0]
        }
    )

    # 过滤相似度，只保留大于 threshold 的结果
    results = results[results["similarity"] > threshold]
    results = results.sort_values("similarity", ascending=False)

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


# 读取 parquet 文件到 Data Frame
parquet_datafile_path = "data/product_data.parquet"
df = pd.read_parquet(parquet_datafile_path)

# 将产品数据保存到 Faiss
index = load_embeddings_to_faiss(df)

# 相似度阈值
threshold_value = 0.5

query = "优雅自然女背包"
results = search_product(index, df, query, n=3, threshold=threshold_value)
print(f"搜索：{query}")
print_results(results)

query = "高颜值高性能手机"
results = search_product(index, df, query, n=3, threshold=threshold_value)
print(f"搜索：{query}")
print_results(results)
