# coding=utf-8
from sentence_transformers import SentenceTransformer
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

# 计算多个文本的向量 (embedding)
def get_embeddings(list_of_texts, batch_size=32):
    return embed_model.encode(list_of_texts, batch_size=batch_size)


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
def search_product(index, df, queries, n=3, threshold=0.5):
    # 要搜索的产品名称的 Embedding，并归一化处理
    query_vectors = np.array(get_embeddings(queries)).astype("float32")
    faiss.normalize_L2(query_vectors)

    # 批量搜索，找出最接近的 n 个产品
    scores, indexes = index.search(query_vectors, n)

    # 构建结果，dict(query=>search results)
    results_dict = {}

    for i, query in enumerate(queries):
        results = pd.DataFrame(
            {
                "id": df.iloc[indexes[i]].index.values,
                "product_name": df.iloc[indexes[i]]["product_name"].values,
                "similarity": scores[i]
            }
        )

        # 过滤相似度，只保留大于 threshold 的结果
        results = results[results["similarity"] > threshold]
        results = results.sort_values("similarity", ascending=False)

        results_dict[query] = results

    return results_dict


# 打印结果
def print_results(results_dict):
    # 遍历结果字典键值对
    for query, results in results_dict.items():
        print(f"搜索：{query}")

        if results.empty:
            print(f"没有找到相似的产品")
        else:
            print(f"找到以下产品：")
            for _, row in results.iterrows():
                id = row["id"]
                name = row["product_name"]
                score = row["similarity"]
                print(f"产品编号：{id:2}｜相似度：{score:.4f}｜产品名称：{name}")
        print("-" * 80)


# 读取 parquet 文件到 Data Frame
parquet_datafile_path = "data/product_data.parquet"
df = pd.read_parquet(parquet_datafile_path)

# 将产品数据保存到 Faiss
index = load_embeddings_to_faiss(df)

# 相似度阈值
threshold_value = 0.5

queries = ["优雅自然女背包", "高颜值高性能手机"]
results_dict = search_product(index, df, queries, 3, threshold_value)
print_results(results_dict)

"""
搜索：优雅自然女背包
找到以下产品：
产品编号：55｜相似度：0.5790｜产品名称：【特卖专场】精美刺绣手提包 优雅女性首选 全场满减
产品编号：81｜相似度：0.5165｜产品名称：【新品首发】运动风背包 时尚休闲 买就送小礼品
产品编号：87｜相似度：0.5040｜产品名称：【限时抢购】流苏装饰斜挎包 时尚新潮 限时优惠
--------------------------------------------------------------------------------
搜索：高颜值高性能手机
找到以下产品：
产品编号： 6｜相似度：0.5706｜产品名称：【抢先体验】Realme GT2 Pro 12G+256G 5G高性能手机
产品编号： 1｜相似度：0.5633｜产品名称：【疯狂抢购】华为Mate 40 Pro 8G+256G 5G旗舰手机
产品编号：11｜相似度：0.5506｜产品名称：【特价促销】三星Note20 Ultra 12G+512G 5G手机
--------------------------------------------------------------------------------
"""
