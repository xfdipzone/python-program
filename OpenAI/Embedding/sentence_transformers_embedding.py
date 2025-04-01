# coding=utf-8
"""
dependency packages
pip install numpy
pip install sentence-transformers
"""
from sentence_transformers import SentenceTransformer
import numpy as np

# 使用 SentenceTransformer 加载计算文本向量模型
embed_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

"""
计算文本的向量 (embedding)
"""
def get_embedding(text):
    # SentenceTransformer 的 encode 方法可以直接处理文本并返回嵌入向量
    return embed_model.encode(text)


"""
计算多个文本的向量 (embedding)
"""
def get_embeddings(list_of_texts, batch_size=32):
    return embed_model.encode(list_of_texts, batch_size=batch_size)


"""
计算向量的余弦相似度
"""
def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    epsilon = 1e-10
    cosine_similarity = dot_product / (norm_a * norm_b + epsilon)
    return cosine_similarity


list_of_texts = [
    "【优惠】气质小清新拼接百搭双肩斜挎包",
    "【热卖】活力色彩精致小巧百搭女士单肩斜挎包",
    "【特价】简约可爱原宿风时尚双肩斜挎包",
    "【折扣】潮流小清新拼接百搭女士单肩斜挎包",
    "【特惠】百搭潮流活力色彩拼色双肩斜挎"
]

"""
遍历 text 逐一计算 embedding
"""
for text in list_of_texts:
    embedding = get_embedding(text)
    print("%s (dimensions %d)\nembedding : %s\n" %
          (text, embedding.shape[0], embedding[:5]))

"""
批量计算多个 text embedding
"""
embeddings = get_embeddings(list_of_texts)
print("num: %d\ndimensions: %d\n" % (embeddings.shape[0], embeddings.shape[1]))

for index, text in enumerate(list_of_texts):
    print("%s\nembedding : %s\n" % (text, embeddings[index][:5]))

"""
计算文本与搜索词的余弦相似度
"""
search = "自然优雅挎包"
search_embedding = get_embedding(search)

# 保存搜索词与文本的余弦相似度
similarities = []

for index, text in enumerate(list_of_texts):
    sim = cosine_similarity(search_embedding, embeddings[index])
    similarities.append((text, sim))

# 按余弦相似度高到低排序
sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

print("%s 与文本相似度排序\n" % search)

for text, sim in sorted_similarities:
    print("%s\nsimilarity : %f\n" % (text, sim))
