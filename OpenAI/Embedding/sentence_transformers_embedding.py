# coding=utf-8
from sentence_transformers import SentenceTransformer

# 使用 SentenceTransformer 加载计算文本向量模型
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

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
    print("%s (dimensions %d)\nembedding : %s\n" % (text, embedding.shape[0], embedding[:5]))

"""
批量计算多个 text embedding
"""
embeddings = get_embeddings(list_of_texts)
print("num: %d\ndimensions: %d\n" % (embeddings.shape[0], embeddings.shape[1]))

for index, text in enumerate(list_of_texts):
    print("%s\nembedding : %s\n" % (text, embeddings[index][:5]))
