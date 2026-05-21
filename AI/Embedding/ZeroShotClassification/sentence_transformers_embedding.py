# coding=utf-8
from google.colab import userdata
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
import numpy as np

"""
使用 SentenceTransformer 计算文本的 Embedding 与维度

dependency packages
pip install sentence-transformers
pip install numpy
"""
# Login HuggingFace Hub
login(token=userdata.get("HF_TOKEN"))

# 加载 SentenceTransformer 的嵌入模型
embed_model = SentenceTransformer("google/embeddinggemma-300M")

# 计算文本的向量 (embedding)
def get_embedding(text):
    # SentenceTransformer 的 encode 方法可以直接处理文本并返回嵌入向量
    return embed_model.encode(text)


# 计算多个文本的向量 (embedding)
def get_embeddings(list_of_texts, batch_size=32):
    return embed_model.encode(list_of_texts, batch_size=batch_size)


# 文本数据
list_of_texts = [
    "【优惠】气质小清新拼接百搭双肩斜挎包",
    "【热卖】活力色彩精致小巧百搭女士单肩斜挎包",
    "【特价】简约可爱原宿风时尚双肩斜挎包",
    "【折扣】潮流小清新拼接百搭女士单肩斜挎包",
    "【特惠】百搭潮流活力色彩拼色双肩斜挎"
]

# 遍历 text 逐一计算 embedding
for text in list_of_texts:
    embedding = get_embedding(text)
    embeddings_np = np.array(embedding)
    print("%s (dimensions %d)\nembedding : %s\n" %
          (text, embeddings_np.shape[0], embedding[:5]))

# 批量计算多个 text embedding
embeddings = get_embeddings(list_of_texts)

# 将列表转换为 NumPy 数组以使用 .shape
embeddings_np = np.array(embeddings)
print("num: %d\ndimensions: %d\n" %
      (embeddings_np.shape[0], embeddings_np.shape[1]))

for index, text in enumerate(list_of_texts):
    print("%s\nembedding : %s\n" % (text, embeddings[index][:5]))
