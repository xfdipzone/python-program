# coding=utf-8
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import PrecisionRecallDisplay
from google.colab import userdata
from huggingface_hub import login
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
测试 embedding 计算的评论分级准确率
基于给出的好评与差评，计算两者的 embedding 余弦相似度，计算评论分级
最后将计算的评论分级结果与真实评论分级比对，计算准确率

dependency packages
pip install sentence-transformers
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib
"""
# Login HuggingFace Hub
login(token=userdata.get("HF_TOKEN"))

# 使用 SentenceTransformer 加载计算文本向量模型
embed_model = SentenceTransformer("google/embeddinggemma-300M")

# 计算文本的向量 (embedding)
def get_embedding(text, is_query=False):
    # Gemma 专用技巧：如果是用于对比的标签/查询，添加特定前缀效果更好
    if is_query:
        # 给标签类短语加上查询前缀，这是 Gemma 模型的标准用法
        text = f"Instruct: Retrieve semantically similar text.\nQuery: {text}"

    return embed_model.encode(text)


# 计算向量的余弦相似度
def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    epsilon = 1e-10
    cosine_similarity = dot_product / (norm_a * norm_b + epsilon)
    return cosine_similarity


# 评估零样本（Zero-shot）文本分类效果，通过 Embedding 相似度计算评论属于好评或差评
def evaluate_embeddings_approach(df, labels=['negative', 'positive']):
    label_embeddings = [get_embedding(label, is_query=True)
                        for label in labels]

    # label_score 为 evaluate_embeddings_approach 的嵌套方法
    def label_score(review_embedding, label_embeddings):
        return cosine_similarity(review_embedding, label_embeddings[1]) - cosine_similarity(review_embedding, label_embeddings[0])

    probas = df["embedding"].apply(lambda x: label_score(x, label_embeddings))
    preds = probas.apply(lambda x: 'positive' if x > -0.05 else 'negative')

    report = classification_report(df.sentiment, preds, labels=[
                                   'positive', 'negative'])
    print("\n测试结果：")
    print(report)

    print("\n好评率可视化评估")
    print("横轴 (Recall)：查全率，代表模型找回好评的覆盖面")
    print("纵轴 (Precision)：查准率，代表模型判定为好评的结果中有多少是真的\n")
    positive_display = PrecisionRecallDisplay.from_predictions(
        df.sentiment, probas, pos_label='positive')
    _ = positive_display.ax_.set_xlabel("Recall (Target: Positive)")
    _ = positive_display.ax_.set_ylabel("Precision (Target: Positive)")
    _ = positive_display.ax_.set_title(
        "Gemma-300M Positive Reviews Precision-Recall curve")
    plt.show()

    print("\n差评率可视化评估")
    print("横轴 (Recall)：查全率，代表模型找回差评的覆盖面")
    print("纵轴 (Precision)：查准率，代表模型判定为差评的结果中有多少是真的\n")
    negative_display = PrecisionRecallDisplay.from_predictions(
        df.sentiment, -probas, pos_label='negative')
    _ = negative_display.ax_.set_xlabel("Recall (Target: Negative)")
    _ = negative_display.ax_.set_ylabel("Precision (Target: Negative)")
    _ = negative_display.ax_.set_title(
        "Gemma-300M Negative Reviews Precision-Recall curve")
    plt.show()


# 数据源
datafile_path = "data/fine_food_reviews_with_embeddings_1k.csv"

# 只取 csv 中 n 条数据 pd.read_csv(datafile_path, nrows=n)
df = pd.read_csv(datafile_path)

# 批量计算 Embedding（正文不需要加 Query 指令前缀）
df["embedding"] = df.combined.apply(lambda x: get_embedding(x, is_query=False))

# convert 5-star rating to binary sentiment
df = df[df.Score != 3]
df = df.assign(sentiment=df["Score"].replace(
    {1: "negative", 2: "negative", 4: "positive", 5: "positive"}))

evaluate_embeddings_approach(df, labels=[
                             'A very disappointed and angry food review with many complaints.',
                             'A very happy and satisfied food review praising the product.'])

"""
测试结果：
              precision    recall  f1-score   support

    positive       0.97      0.98      0.97       789
    negative       0.85      0.82      0.83       136

    accuracy                           0.95       925
   macro avg       0.91      0.90      0.90       925
weighted avg       0.95      0.95      0.95       925
"""
