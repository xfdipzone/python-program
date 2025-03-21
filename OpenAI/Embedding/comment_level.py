# coding=utf-8
"""
测试 embedding 计算的评论分级准确率
基于给出的好评与差评，计算两者的 embedding 余弦相似度，计算评论分级
最后将计算的评论分级结果与真实评论分级比对，计算准确率

dependency packages
pip install numpy
pip install sentence-transformers
pip install pandas
pip install scikit-learn

测试结果：
              precision    recall  f1-score   support

    negative       0.78      0.75      0.76       136
    positive       0.96      0.96      0.96       789

    accuracy                           0.93       925
   macro avg       0.87      0.86      0.86       925
weighted avg       0.93      0.93      0.93       925
"""
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import PrecisionRecallDisplay
import numpy as np
import pandas as pd

# 使用 SentenceTransformer 加载计算文本向量模型
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

"""
计算文本的向量 (embedding)
"""
def get_embedding(text):
    # SentenceTransformer 的 encode 方法可以直接处理文本并返回嵌入向量
    return embed_model.encode(text)

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

def evaluate_embeddings_approach(labels = ['negative', 'positive']):
    label_embeddings = [get_embedding(label) for label in labels]

    # label_score 为 evaluate_embeddings_approach 的嵌套方法
    def label_score(review_embedding, label_embeddings):
        return cosine_similarity(review_embedding, label_embeddings[1]) - cosine_similarity(review_embedding, label_embeddings[0])

    probas = df["embedding"].apply(lambda x: label_score(x, label_embeddings))
    preds = probas.apply(lambda x: 'positive' if x>0 else 'negative')

    report = classification_report(df.sentiment, preds)
    print(report)

    display = PrecisionRecallDisplay.from_predictions(df.sentiment, probas, pos_label='positive')
    _ = display.ax_.set_title("2-class Precision-Recall curve")

datafile_path = "data/fine_food_reviews_with_embeddings_1k.csv"

# 只取 csv 中 n 条数据 pd.read_csv(datafile_path, nrows=n)
df = pd.read_csv(datafile_path)
df["embedding"] = df.combined.apply(lambda x: get_embedding(x))

# convert 5-star rating to binary sentiment
df = df[df.Score != 3]
df = df.assign(sentiment=df["Score"].replace({1: "negative", 2: "negative", 4: "positive", 5: "positive"}))

evaluate_embeddings_approach(labels=['An Amazon review with a negative sentiment.', 'An Amazon review with a positive sentiment.'])
