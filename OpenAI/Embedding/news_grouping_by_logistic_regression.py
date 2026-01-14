# coding=utf-8
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, log_loss, roc_auc_score
import pandas as pd
import numpy as np

"""
通过逻辑回归算法训练预测新闻类别

dependency packages
pip install scikit-learn
pip install pandas
pip install numpy
"""
# 读取样本数据文件
training_data = pd.read_parquet("data/20_newsgroup_with_embedding.parquet")

# 验证数据
print("样本数据概览：")
print(f"数据集形状: {training_data.shape}")
print(f"列名: {training_data.columns.tolist()}")
print(f"类别数量: {training_data['target'].nunique()}\n")

# 从样本中随机抽取 10000 条记录用于训练于测试
# random_state 是随机乱数，用于保证每次随机抽取的记录一致
df = training_data.sample(10000, random_state=42)

# 划分训练集与测试集（80% 的数据作为训练集，20% 的数据作为测试集）
X_train, X_test, y_train, y_test = train_test_split(
    list(df.embedding.values), df.target, test_size=0.2, random_state=42
)

# 创建逻辑回归分类器
clf = LogisticRegression()

# 训练
clf.fit(X_train, y_train)

# 预测
preds = clf.predict(X_test)

report = classification_report(y_test, preds)
print("训练结果：")
print(report)

# 每个类别的概率（用于置信度分析）
probas = clf.predict_proba(X_test)

print("更多分析：")

# 对数损失
logloss = log_loss(y_test, probas)
print(f"对数损失：{logloss:.4f}")

# 置信度分析
# 形状（样本数，类别数），用 axis=1
# 形状（类别数，样本数），用 axis=0
confidences = np.max(probas, axis=1)
print(f"平均置信度：{confidences.mean():.4f}")
print(f"低置信度（<0.6）样本比例：{(confidences<0.6).mean():.2%}")

"""
样本数据概览：
数据集形状: (10640, 5)
列名: ['text', 'target', 'title', 'n_tokens', 'embedding']
类别数量: 20

训练结果：
              precision    recall  f1-score   support

           0       0.56      0.63      0.59        79
           1       0.70      0.76      0.73        98
           2       0.71      0.75      0.73       119
           3       0.65      0.69      0.67       114
           4       0.81      0.65      0.72       114
           5       0.90      0.86      0.88       108
           6       0.81      0.82      0.81       109
           7       0.80      0.81      0.80       106
           8       0.78      0.73      0.75       104
           9       0.78      0.94      0.85       108
          10       0.96      0.88      0.92        91
          11       0.88      0.74      0.80       106
          12       0.75      0.75      0.75       102
          13       0.83      0.88      0.85       104
          14       0.77      0.84      0.80        95
          15       0.77      0.82      0.80        97
          16       0.63      0.81      0.71        91
          17       0.88      0.84      0.86       100
          18       0.62      0.66      0.64        89
          19       0.60      0.14      0.22        66

    accuracy                           0.76      2000
   macro avg       0.76      0.75      0.74      2000
weighted avg       0.76      0.76      0.76      2000

更多分析：
对数损失：1.0094
平均置信度：0.5150
低置信度（<0.6）样本比例：60.10%

运行用时：13s
"""
