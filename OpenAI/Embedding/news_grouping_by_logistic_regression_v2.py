# coding=utf-8
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, log_loss, roc_auc_score
import pandas as pd
import numpy as np

"""
通过逻辑回归算法训练预测新闻类别 V2
增加参数调优

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

# 从样本中随机抽取 10000 条记录用于训练与测试
# random_state 是随机乱数，用于保证每次随机抽取的记录一致
df = training_data.sample(10000, random_state=42)

# 划分训练集与测试集（80% 的数据作为训练集，20% 的数据作为测试集）
X_train, X_test, y_train, y_test = train_test_split(
    list(df.embedding.values), df.target, test_size=0.2, random_state=42
)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 参数调优
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'class_weight': [None, 'balanced'],
}

# 创建逻辑回归分类器
clf = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
)

# 训练
clf.fit(X_train_scaled, y_train)

# 预测
preds = clf.predict(X_test_scaled)

report = classification_report(y_test, preds)
print("训练结果：")
print(report)

# 每个类别的概率（用于置信度分析）
probas = clf.predict_proba(X_test_scaled)

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

           0       0.61      0.62      0.62        79
           1       0.69      0.77      0.72        98
           2       0.67      0.76      0.71       119
           3       0.67      0.69      0.68       114
           4       0.82      0.67      0.73       114
           5       0.92      0.87      0.90       108
           6       0.83      0.82      0.82       109
           7       0.88      0.86      0.87       106
           8       0.85      0.75      0.80       104
           9       0.80      0.94      0.87       108
          10       0.94      0.90      0.92        91
          11       0.84      0.74      0.78       106
          12       0.70      0.74      0.72       102
          13       0.91      0.84      0.87       104
          14       0.78      0.86      0.82        95
          15       0.80      0.85      0.82        97
          16       0.68      0.77      0.72        91
          17       0.87      0.87      0.87       100
          18       0.65      0.74      0.69        89
          19       0.67      0.39      0.50        66

    accuracy                           0.78      2000
   macro avg       0.78      0.77      0.77      2000
weighted avg       0.78      0.78      0.78      2000

更多分析：
对数损失：0.7301
平均置信度：0.8070
低置信度（<0.6）样本比例：20.65%

运行时间：1min 48s
"""
