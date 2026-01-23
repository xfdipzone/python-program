# coding=utf-8
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, log_loss, roc_auc_score
import pandas as pd
import numpy as np

"""
通过随机森林算法训练预测新闻类别

dependency packages
pip install scikit-learn
pip install pandas
pip install numpy

流程：
1.读取源数据文件，随机取出 10000 条记录
2.拆分训练集与测试集（80:20）
3.创建随机森林分类器
4.训练
5.预测
6.打印预测报告
7.打印对数损失，平均置信度，低置信度（<0.6）样本比例
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

# 创建随机森林分类器（包含 300 棵决策树）
clf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
)

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

           0       0.55      0.52      0.54        79
           1       0.65      0.69      0.67        98
           2       0.69      0.70      0.69       119
           3       0.58      0.67      0.62       114
           4       0.77      0.61      0.68       114
           5       0.86      0.86      0.86       108
           6       0.82      0.82      0.82       109
           7       0.83      0.83      0.83       106
           8       0.74      0.75      0.74       104
           9       0.75      0.91      0.82       108
          10       0.89      0.85      0.87        91
          11       0.89      0.69      0.78       106
          12       0.72      0.67      0.69       102
          13       0.79      0.85      0.82       104
          14       0.69      0.82      0.75        95
          15       0.65      0.89      0.75        97
          16       0.61      0.81      0.69        91
          17       0.84      0.84      0.84       100
          18       0.67      0.57      0.62        89
          19       0.80      0.06      0.11        66

    accuracy                           0.73      2000
   macro avg       0.74      0.72      0.71      2000
weighted avg       0.74      0.73      0.72      2000

更多分析：
对数损失：1.6674
平均置信度：0.2672
低置信度（<0.6）样本比例：97.70%

运行时间：2min 13s
"""
