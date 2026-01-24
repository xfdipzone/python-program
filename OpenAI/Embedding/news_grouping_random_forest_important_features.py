# coding=utf-8
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

"""
随机森林模型重要特征分析

dependency packages
pip install scikit-learn
pip install pandas
pip install numpy
"""
# 读取样本数据文件
training_data = pd.read_parquet("data/20_newsgroup_with_embedding.parquet")

# 从样本中随机抽取 10000 条记录用于训练与测试
# random_state 是随机乱数，用于保证每次随机抽取的记录一致
df = training_data.sample(10000, random_state=42)

# 划分训练集与测试集（80% 的数据作为训练集，20% 的数据作为测试集）
X_train, X_test, y_train, y_test = train_test_split(
    list(df.embedding.values), df.target, test_size=0.2, random_state=42
)

# 使用原始数据训练
clf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# ==========================================
# 获取特征重要性及计算 top 100 特征重要性总和
# ==========================================
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

# 打印前 25 个最重要的原始特征维度
print("原始特征重要性排行")

for i in range(25):
    print(f"原始特征 {indices[i]:4}: 重要性评分 {importances[indices[i]]:.4f}")

# 计算 top 100 特征重要性总和（总和达到 0.5 以上表示特征非常集中）
top100_total_importances = 0
for i in range(100):
    top100_total_importances += importances[indices[i]]

print(f"\nTop 100 特征重要性评分总和: {top100_total_importances:.4f}\n\n")
