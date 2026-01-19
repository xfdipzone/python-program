# coding=utf-8
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, log_loss, roc_auc_score
import pandas as pd
import numpy as np

"""
通过梯度提升决策树（LightGBM）算法训练预测新闻类别（Light Gradient Boosting Machine）

dependency packages
pip install lightgbm
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
X_train_val, X_test, y_train_val, y_test = train_test_split(
    list(df.embedding.values), df.target, test_size=0.2, random_state=42
)

# 从训练集中分割出训练集与验证集（90% 的数据作为训练集，10% 的数据作为验证集）
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.125, random_state=42
)

# 将列表转换为适合 LightGBM 的格式（如果 embedding 已经是 numpy 数组格式，则不需要转换）
X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)

# 获取特征维度并生成列名列表
feature_dim = X_train.shape[1]
feature_names = [f"f_{i}" for i in range(feature_dim)]

# 转换为 DataFrame 并指定列名
X_train = pd.DataFrame(X_train, columns=feature_names)
X_val = pd.DataFrame(X_val, columns=feature_names)
X_test = pd.DataFrame(X_test, columns=feature_names)

# 创建梯度提升决策树（LightGBM）分类器
clf = lgb.LGBMClassifier(
    n_estimators=500,         # 树的数量
    learning_rate=0.05,       # 学习率
    max_depth=5,              # 树的最大深度
    num_leaves=28,            # 叶子节点数
    feature_fraction=0.1,     # 使用的向量维度
    class_weight="balanced",  # 平衡类别样本
    bagging_fraction=0.8,     # 样本采样比例
    bagging_freq=5,           # 采样频率
    lambda_l1=0.1,            # L1 正则化，让不重要的特征权重彻底归零
    lambda_l2=0.1,            # L2 正则化，防止某个特征权重过大
    random_state=42,          # 随机种子
    n_jobs=-1,                # 使用所有 CPU 核心
    verbose=-1                # 不输出训练日志
)

# 训练
clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='multi_logloss',
    callbacks=[
        early_stopping(stopping_rounds=20),
        log_evaluation(period=0)
    ]
)

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

Training until validation scores don't improve for 20 rounds
Early stopping, best iteration is:
[330]	valid_0's multi_logloss: 0.760011
训练结果：
              precision    recall  f1-score   support

           0       0.58      0.66      0.62        79
           1       0.69      0.78      0.73        98
           2       0.76      0.79      0.78       119
           3       0.66      0.69      0.68       114
           4       0.76      0.68      0.72       114
           5       0.88      0.84      0.86       108
           6       0.82      0.81      0.81       109
           7       0.87      0.80      0.83       106
           8       0.76      0.76      0.76       104
           9       0.84      0.94      0.89       108
          10       0.93      0.89      0.91        91
          11       0.87      0.71      0.78       106
          12       0.75      0.73      0.74       102
          13       0.88      0.88      0.88       104
          14       0.74      0.82      0.78        95
          15       0.78      0.82      0.80        97
          16       0.69      0.78      0.73        91
          17       0.89      0.87      0.88       100
          18       0.61      0.73      0.67        89
          19       0.56      0.23      0.32        66

    accuracy                           0.77      2000
   macro avg       0.77      0.76      0.76      2000
weighted avg       0.77      0.77      0.77      2000

更多分析：
对数损失：0.7856
平均置信度：0.7959
低置信度（<0.6）样本比例：22.65%

运行时间：2min 14s
"""
