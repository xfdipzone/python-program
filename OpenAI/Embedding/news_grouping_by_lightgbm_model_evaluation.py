# coding=utf-8
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, log_loss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
评估基于梯度提升决策树（LightGBM）算法训练预测新闻类别模型（Light Gradient Boosting Machine）
使用混沌矩阵（Confusion Matrix）可视化辅助模型评估

dependency packages
pip install lightgbm
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install pyarrow
pip install fastparquet
"""
# ==========================================
# 1. 加载数据
# ==========================================
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

# ==========================================
# 2. 数据预处理
# ==========================================
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

# ==========================================
# 3. 模型定义与训练
# ==========================================
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
        log_evaluation(period=0)  # 控制打印训练日志的频率（0:不打印 N:每经过 N 轮打印一次）
    ]
)

# ==========================================
# 4. 预测与评估报告
# ==========================================
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

# ==========================================
# 5. 混淆矩阵可视化 (分析类别 19)
# ==========================================
cm = confusion_matrix(y_test, preds)
# 归一化处理
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(15, 12))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlGnBu",
            xticklabels=range(20), yticklabels=range(20))
plt.title("LightGBM Normalized Confusion Matrix (Recall by Class)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# 输出类别 19 的具体误判情况
target_class = 19
if target_class in y_test.values:
    print(f"\n分析类别 {target_class}：")
    top_error_indices = np.argsort(cm[target_class])[::-1]
    for idx in top_error_indices:
        if idx != target_class and cm[target_class][idx] > 0:
            percent = (cm[target_class][idx] / cm[target_class].sum()) * 100
            print(
                f"被误判为类别 {idx:2} 的样本数: {cm[target_class][idx]:2} ({percent:.1f}%)")
