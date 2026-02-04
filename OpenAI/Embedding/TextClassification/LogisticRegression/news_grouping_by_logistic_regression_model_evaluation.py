# coding=utf-8
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, log_loss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
评估基于逻辑回归算法训练预测新闻类别模型
使用混沌矩阵（Confusion Matrix）可视化辅助模型评估

dependency packages
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
X_train, X_test, y_train, y_test = train_test_split(
    list(df.embedding.values), df.target, test_size=0.2, random_state=42
)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. 模型定义与训练
# ==========================================
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

# ==========================================
# 4. 预测与评估报告
# ==========================================
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

# ==========================================
# 5. 混淆矩阵可视化 (分析类别 19)
# ==========================================
cm = confusion_matrix(y_test, preds)
# 归一化处理
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(15, 12))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlGnBu",
            xticklabels=range(20), yticklabels=range(20))
plt.title("Logistic Regression Normalized Confusion Matrix (Recall by Class)")
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
