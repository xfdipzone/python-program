# coding=utf-8
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, log_loss, roc_auc_score
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

"""
PCA（Principal Component Analysis）主成分分析，用于分析数据特征
PCA Variance - Automated Detection 根据阈值获取对应的维度数

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

# ==========================================
# PCA 使用碎石图分析阈值对应的特征维数
# ==========================================
# 设置你想要的阈值
threshold = 0.95

# 拟合不设限的 PCA
pca_full = PCA().fit(X_train)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# 核心：全自动寻找交点
# 找到第一个累积方差大于等于 threshold 的索引 (索引从0开始，所以要+1)
n_components_found = np.where(cumulative_variance >= threshold)[0][0] + 1
actual_variance = cumulative_variance[n_components_found - 1]

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1),
         cumulative_variance, color='#1f77b4', lw=2)

# 自动画线
plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.6)  # 自动画出阈值水平线
plt.axvline(x=n_components_found, color='g',
            linestyle='--', alpha=0.6)  # 自动画出对应的维度垂直线

# 自动打点标记交点
plt.scatter(n_components_found, actual_variance, color='red', s=50, zorder=5)
plt.annotate(f'Point: ({n_components_found} dims, {actual_variance:.2%})',
             xy=(n_components_found, actual_variance),
             xytext=(n_components_found + 50, actual_variance - 0.05),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

plt.title('PCA Variance - Automated Detection', fontsize=14)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True, linestyle=':', alpha=0.5)
plt.show()

print(f"根据 {threshold:.0%} 的要求，系统自动建议的维度数量为: {n_components_found}\n")

# ==========================================
# PCA 直接计算阈值对应的维度数
# ==========================================
print(f"原始特征维度: {X_train.shape[1]}")

pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

print(f"PCA 降维后维度: {X_train_pca.shape[1]}")
print(f"保留的累计方差贡献率: {sum(pca.explained_variance_ratio_):.2%}")
