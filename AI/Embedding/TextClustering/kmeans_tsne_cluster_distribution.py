# coding=utf-8
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
使用 t-SNE 非线形降维技术，可视化 K-Means 聚类分布

t-SNE（t-distributed Stochastic Neighbor Embedding）是最常用的非线性降维技术。
它能很好地保留数据的局部结构，让聚类结果在 2D 平面上现出原形。

dependency packages
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install pandas
pip install numpy
"""
# ==========================================
# 1. 生成文本聚类
# ==========================================
# 读取数据文件
embedding_df = pd.read_parquet("data/20_newsgroup_with_embedding.parquet")

# 移除文本为空、或只包含换行符/空格的行
embedding_df = embedding_df[embedding_df['text'].str.strip().astype(bool)]

# 移除字数太少的文本，比如少于 10 个字符的
embedding_df = embedding_df[embedding_df['text'].str.len() > 10]

# 准备特征矩阵
matrix = np.vstack(embedding_df.embedding.values)

# 聚类数量
num_of_clusters = 20

# 创建并训练 K-Means 模型
kmeans = KMeans(
    n_clusters=num_of_clusters,  # 聚为 20 个类
    init="k-means++",            # 使用 k-means++ 智能初始化中心点，加速收敛
    n_init=15,                   # 运行 15 次，选最优结果
    random_state=42              # 固定随机种子，保证结果可复现
)

# 在 Embedding 上训练聚类模型
kmeans.fit(matrix)

# 获取聚类结果
labels = kmeans.labels_
embedding_df["cluster"] = labels


# ==========================================
# 2. 生成可视化 K-Means 聚类分布
# ==========================================
# 执行 t-SNE 降维
# perplexity 建议设在 30-50 之间，n_iter 决定收敛程度
tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    init='pca',
    learning_rate='auto'
)

# 降维过程较慢，建议只取前几千个样本，或者直接对全量矩阵执行
vis_dims = tsne.fit_transform(matrix)

# 将降维结果存入 DataFrame
embedding_df['x'] = vis_dims[:, 0]
embedding_df['y'] = vis_dims[:, 1]

# 绘图可视化
plt.figure(figsize=(12, 8))

# 使用 Seaborn 绘制散点图，用 cluster 染色
sns.scatterplot(
    data=embedding_df,
    x='x', y='y',
    hue='cluster',
    palette='viridis',
    legend='full',
    alpha=0.6,
    edgecolor=None
)

plt.title('20 Newsgroups Cluster Distribution (t-SNE)', fontsize=15)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
