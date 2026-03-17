# coding=utf-8
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
使用 t-SNE 非线形降维技术，可视化 K-Means 聚类分布（每个簇自动打上主题标签 Label）

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
# 2. 统计聚类数据
# ==========================================
# 预计算每个 cluster 的数量
new_df = embedding_df.groupby(
    'cluster')['cluster'].count().reset_index(name='count')

# 统计 cluster + title 的组合分布，并按数量倒序排列（最多的分类数量）
title_counts = (
    embedding_df.groupby(['cluster', 'title'])
    .size()
    .reset_index(name='t_count')
    .sort_values(['cluster', 't_count'], ascending=[True, False])
)

# 提取 rank1（每个分组的第一行）
rank1 = title_counts.groupby('cluster').head(1).copy()
rank1 = rank1.rename(columns={'title': 'rank1', 't_count': 'rank1_count'})

# 提取 rank2（每个分组的第二行）
rank2 = title_counts.groupby('cluster').nth(1)
rank2 = rank2.rename(columns={'title': 'rank2', 't_count': 'rank2_count'})

# 合并结果
new_df = new_df.merge(rank1, on='cluster', how='left')
new_df = new_df.merge(rank2, on='cluster', how='left')


# ==========================================
# 3. 生成可视化 K-Means 聚类分布
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

# 计算每个簇在 2D 空间中的中心点（质心）
# 我们利用之前生成的 embedding_df 中的 x, y 坐标
centers = embedding_df.groupby('cluster')[['x', 'y']].mean().reset_index()

# 将之前统计好的主题名称 (rank1) 合并进来
# 假设你之前的统计结果存放在 new_df 中
centers = centers.merge(new_df[['cluster', 'rank1']], on='cluster')

# 开始绘图
plt.figure(figsize=(14, 10))

# 绘制底层的散点
sns.scatterplot(
    data=embedding_df,
    x='x', y='y',
    hue='cluster',
    palette='viridis',
    legend=None,  # 标签已经打在图上了，不需要侧边图例
    alpha=0.4,
    edgecolor=None
)

# 在质心位置添加文本标签
for i, row in centers.iterrows():
    plt.text(
        row['x'], row['y'],
        f"{int(row['cluster'])}: {row['rank1']}",
        fontsize=10,
        fontweight='bold',
        ha='center',  # 水平居中
        va='center',  # 垂直居中
        bbox=dict(facecolor='white', alpha=0.7,
                  edgecolor='none', boxstyle='round,pad=0.3')
    )

plt.title('20 Newsgroups Clustering with Topic Labels', fontsize=15)
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()
