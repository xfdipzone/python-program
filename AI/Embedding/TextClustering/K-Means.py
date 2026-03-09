# coding=utf-8
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

"""
通过 K-Means 算法实现文本聚类

dependency packages
pip install scikit-learn
pip install pandas
pip install numpy
"""
# ==========================================
# 1. 生成文本聚类
# ==========================================
# 读取数据文件
embedding_df = pd.read_parquet("data/20_newsgroup_with_embedding.parquet")

# 准备特征矩阵
matrix = np.vstack(embedding_df.embedding.values)

# 聚为 20 个类
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

# 格式化百分比
new_df['rank1_per'] = (
    new_df['rank1_count'] / new_df['count']).map(lambda x: '{:.2%}'.format(x))

new_df['rank2_per'] = (
    new_df['rank2_count'] / new_df['count']).map(lambda x: '{:.2%}'.format(x))

# 将缺失值替换为 0
new_df.fillna(0, inplace=True)

# 按 rank1_per 从高到低排序
new_df['sort_val'] = new_df['rank1_count'] / new_df['count']
new_df = new_df.sort_values(
    by='sort_val', ascending=False).drop(columns=['sort_val'])

# 输出结果
display(new_df)
