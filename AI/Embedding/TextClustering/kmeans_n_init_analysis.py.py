# coding=utf-8
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

"""
分析 K-Means 算法 n_init 参数影响 Inertia（惯性）的变化曲线

用于找出较稳定且性价比高的 n_init 参数 (trade off)

KMeans 容易陷入局部最优解。如果 n_init 很小（比如 1），算法可能运气不好选到了糟糕的初始点。
随着 n_init 增大，算法有更多机会找到全局最优解，曲线通常会呈现下降趋势并最终趋于平稳。

dependency packages
pip install scikit-learn
pip install pandas
pip install numpy
"""
# 读取数据文件
embedding_df = pd.read_parquet("data/20_newsgroup_with_embedding.parquet")

# 准备特征矩阵
matrix = np.vstack(embedding_df.embedding.values)

# 聚类数量
num_of_clusters = 20

"""
测试 n_init 参数对 KMeans 聚类结果（Inertia）的影响。

参数:
matrix: 训练数据 (feature matrix)
cluster_num: 聚类簇数 (k值)
max_n_init: 测试 n_init 的最大上限，默认为 30
"""
def test_n_init_impact(matrix, cluster_num, max_n_init=30):
    inertias = []
    n_init_range = range(1, max_n_init + 1)

    for n in n_init_range:
        # 注意：这里为了观察 n_init 的随机影响，我们不固定 random_state
        model = KMeans(
            n_clusters=cluster_num,
            init="k-means++",
            n_init=n,
            random_state=None  # 允许观察随机性带来的差异
        )
        model.fit(matrix)
        inertias.append(model.inertia_)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(n_init_range, inertias, marker='o', linestyle='-', color='b')
    plt.title(f'Impact of n_init on KMeans Inertia (k={cluster_num})')
    plt.xlabel('n_init (Number of initializations)')
    plt.ylabel('Inertia (Lower is better)')
    plt.grid(True)
    plt.show()


# 调用函数
test_n_init_impact(matrix, num_of_clusters)
