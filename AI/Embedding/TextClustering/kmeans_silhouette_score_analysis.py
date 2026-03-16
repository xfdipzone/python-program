# coding=utf-8
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
分析 K-Means 算法轮廓系数 (Silhouette Score)

用于找出最优的聚类数量（即划分多少个聚类）

轮廓系数同时考虑了类内的紧凑度和类间的分离度。
它的取值范围在 $[-1, 1]$ 之间，越接近 1 代表聚类效果越好。

dependency packages
pip install scikit-learn
pip install matplotlib
pip install pandas
pip install numpy
"""
# 读取数据文件
embedding_df = pd.read_parquet("data/20_newsgroup_with_embedding.parquet")

# 准备特征矩阵
matrix = np.vstack(embedding_df.embedding.values)

"""
测试不同 K 值的轮廓系数

参数:
X: 特征矩阵 (feature matrix)
k_range: 想要测试的 K 值列表，例如 range(2, 21)
sample_size: 抽样大小，防止高维大数据量导致内存溢出，默认为 5000
"""
def plot_silhouette_scores(X, k_range, sample_size=5000):
    scores = []

    # 如果数据量太大，进行随机抽样以加速计算
    if len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices] if isinstance(X, np.ndarray) else X.iloc[indices]
    else:
        X_sample = X

    for k in k_range:
        # 建立模型
        kmeans = KMeans(n_clusters=k, init="k-means++",
                        n_init=15, random_state=42)
        labels = kmeans.fit_predict(X)  # 在全量数据上训练

        # 提取对应抽样样本的标签
        if len(X) > sample_size:
            sample_labels = labels[indices]
        else:
            sample_labels = labels

        # 计算轮廓系数
        score = silhouette_score(X_sample, sample_labels)
        scores.append(score)
        print(f"K={k:2}, Silhouette Score: {score:.4f}")

    print("\n")

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, scores, marker='s', color='darkorange', linewidth=2)
    plt.title('Silhouette Score for different K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Coefficient')
    plt.grid(True)
    plt.show()


# 调用函数
k_list = list(range(2, 31))
plot_silhouette_scores(matrix, k_list, 10000)


"""
不同 K 值的轮廓系数

K= 2, Silhouette Score: 0.0445
K= 3, Silhouette Score: 0.0416
K= 4, Silhouette Score: 0.0377
K= 5, Silhouette Score: 0.0310
K= 6, Silhouette Score: 0.0277
K= 7, Silhouette Score: 0.0305
K= 8, Silhouette Score: 0.0291
K= 9, Silhouette Score: 0.0291
K=10, Silhouette Score: 0.0317
K=11, Silhouette Score: 0.0335
K=12, Silhouette Score: 0.0335
K=13, Silhouette Score: 0.0335
K=14, Silhouette Score: 0.0345
K=15, Silhouette Score: 0.0236
K=16, Silhouette Score: 0.0332
K=17, Silhouette Score: 0.0257
K=18, Silhouette Score: 0.0258
K=19, Silhouette Score: 0.0254
K=20, Silhouette Score: 0.0274
K=21, Silhouette Score: 0.0268
K=22, Silhouette Score: 0.0262
K=23, Silhouette Score: 0.0266
K=24, Silhouette Score: 0.0273
K=25, Silhouette Score: 0.0276
K=26, Silhouette Score: 0.0274
K=27, Silhouette Score: 0.0275
K=28, Silhouette Score: 0.0281
K=29, Silhouette Score: 0.0249
K=30, Silhouette Score: 0.0247
"""
