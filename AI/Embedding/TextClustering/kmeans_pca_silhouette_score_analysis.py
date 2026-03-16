# coding=utf-8
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
分析 K-Means 算法 PCA 降维轮廓系数 (Silhouette Score)

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
测试不同 K 值的轮廓系数（原始数据会先执行 PCA 降维）

参数:
X: 原始高维特征矩阵 (feature matrix)
k_range: 想要测试的 K 值列表，例如 range(2, 21)
n_components: 保留多少方差，0.95 表示保留能解释 95% 变异的主成分
"""
def analyze_with_pca_and_silhouette(X, k_range, n_components=0.95):
    # 1. 标准化（PCA 之前必须做，确保各维度权重公平）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. PCA 降维
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    print(f"原始维度: {X.shape[1]}, 降维后维度: {X_pca.shape[1]}\n")

    scores = []

    # 3. 循环测试不同 K 值
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init="k-means++",
                        n_init=15, random_state=42)
        labels = kmeans.fit_predict(X_pca)

        # 计算轮廓系数 (全量计算，若慢请参考之前的抽样逻辑)
        score = silhouette_score(X_pca, labels)
        scores.append(score)
        print(f"K={k:2}, Silhouette Score: {score:.4f}")

    print("\n")

    # 4. 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, scores, marker='s', color='teal', linewidth=2)
    plt.title('Silhouette Score after PCA Dimensionality Reduction')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Coefficient')
    plt.grid(True)
    plt.show()

    return X_pca


# 调用函数
k_list = list(range(2, 31))
X_pca = analyze_with_pca_and_silhouette(matrix, k_list)


"""
原始维度: 1536, 降维后维度: 710

原始数据降维后不同 K 值的轮廓系数

K= 2, Silhouette Score: 0.0407
K= 3, Silhouette Score: 0.0399
K= 4, Silhouette Score: 0.0375
K= 5, Silhouette Score: 0.0307
K= 6, Silhouette Score: 0.0283
K= 7, Silhouette Score: 0.0316
K= 8, Silhouette Score: 0.0299
K= 9, Silhouette Score: 0.0301
K=10, Silhouette Score: 0.0326
K=11, Silhouette Score: 0.0346
K=12, Silhouette Score: 0.0348
K=13, Silhouette Score: 0.0356
K=14, Silhouette Score: 0.0344
K=15, Silhouette Score: 0.0290
K=16, Silhouette Score: 0.0291
K=17, Silhouette Score: 0.0305
K=18, Silhouette Score: 0.0296
K=19, Silhouette Score: 0.0307
K=20, Silhouette Score: 0.0300
K=21, Silhouette Score: 0.0303
K=22, Silhouette Score: 0.0315
K=23, Silhouette Score: 0.0312
K=24, Silhouette Score: 0.0312
K=25, Silhouette Score: 0.0315
K=26, Silhouette Score: 0.0317
K=27, Silhouette Score: 0.0299
K=28, Silhouette Score: 0.0328
K=29, Silhouette Score: 0.0323
K=30, Silhouette Score: 0.0311
"""
