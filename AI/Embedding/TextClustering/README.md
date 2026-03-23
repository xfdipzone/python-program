# 文本聚类（Text Clustering）

本项目旨在探索 **K-Means** 算法在文本领域的应用，包含主题提取、超参数优化、聚类评估及高维数据可视化等核心环节。

## K-Means 聚类执行与主题提取

- [K-Means 聚类与聚类主题生成](./kmeans.py)

  使用 K-Means 算法实现文本聚类，并通过 AI 生成每个聚类的主题

- [提取每个聚类的关键词（Top 10）](./kmeans_clustering_top10_keywords.py)

  利用 TfidfVectorizer 提取每个聚类的 TF-IDF 关键词（Top 10）

- [K-Means 聚类重要特征分析](./kmeans_important_features.py)

  基于随机森林（Random Forest）分析 K-Means 聚类重要特征

---

## 模型评估与参数调优

- [K-Means 聚类 n_init 参数分析](./kmeans_n_init_analysis.py.py)

  分析 K-Means 算法 n_init 参数影响 Inertia（惯性）的变化曲线

- [分析 K-Means 算法轮廓系数 (Silhouette Score)](./kmeans_silhouette_score_analysis.py)

  用于找出最优的聚类数量（即划分多少个聚类）

- [分析 K-Means 算法 PCA 降维轮廓系数 (Silhouette Score)](./kmeans_pca_silhouette_score_analysis.py)

  先对原始数据进行降维操作（0.95），再计算最优的聚类数量

---

## 聚类分布可视化分析

- [可视化 K-Means 聚类分布](./kmeans_tsne_cluster_distribution.py)

  使用 t-SNE 非线形降维技术，可视化 K-Means 聚类分布

- [可视化 K-Means 聚类分布（包含聚类主题标签）](./kmeans_tsne_cluster_distribution_with_topic.py)

  使用 t-SNE 非线形降维技术，可视化 K-Means 聚类分布（每个簇自动打上主题标签 Label）
