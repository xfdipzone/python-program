# coding=utf-8
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

"""
利用 TfidfVectorizer 提取每个聚类的 TF-IDF 关键词（Top 10）

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
# 2. 计算每个聚类的 Top 10 关键词
# ==========================================
# 将属于同一个 cluster 的文本拼接在一起
# 假设 embedding_df 中有一列 'text' 存储了原始文本内容
cluster_docs = embedding_df.groupby(
    'cluster')['text'].apply(lambda x: ' '.join(x)).tolist()

# 初始化 TF-IDF 向量化器
# stop_words='english' 可以过滤掉无意义的常用词
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000,
    token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
    min_df=2  # 一个词至少要在 2 个文档中出现过，防止极端长尾词
)
tfidf_matrix = tfidf_vectorizer.fit_transform(cluster_docs)

# 获取特征词列表
feature_names = tfidf_vectorizer.get_feature_names_out()

# 提取每个簇的 Top 10 关键词
top_keywords = {}
for i in range(num_of_clusters):
    # 获取第 i 个簇的 tf-idf 向量评分
    row = tfidf_matrix.getrow(i).toarray()[0]

    # 获取评分最高的前 10 个索引
    top_indices = row.argsort()[-10:][::-1]

    # 映射回单词
    keywords = [feature_names[idx] for idx in top_indices]
    top_keywords[i] = keywords

# 打印结果
for cluster_id, keywords in top_keywords.items():
    print(f"Cluster {cluster_id:02} Keywords: {', '.join(keywords)}")


"""
Cluster 00 Keywords: monitor, vga, card, video, windows, screen, drivers, vesa, color, vram
Cluster 01 Keywords: god, jesus, bible, christians, people, christ, christian, church, faith, believe
Cluster 02 Keywords: just, don, like, think, people, know, say, post, edu, said
Cluster 03 Keywords: sale, shipping, offer, new, condition, price, obo, sell, used, disks
Cluster 04 Keywords: msg, patients, food, don, disease, people, know, doctor, like, foods
Cluster 05 Keywords: people, don, think, like, just, government, morality, make, objective, know
Cluster 06 Keywords: car, cars, like, engine, just, dealer, don, know, ford, new
Cluster 07 Keywords: israel, israeli, armenian, arab, jews, turkish, people, palestinian, jewish, arabs
Cluster 08 Keywords: hockey, team, nhl, game, players, leafs, season, rangers, playoffs, games
Cluster 09 Keywords: bike, bikes, motorcycle, ride, helmet, just, riding, like, rider, honda
Cluster 10 Keywords: windows, dos, file, files, use, program, like, ftp, problem, using
Cluster 11 Keywords: window, lib, xterm, widget, motif, server, use, windows, like, colormap
Cluster 12 Keywords: baseball, year, team, braves, players, game, cubs, good, games, don
Cluster 13 Keywords: scsi, drive, ide, controller, mac, motherboard, bus, know, like, does
Cluster 14 Keywords: amp, voltage, use, like, power, circuit, output, just, ground, good
Cluster 15 Keywords: gun, guns, people, fbi, don, crime, batf, weapons, government, firearms
Cluster 16 Keywords: space, orbit, nasa, lunar, shuttle, moon, launch, just, like, earth
Cluster 17 Keywords: mail, thanks, edu, address, know, information, com, email, list, send
Cluster 18 Keywords: graphics, files, thanks, tiff, image, know, algorithm, polygon, gif, program
Cluster 19 Keywords: encryption, key, clipper, nsa, chip, escrow, government, keys, encrypted, crypto
"""
