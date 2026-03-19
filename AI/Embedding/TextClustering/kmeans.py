# coding=utf-8
from openai import OpenAI
from google.colab import userdata
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import re

"""
通过 K-Means 算法实现文本聚类

dependency packages
pip install openai
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


# ==========================================
# 3. 生成聚类主题
# ==========================================
client = OpenAI(
    api_key=userdata.get("KIMI_API_KEY"),
    base_url="https://api.moonshot.cn/v1"
)

COMPLETION_MODEL = "moonshot-v1-8k"

# 每个聚类中取 10% 的数据用于总结主题
items_per_cluster = 10

print("\n\033[1;32mClustering Themes\033[0m\n")

for i in range(num_of_clusters):
    cluster_name = new_df[new_df.cluster == i].iloc[0].rank1
    print(f"Cluster {i:02}, Rank 1: {cluster_name}, Theme:", end=" ")

    content = "\n".join(
        embedding_df[embedding_df.cluster == i].text.sample(
            items_per_cluster, random_state=42).values
    )

    prompt = f"""
        我们想要给下面的内容，分组成有意义的类别，以便我们可以对其进行总结。
        请根据下面这些内容的共同点，使用中文总结一个 50 字以内的新闻组的名称。
        只需要给出名称，比如 “PC硬件”

        内容:
        {content}
    """

    completions = client.chat.completions.create(
        model=COMPLETION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100,
        top_p=1,
    )

    response = completions.choices[0].message.content

    # 数据清洗
    pattern = r'“|”|\n'
    print(re.sub(pattern, '', response).strip())


"""
cluster  count                     rank1  rank1_count                     rank2  rank2_count  rank1_per  rank2_per
      8    470          rec.sport.hockey          456        rec.sport.baseball           14     97.02%      2.98%
     12    470        rec.sport.baseball          455          rec.sport.hockey           13     96.81%      2.77%
      9    372           rec.motorcycles          357                 rec.autos            6     95.97%      1.61%
     19    371                 sci.crypt          355           sci.electronics            5     95.69%      1.35%
     11    422            comp.windows.x          398             comp.graphics           12     94.31%      2.84%
      4    434                   sci.med          406        talk.politics.misc            9     93.55%      2.07%
     16    438                 sci.space          393           sci.electronics           21     89.73%      4.79%
      7    437     talk.politics.mideast          374               alt.atheism           24     85.58%      5.49%
      3    516              misc.forsale          432     comp.sys.mac.hardware           20     83.72%      3.88%
      6    534                 rec.autos          424           rec.motorcycles           47     79.40%      8.80%
     14    398           sci.electronics          313     comp.sys.mac.hardware           20     78.64%      5.03%
     18    378             comp.graphics          295            comp.windows.x           33     78.04%      8.73%
     15    534        talk.politics.guns          384        talk.politics.misc           34     71.91%      6.37%
     10    589   comp.os.ms-windows.misc          351  comp.sys.ibm.pc.hardware           62     59.59%     10.53%
      1    815    soc.religion.christian          457        talk.religion.misc          172     56.07%     21.10%
     13    820     comp.sys.mac.hardware          335  comp.sys.ibm.pc.hardware          327     40.85%     39.88%
      5    662        talk.politics.misc          233               alt.atheism          162     35.20%     24.47%
      0    389  comp.sys.ibm.pc.hardware          104     comp.sys.mac.hardware           87     26.74%     22.37%
     17    548           sci.electronics           66            comp.windows.x           47     12.04%      8.58%
      2    940        talk.politics.misc          112           rec.motorcycles          102     11.91%     10.85%

Clustering Themes

Cluster 00, Rank 1: comp.sys.ibm.pc.hardware, Theme: 电脑硬件与显示器问题讨论
Cluster 01, Rank 1: soc.religion.christian, Theme: 宗教与信仰讨论
Cluster 02, Rank 1: talk.politics.misc, Theme: 网络幽默与轶事
Cluster 03, Rank 1: misc.forsale, Theme: 电子产品与滑雪设备交易
Cluster 04, Rank 1: sci.med, Theme: 医疗健康与疾病探讨
Cluster 05, Rank 1: talk.politics.misc, Theme: 社会观点与科学哲学讨论
Cluster 06, Rank 1: rec.autos, Theme: 汽车维修与技术讨论
Cluster 07, Rank 1: talk.politics.mideast, Theme: 国际冲突与地缘政治争议
Cluster 08, Rank 1: rec.sport.hockey, Theme: 冰球争议与讨论
Cluster 09, Rank 1: rec.motorcycles, Theme: 摩托车安全与驾驶技巧
Cluster 10, Rank 1: comp.os.ms-windows.misc, Theme: 跨平台文件传输与系统兼容性问题
Cluster 11, Rank 1: comp.windows.x, Theme: "X Window系统开发与问题解决"
Cluster 12, Rank 1: rec.sport.baseball, Theme: 棒球赛事分析与讨论
Cluster 13, Rank 1: comp.sys.mac.hardware, Theme: 计算机硬件与网络问题讨论
Cluster 14, Rank 1: sci.electronics, Theme: 电子工程与传感器技术
Cluster 15, Rank 1: talk.politics.guns, Theme: 枪支权利与自卫争议
Cluster 16, Rank 1: sci.space, Theme: 太空探索与技术发展
Cluster 17, Rank 1: sci.electronics, Theme: 技术交流与资源分享
Cluster 18, Rank 1: comp.graphics, Theme: "图形算法与软件开发"
Cluster 19, Rank 1: sci.crypt, Theme: 加密技术与隐私权讨论
"""
