# coding=utf-8
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

"""
K-Means 聚类使用随机森林（RandomForest）模型计算重要特征

dependency packages
pip install scikit-learn
pip install pandas
pip install numpy
pip install shap
pip install matplotlib
pip install seaborn
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
# 2. 分析聚类的重要特征
# ==========================================
# X 为 Embedding 矩阵，y 为 K-Means 生成的聚类标签
X = matrix  # 形状为 (n_samples, n_features)
y = embedding_df['cluster'].values

# 训练一个代理模型 (Surrogate Model)
# 随机森林可以很好地捕捉特征间的非线性关系
clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
clf.fit(X, y)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

# ==========================================
# 绘制特征重要性柱状图和累积贡献曲线
# ==========================================
def plot_feature_importance(importances, indices, top_n=50):
    # 准备数据
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    cumulative_importances = np.cumsum(importances[indices])  # 计算累积重要性

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 1. 绘制柱状图 (左轴)
    sns.barplot(
        x=list(range(top_n)),
        y=top_importances,
        ax=ax1,
        hue=list(range(top_n)),
        palette="magma",
        legend=False
    )

    # 前 {top_n} 个特征重要性分布 (是否存在断层)
    ax1.set_title(
        f"Top {top_n} K-Means Feature Importances (Random Forest Model)", fontsize=15)

    # 特征排行 (从高到低)
    ax1.set_xlabel("Feature Rank (Index)")

    # 单一特征重要性评分
    ax1.set_ylabel("Importance Score")

    # 设置横坐标标签为原始维度编号
    ax1.set_xticks(range(top_n))
    ax1.set_xticklabels(top_indices, rotation=90, fontsize=8)

    # 2. 绘制累积贡献曲线 (右轴)
    ax2 = ax1.twinx()

    # 累积重要性
    ax2.plot(range(top_n), cumulative_importances[:top_n],
             color='r', marker='o', markersize=3, label='Cumulative')

    # 累积重要性占比
    ax2.set_ylabel("Cumulative Importance Ratio")
    ax2.set_ylim(0, max(cumulative_importances[:top_n]) * 1.1)

    plt.tight_layout()
    plt.show()


# 设置绘图风格
sns.set_theme(style="whitegrid")

# 调用函数
plot_feature_importance(importances, indices, top_n=50)
print("\n")

# ==========================================
# SHAP (SHapley Additive exPlanations)
# 绘制特征重要性柱形图（Summary Bar Plot）与蜂窝图（Beeswarm Plot / Summary Plot）
# ==========================================
def run_shap_analysis(model, X_data, feature_names=None, target_class_index=0, sample_size=100):
    # 数据采样
    X_array = np.array(X_data)
    if sample_size < len(X_array):
        indices = np.random.choice(len(X_array), sample_size, replace=False)
        X_sample = X_array[indices]
    else:
        X_sample = X_array

    # 计算 SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # 兼容多分类返回列表的情况
    current_shap_values = shap_values[target_class_index] if isinstance(
        shap_values, list) else shap_values[:, :, target_class_index]

    # 创建画布并调整间距
    # 增加 figsize 的宽度，并设置 subplot 之间的 wspace (宽度间距)
    fig = plt.figure(figsize=(16, 8))

    # --- 左图：柱状图 ---
    ax1 = fig.add_subplot(1, 2, 1)
    shap.summary_plot(current_shap_values, X_sample, plot_type="bar",
                      feature_names=feature_names, show=False)
    ax1.set_title(f"Cluster {target_class_index}: Global Importance", pad=20)

    # 增加底部标签的边距，防止重叠
    ax1.set_xlabel("Average SHAP value", labelpad=15)

    # --- 右图：蜂窝图 ---
    ax2 = fig.add_subplot(1, 2, 2)
    shap.summary_plot(current_shap_values, X_sample,
                      feature_names=feature_names, show=False)
    ax2.set_title(f"Cluster {target_class_index}: Impact Direction", pad=20)
    ax2.set_xlabel("SHAP value (Impact on Output)", labelpad=15)

    # 设置总标题
    fig.suptitle(
        f"SHAP Analysis for K-Means Cluster {target_class_index} (Random Forest)", fontsize=18, y=0.98)

    # 自适应子图排版
    plt.tight_layout()

    plt.show()


# 获取特征维度并生成列名列表
feature_dim = X.shape[1]
feature_names = [f"f_{i}" for i in range(feature_dim)]

# 打印指定类别的特征重要性柱形图与蜂窝图（target_class_index 设置打印的类别）
run_shap_analysis(
    clf,
    X,
    feature_names=feature_names,
    target_class_index=1,
    sample_size=200
)
