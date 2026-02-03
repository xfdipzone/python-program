# coding=utf-8
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

"""
梯度提升决策树（LightGBM）模型重要特征分析

dependency packages
pip install scikit-learn
pip install pandas
pip install numpy
pip install shap
"""
# 读取样本数据文件
training_data = pd.read_parquet("data/20_newsgroup_with_embedding.parquet")

# 从样本中随机抽取 10000 条记录用于训练与测试
# random_state 是随机乱数，用于保证每次随机抽取的记录一致
df = training_data.sample(10000, random_state=42)

# 划分训练集与测试集（80% 的数据作为训练集，20% 的数据作为测试集）
X_train_val, X_test, y_train_val, y_test = train_test_split(
    list(df.embedding.values), df.target, test_size=0.2, random_state=42
)

# 从训练集中分割出训练集与验证集（90% 的数据作为训练集，10% 的数据作为验证集）
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.125, random_state=42
)

# 将列表转换为适合 LightGBM 的格式（如果 embedding 已经是 numpy 数组格式，则不需要转换）
X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)

# 获取特征维度并生成列名列表
feature_dim = X_train.shape[1]
feature_names = [f"f_{i}" for i in range(feature_dim)]

# 转换为 DataFrame 并指定列名
X_train = pd.DataFrame(X_train, columns=feature_names)
X_val = pd.DataFrame(X_val, columns=feature_names)
X_test = pd.DataFrame(X_test, columns=feature_names)

# 创建梯度提升决策树（LightGBM）分类器
clf = lgb.LGBMClassifier(
    n_estimators=500,         # 树的数量
    learning_rate=0.05,       # 学习率
    max_depth=5,              # 树的最大深度
    num_leaves=28,            # 叶子节点数
    feature_fraction=0.1,     # 使用的向量维度
    class_weight="balanced",  # 平衡类别样本
    bagging_fraction=0.8,     # 样本采样比例
    bagging_freq=5,           # 采样频率
    lambda_l1=0.1,            # L1 正则化，让不重要的特征权重彻底归零
    lambda_l2=0.1,            # L2 正则化，防止某个特征权重过大
    random_state=42,          # 随机种子
    n_jobs=-1,                # 使用所有 CPU 核心
    verbose=-1                # 不输出训练日志
)

# 训练
clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='multi_logloss',
    callbacks=[
        early_stopping(stopping_rounds=20),
        log_evaluation(period=0)  # 控制打印训练日志的频率（0:不打印 N:每经过 N 轮打印一次）
    ]
)

# ==========================================
# 获取特征重要性及计算 top 100 特征重要性总和
# ==========================================
# 获取信息增益重要性（Gain）
importances_gain = clf.booster_.feature_importance(importance_type='gain')
importances = importances_gain / np.sum(importances_gain)
indices = np.argsort(importances)[::-1]

# 打印前 25 个最重要的原始特征维度
print("原始特征重要性排行")

for i in range(25):
    print(f"原始特征 {indices[i]:4}: 重要性评分 {importances[indices[i]]:.4f}")

# 计算 top 100 特征重要性总和（总和达到 0.5 以上表示特征非常集中）
top100_total_importances = 0
for i in range(100):
    top100_total_importances += importances[indices[i]]

print(f"\nTop 100 特征重要性评分总和: {top100_total_importances:.4f}\n\n")


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
        f"Top {top_n} Feature Importances (LightGBM Model)", fontsize=15)

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
    ax1.set_title(f"Class {target_class_index}: Global Importance", pad=20)

    # 增加底部标签的边距，防止重叠
    ax1.set_xlabel("Average SHAP value", labelpad=15)

    # --- 右图：蜂窝图 ---
    ax2 = fig.add_subplot(1, 2, 2)
    shap.summary_plot(current_shap_values, X_sample,
                      feature_names=feature_names, show=False)
    ax2.set_title(f"Class {target_class_index}: Impact Direction", pad=20)
    ax2.set_xlabel("SHAP value (Impact on Output)", labelpad=15)

    # 设置总标题
    fig.suptitle(
        f"SHAP Analysis for Class {target_class_index} (LightGBM)", fontsize=18, y=0.98)

    # 自适应子图排版
    plt.tight_layout()

    plt.show()


# 打印指定类别的特征重要性柱形图与蜂窝图（target_class_index 设置打印的类别）
run_shap_analysis(
    clf,
    X_test,
    feature_names=feature_names,
    target_class_index=1,
    sample_size=100
)
