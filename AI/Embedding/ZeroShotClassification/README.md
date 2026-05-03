# 零样本文本分类（Zero Shot Classification）

本项目不进行任何样本训练（零样本），只基于模型计算 Embedding 相似度来实现文本分类

## 评论文本分类

- [计算评论分级准确率](./comment_level.py)

  基于给出的好评与差评参考文本，通过 Embedding 计算数据源中的评论属于好评/差评的准确率

- [情感分析](./sentiment_analysis.py)

  基于 Embedding 相似度计算评论内容是正面还是负面
