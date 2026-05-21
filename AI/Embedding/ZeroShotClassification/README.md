# 零样本文本分类（Zero Shot Classification）

本项目不进行任何样本训练（零样本），只基于模型计算 Embedding 相似度来实现文本分类

## 计算文本 Embedding

- [HuggingFaceEmbedding 计算文本 Embedding](./hugging_face_embedding.py)

  使用 HuggingFaceEmbedding 计算文本的 Embedding 与维度

- [SentenceTransformer 计算文本 Embedding](./sentence_transformer_embedding.py)

  使用 SentenceTransformer 计算文本的 Embedding 与维度

---

## 评论文本分类

- [计算评论分级准确率](./comment_level.py)

  基于给出的好评与差评参考文本，通过 Embedding 计算数据源中的评论属于好评/差评的准确率

- [情感分析](./sentiment_analysis.py)

  基于 Embedding 相似度计算评论内容是正面还是负面

---

## 相似产品搜索

- [生成带 Embedding 的产品数据文件](./generate_product_file_with_embedding.py)

  生成产品数据文件（CSV 与 Parquet 格式），包含每个产品的 Embedding

  - [CSV 数据文件](../data/product_data.csv)

  - [Parquet 数据文件](../data/product_data.parquet)

- [从 CSV 中搜索相似产品](./csv_similarity_product_search.py)

  读取 CSV 产品文件，利用 Embedding 余弦相似度搜索相似的产品 `(threshold_value > 0.5)`

  Embedding 余弦相似度使用 `scipy` 包的 `cosine` 计算

- [从 Parquet 中搜索相似产品](./parquet_similarity_product_search.py)

  读取 Parquet 产品文件，利用 Embedding 余弦相似度搜索相似的产品 `(threshold_value > 0.5)`

  Embedding 余弦相似度使用 `sentence_transformers` 包的 `util` 计算
