# 零样本文本分类评估

本项目不进行任何样本训练（零样本），只基于模型计算 Embedding 相似度来实现文本分类

---

## 计算评论分级准确率

文本数据文件：[../data/fine_food_reviews_with_embeddings_1k.csv](<../data/fine_food_reviews_with_embeddings_1k.csv>)

基于给出的好评与差评参考文本，通过 Embedding 计算数据源中的评论属于好评/差评的准确率

```txt
测试结果：
              precision    recall  f1-score   support

    positive       0.97      0.98      0.97       789
    negative       0.85      0.82      0.83       136

    accuracy                           0.95       925
   macro avg       0.91      0.90      0.90       925
weighted avg       0.95      0.95      0.95       925
```

Gemma-300M 正面评价的精确率-召回率曲线

![Gemma-300M positive Reviews Precision-Recall curve](<./img/gemma-300m_positive_reviews_precision_recall_curve.svg>)

Gemma-300M 负面评价精确率-召回率曲线

![Gemma-300M Negative Reviews Precision-Recall curve](<./img/gemma-300m_negative_reviews_precision_recall_curve.svg>)
