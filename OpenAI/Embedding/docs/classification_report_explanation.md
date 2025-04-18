# 分类报告指标说明 (Classification Report Explanation)

F1-score 是一种用于评估分类模型性能的指标，特别适用于类别不平衡的数据集。

它综合考虑了模型的**精确率（Precision）**和**召回率（Recall）**，通过调和平均值来衡量模型的整体表现。

---

## 1. 精确率（Precision）

精确率表示在所有被模型预测为正类的样本中，实际为正类的比例。计算公式为：

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

- \(TP\)（True Positives）：真正例，正确预测为正类的样本数。
- \(FP\)（False Positives）：假正例，错误预测为正类的样本数。

---

## 2. 召回率（Recall）

召回率表示在所有实际为正类的样本中，被模型正确预测为正类的比例。计算公式为：

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

- \(FN\)（False Negatives）：假负例，错误预测为负类的实际正类样本数。

---

## 3. F1-score

F1-score 是精确率和召回率的调和平均值，计算公式为：

$$
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

---

## 4. F1-score 的意义

- F1-score 的取值范围是 0 到 1，值越大表示模型性能越好。
- 它在精确率和召回率之间取得平衡，避免单独依赖其中一个指标可能带来的偏差。
- 当精确率和召回率差距较大时，F1-score 能更全面地反映模型表现。

---

## 5. 适用场景

- 类别不平衡问题，例如疾病诊断中阳性样本较少。
- 需要同时关注假阳性和假阴性的场景，如垃圾邮件检测、欺诈识别等。

---

## 总结

F1-score 是评价分类模型性能的重要指标，尤其在不平衡数据集和对错误类型敏感的任务中，能够帮助我们更准确地衡量模型的实际效果。
