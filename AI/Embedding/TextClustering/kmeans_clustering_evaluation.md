# K-Means 文本聚类评估文档

文本数据文件：[../data/20_newsgroup_with_embedding.parquet](<../data/20_newsgroup_with_embedding.parquet>)

## K-Means 聚类结果

包含每个聚类 rank1 与 rank2 的分类数量

| cluster | count | rank1 | rank1_count | rank2 | rank2_count | rank1_per | rank2_per |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 470 | rec.sport.hockey | 456 | rec.sport.baseball | 14 | 97.02% | 2.98% |
| 12 | 470 | rec.sport.baseball | 455 | rec.sport.hockey | 13 | 96.81% | 2.77% |
| 9 | 372 | rec.motorcycles | 357 | rec.autos | 6 | 95.97% | 1.61% |
| 19 | 371 | sci.crypt | 355 | sci.electronics | 5 | 95.69% | 1.35% |
| 11 | 422 | comp.windows.x | 398 | comp.graphics | 12 | 94.31% | 2.84% |
| 4 | 434 | sci.med | 406 | talk.politics.misc | 9 | 93.55% | 2.07% |
| 16 | 438 | sci.space | 393 | sci.electronics | 21 | 89.73% | 4.79% |
| 7 | 437 | talk.politics.mideast | 374 | alt.atheism | 24 | 85.58% | 5.49% |
| 3 | 516 | misc.forsale | 432 | comp.sys.mac.hardware | 20 | 83.72% | 3.88% |
| 6 | 534 | rec.autos | 424 | rec.motorcycles | 47 | 79.40% | 8.80% |
| 14 | 398 | sci.electronics | 313 | comp.sys.mac.hardware | 20 | 78.64% | 5.03% |
| 18 | 378 | comp.graphics | 295 | comp.windows.x | 33 | 78.04% | 8.73% |
| 15 | 534 | talk.politics.guns | 384 | talk.politics.misc | 34 | 71.91% | 6.37% |
| 10 | 589 | comp.os.ms-windows.misc | 351 | comp.sys.ibm.pc.hardware | 62 | 59.59% | 10.53% |
| 1 | 815 | soc.religion.christian | 457 | talk.religion.misc | 172 | 56.07% | 21.10% |
| 13 | 820 | comp.sys.mac.hardware | 335 | comp.sys.ibm.pc.hardware | 327 | 40.85% | 39.88% |
| 5 | 662 | talk.politics.misc | 233 | alt.atheism | 162 | 35.20% | 24.47% |
| 0 | 389 | comp.sys.ibm.pc.hardware | 104 | comp.sys.mac.hardware | 87 | 26.74% | 22.37% |
| 17 | 548 | sci.electronics | 66 | comp.windows.x | 47 | 12.04% | 8.58% |
| 2 | 940 | talk.politics.misc | 112 | rec.motorcycles | 102 | 11.91% | 10.85% |

---

## K-Means 聚类主题

**使用 AI 为每个聚类生成主题：**

| Cluster | Rank1 | Theme |
|:--------|:------|:------|
| 00 | comp.sys.ibm.pc.hardware | 电脑硬件与显示器问题讨论 |
| 01 | soc.religion.christian | 宗教与信仰讨论 |
| 02 | talk.politics.misc | 网络幽默与轶事 |
| 03 | misc.forsale | 电子产品与滑雪设备交易 |
| 04 | sci.med | 医疗健康与疾病探讨 |
| 05 | talk.politics.misc | 社会观点与科学哲学讨论 |
| 06 | rec.autos | 汽车维修与技术讨论 |
| 07 | talk.politics.mideast | 国际冲突与地缘政治争议 |
| 08 | rec.sport.hockey | 冰球争议与讨论 |
| 09 | rec.motorcycles | 摩托车安全与驾驶技巧 |
| 10 | comp.os.ms-windows.misc | 跨平台文件传输与系统兼容性问题 |
| 11 | comp.windows.x | X Window系统开发与问题解决 |
| 12 | rec.sport.baseball | 棒球赛事分析与讨论 |
| 13 | comp.sys.mac.hardware | 计算机硬件与网络问题讨论 |
| 14 | sci.electronics | 电子工程与传感器技术 |
| 15 | talk.politics.guns | 枪支权利与自卫争议 |
| 16 | sci.space | 太空探索与技术发展 |
| 17 | sci.electronics | 技术交流与资源分享 |
| 18 | comp.graphics | 图形算法与软件开发 |
| 19 | sci.crypt | 加密技术与隐私权讨论 |

---

## K-Means n_init 参数分析

分析 K-Means 算法 n_init 参数影响 Inertia（惯性）的变化曲线

![K-Means n_init 参数分析](<./img/kmeans_ninit-analysis.svg>)

取值 `n_init = 15` 性价比最高

---

## K-Means 轮廓系数 (Silhouette Score)分析

用于找出最优的聚类数量（即划分多少个聚类），**分为 `14` 个类最优**

![K-Means 轮廓系数 (Silhouette Score)分析](<./img/kmeans_silhouette_score_analysis.svg>)

先对原始数据进行降维操作（0.95），再计算最优的聚类数量，**分为 `13` 个类最优**

![K-Means PCA 降维轮廓系数 (Silhouette Score)分析](<./img/kmeans_pca_silhouette_score_analysis.svg>)

---

## K-Means t-SNE 可视化聚类分布

t-SNE（t-distributed Stochastic Neighbor Embedding）是最常用的非线性降维技术

使用 t-SNE 非线形降维技术，可视化 K-Means 聚类分布

![可视化 K-Means 聚类分布](<./img/kmeans_tsne_cluster_distribution.svg>)

使用 t-SNE 非线形降维技术，可视化 K-Means 聚类分布（每个簇自动打上主题标签 Label）

![可视化 K-Means 聚类分布（包含聚类主题标签）](<./img/kmeans_tsne_cluster_distribution_with_topic.svg>)

---

## K-Means 提取每个聚类的关键词（Top 10）

利用 `TfidfVectorizer` 提取每个聚类的 `TF-IDF` 关键词（Top 10）

| Cluster | Keywords |
|:--------|:---------|
| 00 | monitor, vga, card, video, windows, screen, drivers, vesa, color, vram |
| 01 | god, jesus, bible, christians, people, christ, christian, church, faith, believe |
| 02 | just, don, like, think, people, know, say, post, edu, said |
| 03 | sale, shipping, offer, new, condition, price, obo, sell, used, disks |
| 04 | msg, patients, food, don, disease, people, know, doctor, like, foods |
| 05 | people, don, think, like, just, government, morality, make, objective, know |
| 06 | car, cars, like, engine, just, dealer, don, know, ford, new |
| 07 | israel, israeli, armenian, arab, jews, turkish, people, palestinian, jewish, arabs |
| 08 | hockey, team, nhl, game, players, leafs, season, rangers, playoffs, games |
| 09 | bike, bikes, motorcycle, ride, helmet, just, riding, like, rider, honda |
| 10 | windows, dos, file, files, use, program, like, ftp, problem, using |
| 11 | window, lib, xterm, widget, motif, server, use, windows, like, colormap |
| 12 | baseball, year, team, braves, players, game, cubs, good, games, don |
| 13 | scsi, drive, ide, controller, mac, motherboard, bus, know, like, does |
| 14 | amp, voltage, use, like, power, circuit, output, just, ground, good |
| 15 | gun, guns, people, fbi, don, crime, batf, weapons, government, firearms |
| 16 | space, orbit, nasa, lunar, shuttle, moon, launch, just, like, earth |
| 17 | mail, thanks, edu, address, know, information, com, email, list, send |
| 18 | graphics, files, thanks, tiff, image, know, algorithm, polygon, gif, program |
| 19 | encryption, key, clipper, nsa, chip, escrow, government, keys, encrypted, crypto |

---

## K-Means 使用随机森林（RandomForest）模型计算重要特征

Top 50 重要特征

![K-Means Top 50 重要特征](<./img/kmeans_top50_important_features.svg>)

重要特征柱状图与蜂窝图 SHAP (SHapley Additive exPlanations)

![K-Means 随机森林模型重要特征](<./img/kmeans_random_forest_shap_analysis.svg>)
