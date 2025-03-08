# coding=utf-8
"""
基于 embedding 做情感分析
判断评论是好评还是差评

dependency packages
pip install numpy
pip install sentence-transformers
"""
from sentence_transformers import SentenceTransformer
import numpy as np

# 使用 SentenceTransformer 加载计算文本向量模型
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

"""
计算文本的向量 (embedding)
"""
def get_embedding(text):
    # SentenceTransformer 的 encode 方法可以直接处理文本并返回嵌入向量
    return embed_model.encode(text)

"""
计算向量的余弦相似度
"""
def cosine_similarity(vector_a, vector_b):
  dot_product = np.dot(vector_a, vector_b)
  norm_a = np.linalg.norm(vector_a)
  norm_b = np.linalg.norm(vector_b)
  epsilon = 1e-10
  cosine_similarity = dot_product / (norm_a * norm_b + epsilon)
  return cosine_similarity

"""
计算评分
"""
def get_score(sample_embedding, positive_weight=0.7, negative_weight=0.3):
    positive_sim = cosine_similarity(sample_embedding, positive_review)
    negative_sim = cosine_similarity(sample_embedding, negative_review)
    return positive_weight * positive_sim - negative_weight * negative_sim

# 获取好评和差评的 embedding
positive_review = get_embedding("好评")
negative_review = get_embedding("差评")

good_restaurant = get_embedding("这家餐馆太好吃了，一点都不糟糕")
bad_restaurant = get_embedding("这家餐馆太糟糕了，一点都不好吃")

good_score = get_score(good_restaurant)
bad_score = get_score(bad_restaurant)
print("好评餐馆的评分：%2f" % good_score)
print("差评餐馆的评分：%2f\n" % bad_score)

# 获取示例文本的 embedding
positive_examples = [
    get_embedding("买的银色版真的很好看，一天就到了，晚上就开始拿起来完系统很丝滑流畅，做工扎实，手感细腻，很精致哦苹果一如既往的好品质"),
    get_embedding("这款产品真的是超出预期！质量非常好，做工精细，用起来非常顺手。功能也很强大，完全满足了我的需求。客服也很耐心，有问题随时解答。五星好评，强烈推荐！"),
    get_embedding("这次购物体验太棒了！从下单到收货，速度飞快，包装严实。产品本身质量也很不错，性价比超高。客服态度特别好，解答问题很详细。下次还会再来！"),
    get_embedding("真的太惊喜了！产品不仅质量好，而且功能比我想象的还要强大。客服也很贴心，提前告知了一些使用小技巧。物流也很快，两天就到了。五星好评，值得购买！"),
    get_embedding("性价比超高的一款产品！价格实惠，质量却不含糊。功能齐全，操作简单。客服也很热情，解答问题很及时。买了之后完全不后悔，推荐给大家！"),
    get_embedding("从产品的外观设计到实际使用，都无可挑剔。细节处理得很好，功能也很实用。客服服务周到，物流也很给力。五星好评，希望商家继续保持这样的品质！"),
    get_embedding("这款产品让我非常满意！质量好，性能稳定，功能丰富。客服也很专业，解答问题很详细。物流速度也很快。五星好评，下次还会继续支持！"),
    get_embedding("性价比很高的一款产品，价格实惠，质量却很好。功能齐全，操作也很方便。客服服务周到，物流也很及时。五星好评，值得购买！"),
    get_embedding("收到产品后真的太惊喜了！质量超出预期，功能也很强大。客服态度很好，物流速度也很快。五星好评，推荐给大家！"),
    get_embedding("从购买到使用，整个过程都非常顺利。产品质量好，功能齐全。客服服务也很贴心，物流也很给力。五星好评，下次还会再来！"),
]

negative_examples = [
    get_embedding("随意降价，不予价保，服务态度差"),
    get_embedding("很失望，产品质量不太好。刚用不久就出现了问题，功能也不太稳定。客服态度也不好，解决问题很拖沓。希望商家能改进一下。"),
    get_embedding("物流速度太慢了，等了好久才收到。而且包装也很简陋，产品有些磨损。质量一般，功能也不太好用。不推荐购买。"),
    get_embedding("客服态度很差，问问题很久才回复，而且解决不了实际问题。产品质量也不好，功能很有限。物流也很慢，整体体验很差。不推荐。"),
    get_embedding("这款产品真的让我很失望！质量很差，功能也不实用。客服服务几乎没有，物流速度也很慢。一星都不想给！")
]

print("好评例子的分数 (>0)\n")
for i, example in enumerate(positive_examples):
    positive_score = get_score(example)
    print("%02d 的评分: %f" % (i + 1, positive_score))

print("\n差评例子的分数 (<0)\n")
for i, example in enumerate(negative_examples):
    negative_score = get_score(example)
    print("%02d 的评分: %f" % (i + 1, negative_score))
