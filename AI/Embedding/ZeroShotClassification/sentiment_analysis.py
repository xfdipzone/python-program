# coding=utf-8
from sentence_transformers import SentenceTransformer
from google.colab import userdata
from huggingface_hub import login
import numpy as np

"""
基于 embedding 实现情感分析
根据 embedding 相似度判断评论是好评还是差评

dependency packages
pip install sentence-transformers
pip install numpy
"""
# Login HuggingFace Hub
login(token=userdata.get("HF_TOKEN"))

# 使用 SentenceTransformer 加载计算文本向量模型
embed_model = SentenceTransformer("google/embeddinggemma-300M")

# 计算文本的向量 (embedding)
def get_embedding(text):
    return embed_model.encode(text)


# 计算向量的余弦相似度
def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    epsilon = 1e-10
    cosine_similarity = dot_product / (norm_a * norm_b + epsilon)
    return cosine_similarity


# 计算评分
def get_score(sample_embedding):
    positive_sim = cosine_similarity(
        sample_embedding, positive_review_reference)
    negative_sim = cosine_similarity(
        sample_embedding, negative_review_reference)
    return positive_sim - negative_sim


# 获取好评和差评参考文本的 embedding
positive_review_reference = get_embedding("好评")
negative_review_reference = get_embedding("差评")

good_restaurant = "这家餐馆太好吃了，一点都不糟糕"
bad_restaurant = "这家餐馆太糟糕了，一点都不好吃"

good_score = get_score(get_embedding(good_restaurant))
bad_score = get_score(get_embedding(bad_restaurant))
print("%s 评分：%2f" % (good_restaurant, good_score))
print("%s 评分：%2f\n" % (bad_restaurant, bad_score))

# 好评示例文本列表
positive_examples = [
    "买的银色版真的很好看，一天就到了，晚上就开始拿起来完系统很丝滑流畅，做工扎实，手感细腻，很精致哦苹果一如既往的好品质",
    "这款产品真的是超出预期！质量非常好，做工精细，用起来非常顺手。功能也很强大，完全满足了我的需求。客服也很耐心，有问题随时解答。五星好评，强烈推荐！",
    "这次购物体验太棒了！从下单到收货，速度飞快，包装严实。产品本身质量也很不错，性价比超高。客服态度特别好，解答问题很详细。下次还会再来！",
    "真的太惊喜了！产品不仅质量好，而且功能比我想象的还要强大。客服也很贴心，提前告知了一些使用小技巧。物流也很快，两天就到了。五星好评，值得购买！",
    "性价比超高的一款产品！价格实惠，质量却不含糊。功能齐全，操作简单。客服也很热情，解答问题很及时。买了之后完全不后悔，推荐给大家！",
    "从产品的外观设计到实际使用，都无可挑剔。细节处理得很好，功能也很实用。客服服务周到，物流也很给力。五星好评，希望商家继续保持这样的品质！",
    "这款产品让我非常满意！质量好，性能稳定，功能丰富。客服也很专业，解答问题很详细。物流速度也很快。五星好评，下次还会继续支持！",
    "性价比很高的一款产品，价格实惠，质量却很好。功能齐全，操作也很方便。客服服务周到，物流也很及时。五星好评，值得购买！",
    "收到产品后真的太惊喜了！质量超出预期，功能也很强大。客服态度很好，物流速度也很快。五星好评，推荐给大家！",
    "从购买到使用，整个过程都非常顺利。产品质量好，功能齐全。客服服务也很贴心，物流也很给力。五星好评，下次还会再来！",
]

# 差评示例文本列表
negative_examples = [
    "随意降价，不予价保，服务态度差",
    "很失望，产品质量不太好。刚用不久就出现了问题，功能也不太稳定。客服态度也不好，解决问题很拖沓。希望商家能改进一下。",
    "物流速度太慢了，等了好久才收到。而且包装也很简陋，产品有些磨损。质量一般，功能也不太好用。不推荐购买。",
    "客服态度很差，问问题很久才回复，而且解决不了实际问题。产品质量也不好，功能很有限。物流也很慢，整体体验很差。不推荐。",
    "这款产品真的让我很失望！质量很差，功能也不实用。客服服务几乎没有，物流速度也很慢。一星都不想给！"
]

print("好评例子的分数 (>0)\n")
for i, example in enumerate(positive_examples):
    positive_score = get_score(get_embedding(example))
    print("%02d %s\n评分: %f\n" % (i + 1, example, positive_score))

print("\n差评例子的分数 (<0)\n")
for i, example in enumerate(negative_examples):
    negative_score = get_score(get_embedding(example))
    print("%02d %s\n评分: %f\n" % (i + 1, example, negative_score))

"""
这家餐馆太好吃了，一点都不糟糕 评分：0.081776
这家餐馆太糟糕了，一点都不好吃 评分：-0.340589

好评例子的分数 (>0)

01 买的银色版真的很好看，一天就到了，晚上就开始拿起来完系统很丝滑流畅，做工扎实，手感细腻，很精致哦苹果一如既往的好品质
评分: 0.175164

02 这款产品真的是超出预期！质量非常好，做工精细，用起来非常顺手。功能也很强大，完全满足了我的需求。客服也很耐心，有问题随时解答。五星好评，强烈推荐！
评分: 0.259196

03 这次购物体验太棒了！从下单到收货，速度飞快，包装严实。产品本身质量也很不错，性价比超高。客服态度特别好，解答问题很详细。下次还会再来！
评分: 0.293813

04 真的太惊喜了！产品不仅质量好，而且功能比我想象的还要强大。客服也很贴心，提前告知了一些使用小技巧。物流也很快，两天就到了。五星好评，值得购买！
评分: 0.233387

05 性价比超高的一款产品！价格实惠，质量却不含糊。功能齐全，操作简单。客服也很热情，解答问题很及时。买了之后完全不后悔，推荐给大家！
评分: 0.240070

06 从产品的外观设计到实际使用，都无可挑剔。细节处理得很好，功能也很实用。客服服务周到，物流也很给力。五星好评，希望商家继续保持这样的品质！
评分: 0.228930

07 这款产品让我非常满意！质量好，性能稳定，功能丰富。客服也很专业，解答问题很详细。物流速度也很快。五星好评，下次还会继续支持！
评分: 0.303972

08 性价比很高的一款产品，价格实惠，质量却很好。功能齐全，操作也很方便。客服服务周到，物流也很及时。五星好评，值得购买！
评分: 0.239161

09 收到产品后真的太惊喜了！质量超出预期，功能也很强大。客服态度很好，物流速度也很快。五星好评，推荐给大家！
评分: 0.233845

10 从购买到使用，整个过程都非常顺利。产品质量好，功能齐全。客服服务也很贴心，物流也很给力。五星好评，下次还会再来！
评分: 0.286375


差评例子的分数 (<0)

01 随意降价，不予价保，服务态度差
评分: -0.211911

02 很失望，产品质量不太好。刚用不久就出现了问题，功能也不太稳定。客服态度也不好，解决问题很拖沓。希望商家能改进一下。
评分: -0.260926

03 物流速度太慢了，等了好久才收到。而且包装也很简陋，产品有些磨损。质量一般，功能也不太好用。不推荐购买。
评分: -0.253379

04 客服态度很差，问问题很久才回复，而且解决不了实际问题。产品质量也不好，功能很有限。物流也很慢，整体体验很差。不推荐。
评分: -0.273339

05 这款产品真的让我很失望！质量很差，功能也不实用。客服服务几乎没有，物流速度也很慢。一星都不想给！
评分: -0.397410
"""
