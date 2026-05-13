# coding=utf-8
from sentence_transformers import SentenceTransformer
from google.colab import userdata
from huggingface_hub import login
import pandas as pd
import os

"""
生成产品数据文件
计算文件中每个产品的 Embedding
输出 csv 与 parquet 格式

dependency packages
pip install sentence-transformers
pip install pandas
"""
# Login HuggingFace Hub
login(token=userdata.get("HF_TOKEN"))

# 使用 SentenceTransformer 加载计算文本向量模型
embed_model = SentenceTransformer("google/embeddinggemma-300M")

# 计算多个文本的向量 (embedding)
def get_embeddings(list_of_texts, batch_size=32):
    return embed_model.encode(list_of_texts, batch_size=batch_size)


# 产品数据
product_data = [
    "【限时特惠】苹果iPhone 13 Pro 512G 高速5G手机",
    "【疯狂抢购】华为Mate 40 Pro 8G+256G 5G旗舰手机",
    "【热销爆款】小米11 Ultra 12G+256G 超大存储手机",
    "【新品上市】三星Galaxy S23 128G 5G全网通手机",
    "【直降500】vivo X80 Pro 12G+512G 专业摄影手机",
    "【评价如潮】OPPO Reno8 Pro+ 8GB+256GB 5G手机",
    "【抢先体验】Realme GT2 Pro 12G+256G 5G高性能手机",
    "【买就送壳】一加10 Pro 12GB+512GB 5G旗舰手机",
    "【学生专享】iPhone SE 3 128G 苹果手机 特惠中",
    "【最高配】小米12S Ultra 12G+512G 5G手机直降300",
    "【7天无理由退】华为P50 8G+128G 5G智能手机",
    "【特价促销】三星Note20 Ultra 12G+512G 5G手机",
    "【官方正品】vivo iQOO 12G+256G 5G手机限时抢",
    "【送礼必备】OPPO Find X5 Pro 12G+256G 5G手机",
    "【晒单返现】荣耀Magic4 Pro 8G+256G 5G旗舰机",
    "【独家优惠】红米Note 12 Pro 6G+128G 超值手机",
    "【满减优惠】苹果iPad Pro 12.9英寸 512G 平板电脑",
    "【新品首发】华为MatePad Pro 11 8GB+256GB 平板电脑",
    "【热销推荐】小米Pad 5 6GB+128GB 高性能平板电脑",
    "【限时7折】三星Galaxy Tab S8 8GB+128GB 平板电脑",
    "【买一送一】苹果AirPods Pro 无线降噪耳机",
    "【立减100】华为FreeBuds Pro 2 高清无线耳机",
    "【热销爆款】索尼WH-1000XM4 降噪无线耳机",
    "【限时特惠】JBL TUNE 225TWS 无线蓝牙耳机",
    "【超值套装】Bose QuietComfort 45 降噪耳机",
    "【新品尝鲜】小米Air 2 Pro 无线降噪耳机 直降50",
    "【限时秒杀】苹果MacBook Air M1 芯片 笔记本电脑",
    "【立省1000】华为MateBook 14 笔记本电脑 轻薄便携",
    "【开学季特惠】联想小新Pro 16 笔记本电脑 直降200",
    "【独家首发】戴尔XPS 13 笔记本电脑 超轻薄设计",
    "【限时抢购】惠普ENVY 13 笔记本电脑 学生特惠",
    "【性价比之王】宏碁Swift 3 笔记本电脑 立省300",
    "【旗舰新品】华硕ROG Zephyrus G15 游戏笔记本电脑",
    "【好评如潮】雷蛇Blade 15 游戏笔记本电脑 直降500",
    "【新品直降】外星人Area-51m R2 游戏笔记本电脑",
    "【买就送包】苹果iPhone 12 128G 5G手机 限时特惠",
    "【限时立减】三星Galaxy A53 5G 手机 学生专享价",
    "【大内存】vivo Y77 12GB+256GB 5G手机 特惠来袭",
    "【官方补贴】OPPO A97 8GB+128GB 5G手机 直降100",
    "【新品热销】荣耀60 8GB+256GB 5G手机 立省200",
    "【全网最低价】一加9R 5G手机 12GB+256GB 限时抢购",
    "【满减优惠】红米K50 Pro 12GB+256GB 5G手机 立减50",
    "【新品上架】iPhone 14 Pro 256G 5G手机 限时直降",
    "【官方正品】华为nova 9 8GB+128GB 5G手机 直降300",
    "【学生优惠】vivo S15 12GB+256GB 5G手机 特惠中",
    "【限时特惠】OPPO Reno9 Pro 8GB+256GB 5G手机",
    "【新品上市】realme GT Neo3 12GB+256GB 5G手机",
    "【限时立减】小米12X 8GB+256GB 5G手机 特惠来袭",
    "【热销推荐】三星Galaxy S22 FE 128GB 5G手机 直降100",
    "【独家优惠】魅族18S 8GB+256GB 5G手机 立省200",
    "【限时优惠】春季新款蕾丝衫 显瘦优雅 限时8折抢",
    "【热销爆款】时尚百搭小方包 多色可选 买就送挂件",
    "【限时直降】复古波点连衣裙 清新脱俗 满200减50",
    "【抢先体验】个性印花T恤 潮流必备 拍下立减20元",
    "【新品上市】韩版高腰牛仔裤 显高显瘦 新品特惠",
    "【特卖专场】精美刺绣手提包 优雅女性首选 全场满减",
    "【会员专享】复古风长款风衣 气质非凡 会员价立减",
    "【女神必备】简约风格高跟鞋 舒适时尚 买一送一",
    "【限时抢购】V领修身打底衫 性感迷人 限时买一送一",
    "【热销推荐】百变丝巾 多用途搭配 满额立减20元",
    "【限时折扣】甜美公主裙 少女心爆棚 限时9折抢购",
    "【限时优惠】高级感手提包 质感满分 抢先下单立减",
    "【新品首发】潮流拼接卫衣 个性十足 买就送配饰",
    "【限时折扣】时尚果冻包 清新可爱 满300减30",
    "【热销爆款】复古波点衬衫 优雅大方 热销疯抢",
    "【限时特惠】简约风格单鞋 舒适百搭 限时特卖中",
    "【新品上市】休闲风帆布鞋 多色可选 立减20元",
    "【限时抢购】荷叶边半身裙 甜美可爱 限时特价",
    "【热销推荐】时尚链条包 潮流必备 满额立减",
    "【限时特惠】气质款连衣裙 修身显瘦 限时9折",
    "【新品首发】个性图案背包 潮流先锋 买就送挂件",
    "【会员专享】修身小脚裤 时尚百搭 会员价更优",
    "【限时折扣】波西米亚风长裙 异域风情 满200减50",
    "【热销爆款】简约贝壳包 经典百搭 买就送小礼品",
    "【限时优惠】复古风风衣 气质优雅 抢先下单减20元",
    "【特卖专场】流苏装饰单鞋 时尚个性 限时优惠中",
    "【新品上市】潮流帆布包 多色可选 立享9折优惠",
    "【热销推荐】拼接设计打底衫 个性时尚 买一送一",
    "【限时抢购】水桶包 时尚新潮 限时特价疯抢",
    "【女神必备】金属扣高跟鞋 气质女神 买就送袜子",
    "【限时特惠】透视感雪纺衫 优雅迷人 限时8折优惠",
    "【新品首发】运动风背包 时尚休闲 买就送小礼品",
    "【会员专享】蕾丝拼接半裙 甜美风格 会员专享折扣",
    "【限时折扣】撞色设计风衣 个性十足 满额立减",
    "【热销爆款】简约风手拿包 时尚百搭 热销疯抢",
    "【限时优惠】V领针织衫 显瘦修身 抢先下单立减",
    "【新品上市】韩版休闲裤 多色可选 新品特惠",
    "【限时抢购】流苏装饰斜挎包 时尚新潮 限时优惠",
    "【热销推荐】简约圆领T恤 经典百搭 满300立减",
    "【限时折扣】高腰阔腿裤 时尚显瘦 限时9折优惠",
    "【新品首发】复古风围巾 优雅大方 独家发售",
    "【会员专享】丝绒高跟鞋 舒适时尚 会员价更优",
    "【限时抢购】皮质手环包 潮流必备 限时特价",
    "【热销爆款】宽松版卫衣 时尚舒适 热销疯抢",
    "【限时特惠】镂空设计半裙 甜美可爱 限时8折",
    "【新品上市】个性图案T恤 潮流先锋 买就送配饰",
    "【限时抢购】链条装饰小方包 时尚新潮 限时优惠",
    "【热销推荐】修身款牛仔裤 显瘦显高 满额立减",
    "【限时优惠】简约风格长款钱包 时尚百搭 抢先下单立减",
    "【品牌直降】高端鳄鱼纹手提包 品牌直降 买就赚"
]

# 创建 Data Frame
df = pd.DataFrame(product_data, columns=['product_name'])

# 计算每个产品的 Embedding
embeddings = get_embeddings(df['product_name'].tolist())

# 将结果转为列表存入 DataFrame
df["embedding"] = list(embeddings)

# 输出路径
output_dir = './data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# csv 文件
csv_file = os.path.join(output_dir, 'product_data.csv')
df.to_csv(csv_file, index=False, encoding='utf_8_sig')
print(f"CSV 已保存：{csv_file}")