# coding=utf-8
from openai import OpenAI
from google.colab import userdata
import pandas as pd
import numpy as np
import re
import inspect

# 显示所有行
pd.set_option('display.max_rows', None)

# 显示所有列
pd.set_option('display.max_columns', None)

# 单元格内容完整显示，不截断
pd.set_option('display.max_colwidth', None)

# 提高换行阈值，防止自动换行
pd.set_option('display.width', 1000)

"""
AI 产品数据生成器

dependency packages
pip install openai
pip install pandas
pip install numpy
"""
client = OpenAI(
    api_key=userdata.get("KIMI_API_KEY"),
    base_url="https://api.moonshot.cn/v1"
)

COMPLETION_MODEL = "moonshot-v1-8k"

# 产品数据生成器
class ProductDataGenerator:
    def __init__(self):
        self.model = COMPLETION_MODEL

        # 使用 cleandoc 自动去除多余的首行空行和缩进
        self.system_prompt = inspect.cleandoc("""
            你是一个熟悉中国电商平台（淘宝、京东等）的运营专家，擅长撰写吸引消费者的商品标题，了解各品类的促销话术和用户心理。
            输出格式：【促销类型】商品标题
        """)

    def generate(self, prompt):
        try:
            completions = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                n=1,
                stop=None,
                temperature=0.8
            )
        except Exception as e:
            raise e

        message = completions.choices[0].message
        return message.content


# 清洗生成的产品数据
def format_product_data(data):
    # 将数据放入 data frame
    product_names = data.strip().split('\n')
    df = pd.DataFrame({'product_name': product_names})

    # 去除返回结果的标号
    df.product_name = df.product_name.apply(
        lambda x: re.sub(r'^\d+[\.\s、]+', '', x).strip())

    return df


# 提示词
prompts = [
    """请你生成 50 条淘宝网里的商品的标题，每条在 30 个字左右，品类是3C数码产品，标题里往往也会有一些促销类的信息，每行一条。""",
    """请你生成 50 条淘宝网里的商品的标题，每条在 30 个字左右，品类是女性的服饰箱包等等，标题里往往也会有一些促销类的信息，每行一条。"""
]

# Data Frames
dfs = []

# 生成产品数据
generator = ProductDataGenerator()

for prompt in prompts:
    product_data = generator.generate(prompt)
    dfs.append(format_product_data(product_data))

# 合并数据
total_df = pd.concat(dfs, ignore_index=True)

# 索引从 1 开始
total_df.index = np.arange(1, len(total_df) + 1)

# 显示数据集
display(total_df)


"""
    product_name
1   【限时特惠】苹果iPhone 13 Pro 512G 高速5G手机
2   【疯狂抢购】华为Mate 40 Pro 8G+256G 5G旗舰手机
3   【热销爆款】小米11 Ultra 12G+256G 超大存储手机
4   【新品上市】三星Galaxy S23 128G 5G全网通手机
5   【直降500】vivo X80 Pro 12G+512G 专业摄影手机
6   【评价如潮】OPPO Reno8 Pro+ 8GB+256GB 5G手机
7   【抢先体验】Realme GT2 Pro 12G+256G 5G高性能手机
8   【买就送壳】一加10 Pro 12GB+512GB 5G旗舰手机
9   【学生专享】iPhone SE 3 128G 苹果手机 特惠中
10  【最高配】小米12S Ultra 12G+512G 5G手机直降300
11  【7天无理由退】华为P50 8G+128G 5G智能手机
12  【特价促销】三星Note20 Ultra 12G+512G 5G手机
13  【官方正品】vivo iQOO 12G+256G 5G手机限时抢
14  【送礼必备】OPPO Find X5 Pro 12G+256G 5G手机
15  【晒单返现】荣耀Magic4 Pro 8G+256G 5G旗舰机
16  【独家优惠】红米Note 12 Pro 6G+128G 超值手机
17  【满减优惠】苹果iPad Pro 12.9英寸 512G 平板电脑
18  【新品首发】华为MatePad Pro 11 8GB+256GB 平板电脑
19  【热销推荐】小米Pad 5 6GB+128GB 高性能平板电脑
20  【限时7折】三星Galaxy Tab S8 8GB+128GB 平板电脑
21  【买一送一】苹果AirPods Pro 无线降噪耳机
22  【立减100】华为FreeBuds Pro 2 高清无线耳机
23  【热销爆款】索尼WH-1000XM4 降噪无线耳机
24  【限时特惠】JBL TUNE 225TWS 无线蓝牙耳机
25  【超值套装】Bose QuietComfort 45 降噪耳机
26  【新品尝鲜】小米Air 2 Pro 无线降噪耳机 直降50
27  【限时秒杀】苹果MacBook Air M1 芯片 笔记本电脑
28  【立省1000】华为MateBook 14 笔记本电脑 轻薄便携
29  【开学季特惠】联想小新Pro 16 笔记本电脑 直降200
30  【独家首发】戴尔XPS 13 笔记本电脑 超轻薄设计
31  【限时抢购】惠普ENVY 13 笔记本电脑 学生特惠
32  【性价比之王】宏碁Swift 3 笔记本电脑 立省300
33  【旗舰新品】华硕ROG Zephyrus G15 游戏笔记本电脑
34  【好评如潮】雷蛇Blade 15 游戏笔记本电脑 直降500
35  【新品直降】外星人Area-51m R2 游戏笔记本电脑
36  【买就送包】苹果iPhone 12 128G 5G手机 限时特惠
37  【限时立减】三星Galaxy A53 5G 手机 学生专享价
38  【大内存】vivo Y77 12GB+256GB 5G手机 特惠来袭
39  【官方补贴】OPPO A97 8GB+128GB 5G手机 直降100
40  【新品热销】荣耀60 8GB+256GB 5G手机 立省200
41  【全网最低价】一加9R 5G手机 12GB+256GB 限时抢购
42  【满减优惠】红米K50 Pro 12GB+256GB 5G手机 立减50
43  【新品上架】iPhone 14 Pro 256G 5G手机 限时直降
44  【官方正品】华为nova 9 8GB+128GB 5G手机 直降300
45  【学生优惠】vivo S15 12GB+256GB 5G手机 特惠中
46  【限时特惠】OPPO Reno9 Pro 8GB+256GB 5G手机
47  【新品上市】realme GT Neo3 12GB+256GB 5G手机
48  【限时立减】小米12X 8GB+256GB 5G手机 特惠来袭
49  【热销推荐】三星Galaxy S22 FE 128GB 5G手机 直降100
50  【独家优惠】魅族18S 8GB+256GB 5G手机 立省200
51  【限时优惠】春季新款蕾丝衫 显瘦优雅 限时8折抢
52  【热销爆款】时尚百搭小方包 多色可选 买就送挂件
53  【限时直降】复古波点连衣裙 清新脱俗 满200减50
54  【抢先体验】个性印花T恤 潮流必备 拍下立减20元
55  【新品上市】韩版高腰牛仔裤 显高显瘦 新品特惠
56  【特卖专场】精美刺绣手提包 优雅女性首选 全场满减
57  【会员专享】复古风长款风衣 气质非凡 会员价立减
58  【女神必备】简约风格高跟鞋 舒适时尚 买一送一
59  【限时抢购】V领修身打底衫 性感迷人 限时买一送一
60  【热销推荐】百变丝巾 多用途搭配 满额立减20元
61  【限时折扣】甜美公主裙 少女心爆棚 限时9折抢购
62  【限时优惠】高级感手提包 质感满分 抢先下单立减
63  【新品首发】潮流拼接卫衣 个性十足 买就送配饰
64  【限时折扣】时尚果冻包 清新可爱 满300减30
65  【热销爆款】复古波点衬衫 优雅大方 热销疯抢
66  【限时特惠】简约风格单鞋 舒适百搭 限时特卖中
67  【新品上市】休闲风帆布鞋 多色可选 立减20元
68  【限时抢购】荷叶边半身裙 甜美可爱 限时特价
69  【热销推荐】时尚链条包 潮流必备 满额立减
70  【限时特惠】气质款连衣裙 修身显瘦 限时9折
71  【新品首发】个性图案背包 潮流先锋 买就送挂件
72  【会员专享】修身小脚裤 时尚百搭 会员价更优
73  【限时折扣】波西米亚风长裙 异域风情 满200减50
74  【热销爆款】简约贝壳包 经典百搭 买就送小礼品
75  【限时优惠】复古风风衣 气质优雅 抢先下单减20元
76  【特卖专场】流苏装饰单鞋 时尚个性 限时优惠中
77  【新品上市】潮流帆布包 多色可选 立享9折优惠
78  【热销推荐】拼接设计打底衫 个性时尚 买一送一
79  【限时抢购】水桶包 时尚新潮 限时特价疯抢
80  【女神必备】金属扣高跟鞋 气质女神 买就送袜子
81  【限时特惠】透视感雪纺衫 优雅迷人 限时8折优惠
82  【新品首发】运动风背包 时尚休闲 买就送小礼品
83  【会员专享】蕾丝拼接半裙 甜美风格 会员专享折扣
84  【限时折扣】撞色设计风衣 个性十足 满额立减
85  【热销爆款】简约风手拿包 时尚百搭 热销疯抢
86  【限时优惠】V领针织衫 显瘦修身 抢先下单立减
87  【新品上市】韩版休闲裤 多色可选 新品特惠
88  【限时抢购】流苏装饰斜挎包 时尚新潮 限时优惠
89  【热销推荐】简约圆领T恤 经典百搭 满300立减
90  【限时折扣】高腰阔腿裤 时尚显瘦 限时9折优惠
91  【新品首发】复古风围巾 优雅大方 独家发售
92  【会员专享】丝绒高跟鞋 舒适时尚 会员价更优
93  【限时抢购】皮质手环包 潮流必备 限时特价
94  【热销爆款】宽松版卫衣 时尚舒适 热销疯抢
95  【限时特惠】镂空设计半裙 甜美可爱 限时8折
96  【新品上市】个性图案T恤 潮流先锋 买就送配饰
97  【限时抢购】链条装饰小方包 时尚新潮 限时优惠
98  【热销推荐】修身款牛仔裤 显瘦显高 满额立减
99  【限时优惠】简约风格长款钱包 时尚百搭 抢先下单立减
100 【品牌直降】高端鳄鱼纹手提包 品牌直降 买就赚
"""
