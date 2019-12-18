#!/usr/local/bin/python3
# coding=utf-8
# 基础语法学习

# 打印Hello World及中文世界你好
print("Hello World, 世界你好！", end="\n\n")

# 输出保留字
import keyword

print(keyword.kwlist, end="\n\n")

# 单行注释

'''
多行注释
多行注释
多行注释
'''

# 等待用户输入
content = input("请输入姓名并按 Enter 确定\n")
print("你的姓名是：" + content, end="\n\n")

# 同一行写多条语句
import sys; x = 'hello world'; sys.stdout.write(x + ",good bye world\n\n")

# 生成随机数
import random

random_num1 = random.choice(range(10))  # 随机0~9
random_num2 = random.randrange(10, 200) # 随机 10~200
random_num3 = random.random()  # 随机 0~1
print('random_nums: %d %d %f' %(random_num1,random_num2,random_num3), end="\n\n")

# 数学常量
import math
print('圆周率：%.40f' %(math.pi), end="\n\n")
print('自然常数：%.20f' %(math.e), end="\n\n")
