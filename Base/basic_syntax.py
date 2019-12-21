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

# 字符串拼接,转义,重复
print('this' + ' is ' + 'string', end="\n")
print('this is string \n', end="")
print(r'this is string \n', end="\n")
print('fdipzone ' * 2, end="\n\n")

# 字符串长度，截取
str = 'abcdefghi'
str_len = len(str)
print('str len: %d' %(len(str)), end="\n")
print(str[0:4], end="\n")
print(str[4:8], end="\n")
print(str[str_len-4:str_len-1], end="\n\n")

# 多行字符串
str = '''
故人西辞黄鹤楼，
烟花三月下扬州。
孤帆远影碧空尽，
唯见长江天际流。
'''
print(str, end="\n\n")

# 列表（数组）
list = [1,2,3,4]
print('list len: %d' %(len(list)))
print(list)
print(list[0],list[1],list[2], end="\n\n")

lists = [[1,2,3,4],[5,6,7,8]]
print('lists len: %d' %(len(lists)))
print(lists)
print(lists[0][0],lists[0][1],lists[0][2],lists[0][3])
print(lists[1][0],lists[1][1],lists[1][2],lists[1][3], end="\n\n")

# 修改列表元素
lists[1] = [9,10,11,12]
print(lists)

# 增加列表元素
lists.insert(1, ['a','b','c','d'])
print(lists)

# 追加列表元素
lists.append([13,14,15,16])
print(lists)

# 删除列表元素
del lists[1]
print(lists)
