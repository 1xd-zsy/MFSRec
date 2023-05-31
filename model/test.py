import json
import os
import sys

# curPath = os.path.abspath(os.path.dirname('__file__'))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootPath)
#
# with open(rootPath + '/data/mashup_description.json', 'r') as f:
#     mashup_description = json.load(f)
# with open(rootPath + '/data/used_api_description.json', 'r') as f:
#     api_description = json.load(f)
# with open(rootPath + '/data/' + 'train_mashup_api.json', 'r') as f:
#     train_mashup_api = json.load(f)
# with open(rootPath + '/data/' + 'test_mashup_api.json', 'r') as f:
#     test_mashup_api = json.load(f)
#
# print("mashup_description",len(mashup_description))
# print("api_description",len(api_description))
# print("train_mashup_api",len(train_mashup_api))
# print("test_mashup_api",len(test_mashup_api))
# print("all",len(train_mashup_api)+len(test_mashup_api))
import matplotlib.pyplot as plt

# 示例数据
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# 创建折线图
plt.plot(x, y, marker='o')

# 设置图表标题和坐标轴标签
plt.title('折线图示例')
plt.xlabel('X轴')
plt.ylabel('Y轴')

# 显示网格线
plt.grid(True)

# 显示图表
plt.show()

