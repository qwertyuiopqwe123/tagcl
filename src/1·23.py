
import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ['$L_{sup}+L_{GL}$', '$L_{sup}+L_{GL}+L_{SC}$', '$L_{sup}+L_{GL}+L_{NC}$', '$L_{Tc-SCL}$']
values = [66.87, 70.38, 68.21, 72.82]  # 这里有超过 70 的值

# 创建柱状图
plt.bar(categories, values, color='green')

# 设置 y 轴上限为 100
plt.ylim(65, 74)

# 设置 y 轴刻度间隔
y_ticks = np.arange(65, 74, 1)  # 0 到 100，间隔为 10
plt.yticks(y_ticks)

# 添加标题和标签
plt.title('PubMed')
plt.ylabel('Accuracy')

# 显示图表
plt.show()