import matplotlib.pyplot as plt

# 示例数量（实际值）
actual_examples = [1, 3, 5, 10]

# 用于均匀分布的虚拟索引
uniform_x = range(len(actual_examples))  # [0, 1, 2, 3]

# 各方法的 BLEU 分数
Perfective = [21.41, 23.65, 24.29, 25.39]
Adaptive = [23.35, 24.31, 25.08, 25.66]
Corrective = [25.22, 26.21, 26.82, 26.91]

plt.rcParams.update({'font.size': 14})

# 颜色和样式
plt.plot(uniform_x, Perfective, 'r-s', label="Perfective")  # 红色，方块标记
plt.plot(uniform_x, Adaptive, 'y-o', label="Adaptive")  # 黑色，圆形标记
plt.plot(uniform_x, Corrective, 'b-^', label="Corrective")  # 蓝色，三角形标记

# 设定新的刻度位置 & 标签
plt.xticks(uniform_x, actual_examples)  # 让 1, 3, 5, 10 均匀分布

# 图表标签和标题
plt.xlabel("Example", fontsize=16)
plt.title("METEOR", fontsize=16)

# 显示图例
plt.legend(fontsize=13)

# 显示网格
# plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig("meteor_java_plot.png", dpi=300, bbox_inches='tight')

# 显示图像
plt.show()
