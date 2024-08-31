import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import PathPatch
import numpy as np

def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """
    # iterating through Axes instances
    for ax in g.axes:
        # iterating through axes artists:
        for c in ax.get_children():
            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                # setting new width of box
                xmin_new = xmid - fac * xhalf
                xmax_new = xmid + fac * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new



# 读取CSV文件
data1 = pd.read_csv(r'C:\Users\wjq\Desktop\test-data.csv')
data2 = pd.read_csv(r'C:\Users\wjq\Desktop\test-data2.csv')
# 计算y1, y2, y3的平均值
data1['average'] = data1[['y1', 'y2', 'y3']].mean(axis=1)
data2['average'] = data2[['y1', 'y2', 'y3']].mean(axis=1)

# 将数据分组，每组数据的范围是0-100, 101-200, ...
bins = [0, 100, 200, 300, 400, 500, 600]
labels = ['0-100', '101-200', '201-300', '301-400', '401-500', '501-600']
data1['group'] = pd.cut(data1['x'], bins=bins, labels=labels, right=False)
data2['group'] = pd.cut(data2['x'], bins=bins, labels=labels, right=False)

# 添加一个新列来区分两个数据文件
data1['file'] = '500 ng/ml'
data2['file'] = '30 ng/ml'

# 合并数据
data_combined = pd.concat([data1, data2])

# 绘制箱线图

sns.set(style="whitegrid")
plt.style.use('ggplot')
# 绘制箱线图
fig = plt.figure(figsize=(14, 8))  # 调整图表大小
bp = sns.boxplot(x='group', y='average', hue='file', data=data_combined, palette='Set3', width=0.6, fliersize=3, dodge=60)
plt.title('Grouped Box Plot of Average Values', fontsize=16)  # 调整标题字体大小
plt.xlabel('frame', fontsize=14)  # 调整x轴标签字体大小
plt.ylabel('death ratio', fontsize=14)  # 调整y轴标签字体大小
plt.legend(loc='lower right', title_fontsize='13', fontsize='12')  # 调整图例样式
plt.xticks(rotation=0, fontsize=12)  # 调整x轴刻度标签旋转角度和字体大小
plt.yticks(fontsize=12)  # 调整y轴刻度标签字体大小
# plt.grid(True)  # 显示网格

# 调整不同bin之间的间隔
plt.subplots_adjust(hspace=0.5)  # 调整子图之间的垂直间隔

plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域

ax = plt.gca()  # 获得坐标轴的句柄
# ax.yaxis.set_major_locator(plt.MultipleLocator(multiplier / 10))  # 以每(multiplier / 10)间隔显示
# bp.set(xlabel=None)
# bp.set(ylabel=None)
adjust_box_widths(fig, 0.6)


# 显示图表
plt.show()