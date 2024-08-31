import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


data = pd.read_csv(r'C:\Users\wjq\Desktop\test-data.csv')
data2 = pd.read_csv(r'C:\Users\wjq\Desktop\test-data2.csv')

# 取出x值和y值
x = data['x']
y1 = data['y1']
y2 = data['y2']
y3 = data['y3']

# 计算y值的平均值和标准差
y_mean = np.mean([y1, y2, y3], axis=0)
y_std = np.std([y1, y2, y3], axis=0)

# 使用Savitzky-Golay平滑滤波器平滑y_mean
window_length = 155  # 窗口大小，必须是正奇数
polyorder = 2      # 多项式阶数
y_smooth = savgol_filter(y_mean, window_length, polyorder)
y_std_smooth = savgol_filter(y_std, window_length, polyorder)
# 绘制曲线图
plt.figure(figsize=(10, 6))
plt.plot(x, y_smooth, label='500 Death ratio', color='blue')
# 绘制误差阴影
plt.fill_between(x, y_smooth - y_std_smooth, y_smooth + y_std_smooth, color='blue', alpha=0.2, label='500 Std Dev')


x2 = data2['x']
y2_1 = data2['y1']
y2_2 = data2['y2']
y2_3 = data2['y3']

# 计算第二组数据的y值的平均值和标准差
y2_mean = np.mean([y2_1, y2_2, y2_3], axis=0)
y2_std = np.std([y2_1, y2_2, y2_3], axis=0)

# 使用Savitzky-Golay平滑滤波器平滑第二组数据
y2_smooth = savgol_filter(y2_mean, window_length, polyorder)
y2_std_smooth = savgol_filter(y2_std, window_length, polyorder)


plt.plot(x2, y2_smooth, label='30 Death ratio', color='red')
plt.fill_between(x2, y2_smooth - y2_std_smooth, y2_smooth + y2_std_smooth, color='red', alpha=0.2, label='30 Std Dev')


# 添加图例
plt.legend(loc='lower right')



# 添加标题和轴标签
plt.title('death info')
plt.xlabel('frame')
plt.ylabel('death rate')
plt.show()