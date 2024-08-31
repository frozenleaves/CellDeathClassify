import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sympy as sp
import logging
import seaborn as sns

class Model(object):
    def __init__(self, model=None, popt=None):
        self.model = model
        if not self.model:
            self.model = self._model
        self.popt = popt

    def _model(self, x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    def derivative(self, L, k, x0):
        """对sigmoid函数求导，算出每个点的瞬时死亡率，返回其一阶导函数"""
        x = sp.symbols('x')
        fx = L / (1 + sp.exp(-k * (x - x0)))
        dx = sp.diff(fx, x) # x对fx的一阶导数
        print(dx)
        return dx

    def calc_derivative(self, X, dx):
        tmp = []
        x = sp.symbols('x')
        for i in X:
            values = dx.evalf(subs={x: i})
            tmp.append(float(values))
        sns.set_style("whitegrid")
        sns.lineplot(x = 'frame', y = 'ratio', data=pd.DataFrame({'ratio': tmp, 'frame': X}))
        plt.xlabel('frame')
        plt.ylabel('death ratio')
        plt.show()

    def fit(self, X, y):
        initial_guess = [max(y), 1, np.median(X)]
        # 拟合模型
        popt, pcov = curve_fit(self.model, X, y, p0=initial_guess, maxfev=500000)
        L_fit, k_fit, x0_fit = popt
        logging.info(f'L: {L_fit} \tk: {k_fit} \tx0: {x0_fit}')
        self.popt = popt
        self.pcov = pcov
        return popt, pcov

    def predict(self, X):
        if self.popt is None:
            raise RuntimeError("参数尚未拟合")
        X_fit = np.linspace(min(X), max(X), 1000)
        y_fit = self.model(X_fit, *self.popt)
        return X_fit, y_fit

    def plot(self, x, y):
        X_fit, y_fit = self.predict(x)
        plt.scatter(x, y, s=5, color='blue', label='Data')     # 图例
        plt.plot(X_fit, y_fit, color='red', label='Sigmoid fit')  # 图例
        plt.xlabel('X') # X 轴名称
        plt.ylabel('y') # Y轴名称
        plt.legend()
        plt.show()

def sigmoid(x, L, k, x0):
    """定义sigmoid曲线拟合模型"""
    return L / (1 + np.exp(-k * (x - x0)))


def predict(X, model: Model):
    X_fit = np.linspace(min(X), max(X), 1000)
    X_fit, y_fit = model.predict(X_fit)
    return X_fit, y_fit


def fit(path_list: list, label_list):
    assert len(path_list) == len(label_list)
    results = {}  # path: [x, y, model]
    for path in path_list:
        df = pd.read_csv(path)
        y = df['death_count'] / df['all_count']
        y = np.array(y.values)
        x = np.array(list(range(1, len(y) + 1)))
        model = Model()
        model.fit(x, y)
        results[path] = [x, y, model]

    for item in results:
        x_fit, y_fit = predict(results[item][0], results[item][2])
        results[item].extend([x_fit, y_fit])
    for item in zip(path_list, label_list):
        x_fit = results[item[0]][-2]
        y_fit = results[item[0]][-1]
        label = item[1]
        plt.plot(x_fit, y_fit, label=label)  # 图例
        # plt.scatter(results[item[0]][0], results[item[0]][1], s=5,  label=f'Data-{label}')     # 绘制数据散点图
    plt.xlabel('Frame')  # X 轴名称
    plt.ylabel('Death Rate')  # Y轴名称
    plt.legend()
    plt.savefig(r'C:\Users\wjq\Desktop\demo.pdf')
    plt.show()


model = Model()
path = r"H:\20240729rpe-pcna-y530f+wtrpe-pcna-src-y530f-500-30-10-ctrl-wt-50-30-10-ctrl-per3slide\0\copy_of_rpe=pcna-src-y530f-500-30-10-ctrl-wt-50-30-10-ctrl-per3slidend2001_0_pcna.csv"
path3 = r"H:\20240729rpe-pcna-y530f+wtrpe-pcna-src-y530f-500-30-10-ctrl-wt-50-30-10-ctrl-per3slide\3\copy_of_rpe=pcna-src-y530f-500-30-10-ctrl-wt-50-30-10-ctrl-per3slidend2001_3_pcna.csv"
path5 = r"H:\20240729rpe-pcna-y530f+wtrpe-pcna-src-y530f-500-30-10-ctrl-wt-50-30-10-ctrl-per3slide\6\copy_of_rpe=pcna-src-y530f-500-30-10-ctrl-wt-50-30-10-ctrl-per3slidend2001_6_pcna.csv"
# df = pd.read_csv(path)
# y = df['death_count'] / df['all_count']
# print(y.values)
# y = y.values
# y = np.array(y)
# xx = list(range(1, len(y) + 1))
# xx = np.array(xx)


fit([path, path3, path5], ['500ng/ml', '30ng/ml', '10ng/ml'])
# model.fit(xx, y)
# model.plot(xx, y)
# # dx = model.derivative(*model.popt)
# model.calc_derivative(list(range(1, len(y) + 1)), dx)