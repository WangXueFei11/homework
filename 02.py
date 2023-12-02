# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
# 生成100个服从标准正态分布的样本
samples = np.random.randn(100)
# 绘制样本图
plt.hist(samples, bins=20, density=True, alpha=0.6, color='g')
# 添加正态分布曲线
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = np.exp(-np.power(x - 0, 2) / (2 * 1**2)) / np.sqrt(2 * np.pi * 1**2)
plt.plot(x, p, 'r', alpha=0.7)
plt.show()
