from scipy.linalg import norm, pinv
from matplotlib import pyplot as plt
import numpy as np  # 统一用numpy处理数组，避免混用random和np.random

class RBF:
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        # 修正：用numpy生成多维随机中心（替换random.uniform）
        self.centers = [np.random.uniform(-1, 1, indim) for _ in range(numCenters)]
        self.beta = 8
        self.W = np.random.random((self.numCenters, self.outdim))  # 用np.random更兼容数组操作

    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return np.exp(-self.beta * norm(c - d) **2)  # 显式用np.exp确保数组支持

    def _calcAct(self, X):
        G = np.zeros((X.shape[0], self.numCenters), float)  # 用np.zeros
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._basisfunc(c, x)
        return G

    def train(self, X, Y):
        # 从训练集随机选择中心（确保X是numpy数组）
        rnd_idx = np.random.permutation(X.shape[0])[:self.numCenters]  # 用np.random.permutation
        self.centers = [X[i, :] for i in rnd_idx]
        print("centers:", self.centers)  # 修正print语法
        G = self._calcAct(X)
        print("Activation matrix G:\n", G)
        self.W = np.dot(pinv(G), Y)  # 用np.dot确保矩阵运算兼容

    def test(self, X):
        G = self._calcAct(X)
        Y = np.dot(G, self.W)
        return Y


if __name__ == '__main__':
    n = 100
    # 修正：用complex()生成n个点的间隔，替换complex_
    x = np.mgrid[-1:1:complex(0, n)].reshape(n, 1)  # 显式用np.mgrid
    y = np.sin(3 * (x + 0.5)** 3 - 1)  # 用np.sin支持数组运算

    # 初始化RBF模型（1维输入，10个中心，1维输出）
    rbf = RBF(1, 10, 1)
    rbf.train(x, y)
    z = rbf.test(x)

    # 绘图
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, 'k-', label='Original data')
    plt.plot(x, z, 'r-', linewidth=2, label='RBF prediction')
    plt.plot(rbf.centers, np.zeros(rbf.numCenters), 'gs', label='RBF centers')

    # 绘制RBF基函数
    for c in rbf.centers:
        cx = np.arange(float(c) - 0.7, float(c) + 0.7, 0.01)  # 确保c为标量
        # 修正：用实例名rbf调用方法，而非self
        cy = [rbf._basisfunc(np.array([cx_]), np.array([c])) for cx_ in cx]
        plt.plot(cx, cy, '-', color='gray', linewidth=0.2)

    plt.xlim(-1.2, 1.2)
    plt.legend()
    plt.show()