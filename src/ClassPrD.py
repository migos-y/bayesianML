from abc import ABCMeta, abstractmethod
from matplotlib import projections
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np


## class for prior and posterior
class PrD(metaclass=ABCMeta):
    prd = None
    def __init__(self, prd) -> None:
        self.prd = prd
    @abstractmethod
    def calP(self, theta):
        return self.prd(theta)


class Beta(PrD):
    a = None
    b = None
    prd = None

    def __init__(self, a,b):
        self.a = a
        self.b = b
        def p(theta):
            return stats.beta.pdf(theta, self.a, self.b, loc=0, scale = 1)
        self.prd = p

    def calP(self, theta):
        return super().calP(theta)

    def plot(self, title):
        x = np.arange(0.01, 1,0.01)
        y = self.prd(x)

        plt.plot(x,y)
        plt.title(title)
        plt.xlim(0,1)
        plt.show()

class Dir(PrD):
    alpha = None
    dim = 0
    def __init__(self, alpha):
        self.alpha = alpha
        self.dim = len(alpha)
        def p(pi):
            return stats.dirichlet.pdf(pi, alpha)
        self.prd = p
    def calP(self, pi):
        return super().calP(pi)
    def plot(self, title):
        if self.dim <= 1 or self.dim >= 4:
            return
        elif self.dim == 2:
            
            x1 = np.arange(0.01, 0.99, 0.01)
            x2 = 1-x1
            x = np.stack((x1, x2), axis = 1)
            n = len(x1)
            y = np.empty(n)
            for i in range(n):
                y[i] = self.prd(x[i])

            plt.plot(x1,y)
            plt.title(title)
            plt.xlim(0,1)
            plt.xlabel(r"$\pi _1$")
            plt.show()
        else:
            x1 = np.arange(0.01, 0.99, 0.01)
            x2 = np.arange(0.01, 0.99, 0.01)
            x, y= np.meshgrid(x1, x2)
            x[x + y > 0.99] = 0.01
            y[x + y > 0.99] = 0.01 # 参考:https://cartman0.hatenablog.com/entry/2021/02/27/%E3%83%87%E3%82%A3%E3%83%AA%E3%82%AF%E3%83%AC%28Dirichlet%29%E5%88%86%E5%B8%83%E3%82%923D%E3%81%A7%E5%8F%AF%E8%A6%96%E5%8C%96%E3%81%99%E3%82%8B
            X = x.flatten()
            Y = y.flatten()
            n = len(X)
            Z = np.empty(n)
            for i in range(n):
                Z[i] = self.prd(np.array([X[i], Y[i], 1-X[i]-Y[i]]))
            z = Z.reshape(x.shape)
            ax3d = plt.axes(projection='3d')
            ax3d.plot_surface(x, y, z,cmap='plasma')
            ax3d.set_xlabel(r"$\pi _1$")
            ax3d.set_ylabel(r"$\pi _2$")
            ax3d.set_zlabel(r"Dir($\pi , \alpha$)")
            ax3d.set_title(title)
            plt.show()

class Gam(PrD):
    a = 0.5
    b = 0.5
    def __init__(self, a, b):
        self.a = a
        self.b = b
        def p(lam):
            return stats.gamma.pdf(lam, a,scale=1./b)
        self.prd = p

    def calP(self, lam):
        return super().calP(lam)
    
    def plot(self, title):
        x = np.arange(0.01, 20,0.01)
        y = self.prd(x)

        plt.plot(x,y)
        plt.title(title)
        plt.xlim(0,20)
        plt.show()
