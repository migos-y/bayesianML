from abc import ABCMeta, abstractmethod
from scipy import stats

class Predictive(metaclass=ABCMeta):
    predictive = None
    @abstractmethod
    def __init__(self, predictive) -> None:
        self.predictive = predictive
    def predict(self, x):
        return self.predictive(x)

class Predictive_Ber(Predictive):
    theta = None
    predictive = None
    def __init__(self, theta):
        self.theta = theta
        def predictive(x):
            return stats.bernoulli.pmf(x, theta)
        self.predictive = predictive
    def predict(self, x):
        return super().predict(x)

class Predictive_Cat(Predictive):
    pi = None
    predictive = None
    def __init__(self, pi):
        self.pi = pi
        def predictive(s):
            return stats.multinomial.pmf(x=s, n=1, p = pi)
        self.predictive = predictive
    def predict(self, x):
        return super().predict(x)

class Predictive_NB(Predictive):
    r = 1
    p = 0.5
    def __init__(self, r, p):
        self.r = r
        self.p = p
        def predictive(x):
            return stats.nbinom.pmf(x, r, p)
        self.predictive = predictive
    def predict(self, x):
        return super().predict(x)