from abc import ABCMeta, abstractmethod
from scipy import stats


class Model(metaclass=ABCMeta):
    @abstractmethod
    def model_calP(self, x, theta):
        pass



class model_Bernoulli(Model):
    def model_calP(self, x, theta):
        return stats.bernoulli.pmf(x,theta,loc=0)

class model_Categorical(Model):
    def model_calP(self, s, pi):
        return stats.multinomial.pmf(x=s, n=1, p = pi)

class model_Poisson(Model):
    def model_calP(self, x, lam):
        return stats.poisson.pmf(x, lam)