import numpy as np
from scipy import stats
import ClassModel
import ClassPrD
import ClassPredictive

class BayesianML():
    model = None
    prior = None
    def __init__(self, model, prior):
         self.model = model
         self.prior = prior

    # train self with trainingData and return joint probability distribution
    def train(self, trainingData):

        # define likelihood funciton likelihood_function(theta, trainingData) = P(trainingData| theta)
        def likelihood_function(theta):
            likelihood = 1
            for x in trainingData:
                likelihood = likelihood * self.model.model_calP(x,theta)
            return likelihood

        # define joint function jointPD(theta, trainingData) = P(trainingData, theta)
        def joint_PD(theta):
            return likelihood_function(theta) * self.prior.calP(theta)

        return ClassPrD.PrD(joint_PD)

# 返されるのは同時分布ではなく、事後分布
class Ber_Beta(BayesianML):
    a = 0
    b = 0
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.model = ClassModel.model_Bernoulli()
        self.prior = ClassPrD.Beta(a, b)
    def train(self, trainingData):
        n = len(trainingData)
        sum = trainingData.sum()
        a_hat = sum + self.a
        b_hat = n - sum + self.b
        pre_theta = a_hat/(a_hat + b_hat)
        return (ClassPrD.Beta(a_hat, b_hat),ClassPredictive.Predictive_Ber(pre_theta))

class Cat_Dir(BayesianML):
    alpha = None
    def __init__(self, alpha):
        self.alpha = alpha
        self.model = ClassModel.model_Categorical()
        self.prior = ClassPrD.Dir(alpha)
    
    def train(self, trainingData):
        alpha_hat = np.sum(np.block([[trainingData], [self.alpha]]), axis=0)
        pre_pi = alpha_hat / np.sum(alpha_hat)
        return (ClassPrD.Dir(alpha_hat), ClassPredictive.Predictive_Cat(pre_pi))

class Poi_Gam(BayesianML):
    a = 0.5
    b = 0.5
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.model = ClassModel.model_Poisson()
        self.prior = ClassPrD.Gam(a, b)
    
    def train(self, trainingData):
        a_hat = trainingData.sum() + self.a
        b_hat = len(trainingData) + self.b
        return (ClassPrD.Gam(a_hat, b_hat), ClassPredictive.Predictive_NB(a_hat, 1 / (1 + b_hat)))


