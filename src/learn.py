import ClassBayesianML
import numpy as np
import matplotlib
from scipy import stats
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import math

# make N bernoulli sample with paramater p
def make_bernoulli_sample(p, N, randomState):
    return stats.bernoulli.rvs(p=p, size=N, random_state=randomState)

# make N Categorical sample with paramater pi
def make_categorical_sample(pi, N, randomState):
    return stats.multinomial.rvs(n=1,p=pi, size=N, random_state=randomState)

def make_poisson_sample(lam, N, randomState):
    return stats.poisson.rvs(lam, size=N, random_state=randomState)

# learn max N trainingSamples with CPclass
def learn(CPclass, trainingSamples, max_N):
    trainingData = trainingSamples[0:max_N]
    return CPclass.train(trainingData)

def calLoss(predictive, trainingSamples, train_N, testData):
    N = min(len(trainingSamples), train_N)
    tn = -1
    gm = -1
    if N != 0:
        tn = 0
        for x in trainingSamples[0:N]:
            tn = tn + math.log(predictive(x))
        tn = -tn/N
    if len(testData) != 0:
        gm = 0
        for x in testData:
            gm = gm + math.log(predictive(x))
        gm = -gm / len(testData)

    return (tn, gm)

def test():
    trainingRandomState = 1
    testRandomState = 2
    trainingSamples = make_bernoulli_sample(0.25, 200, trainingRandomState)
    testData = make_bernoulli_sample(0.25, 100, testRandomState)
    a = 10
    b = 1
    berBeta = ClassBayesianML.Ber_Beta(a,b)
    train_N = 200
    tapleOf = learn(berBeta, trainingSamples, train_N)
    tapleOf[0].plot('N=' + str(train_N))
    loss = calLoss(tapleOf[1].predict, trainingSamples, train_N, testData)
    print('predictive theta = ' + str(tapleOf[1].theta))
    print('Tn='+str(loss[0]))
    print('Gm='+str(loss[1]))

def testCatDir2d():
    trainingRandomState = 1
    testRandomState = 2
    trainingSamples = make_categorical_sample(np.array([0.5,0.5]), 200, trainingRandomState)
    testData = make_categorical_sample(np.array([0.5,0.5]), 100, testRandomState)
    alpha = np.array([1,1])
    catDir = ClassBayesianML.Cat_Dir(alpha)
    train_N = 200
    tapleOf = learn(catDir, trainingSamples, train_N)
    tapleOf[0].plot('N=' + str(train_N))
    loss = calLoss(tapleOf[1].predict, trainingSamples, train_N, testData)
    print('predictive pi = ' + str(tapleOf[1].pi))
    print('Tn='+str(loss[0]))
    print('Gm='+str(loss[1]))

def testCatDir3d():
    trainingRandomState = 1
    testRandomState = 2
    trainingSamples = make_categorical_sample(np.array([0.2,0.3,0.5]), 200, trainingRandomState)
    testData = make_categorical_sample(np.array([0.2,0.3,0.5]), 100, testRandomState)
    alpha = np.array([1,1,1])
    catDir = ClassBayesianML.Cat_Dir(alpha)
    train_N = 200
    tapleOf = learn(catDir, trainingSamples, train_N)
    tapleOf[0].plot('N=' + str(train_N))
    loss = calLoss(tapleOf[1].predict, trainingSamples, train_N, testData)
    print('predictive pi = ' + str(tapleOf[1].pi))
    print('Tn='+str(loss[0]))
    print('Gm='+str(loss[1]))

def testPoiGam():
    trainingRandomState = 1
    testRandomState = 2
    trainingSamples = make_poisson_sample(2, 200, trainingRandomState)
    testData = make_poisson_sample(2, 100, testRandomState)
    a = 0.5
    b = 0.5
    poiGam = ClassBayesianML.Poi_Gam(a, b)
    train_N = 10
    tapleOf = learn(poiGam, trainingSamples, train_N)
    tapleOf[0].plot('N=' + str(train_N))
    loss = calLoss(tapleOf[1].predict, trainingSamples, train_N, testData)
    print('Tn='+str(loss[0]))
    print('Gm='+str(loss[1]))
    
