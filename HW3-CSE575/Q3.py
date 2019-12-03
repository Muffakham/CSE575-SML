
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
np.random.seed(42)

class GMM:
    def __init__(self, k, reps=5):
        self.k = k
        self.reps = int(reps)

    def assign(self, X):
        self.shape = X.shape
        self.row, self.m = self.shape

        randomRow = np.random.randint(low=0, high=self.row, size=self.k) # generating a random row value 
        self.mean = [  X[l,:] for l in randomRow ]
        self.stdDev = [ np.cov(X.T) for _ in range(self.k) ]

        self.prob = np.full(shape=self.k, fill_value=1/self.k)
        self.weights = np.full( shape=self.shape, fill_value=1/self.k)
        
    def maximisationStep(self, X):
    
        for i in range(self.k):
            weight = self.weights[:, [i]]
            total_weight = weight.sum()
            self.mean[i] = (X * weight).sum(axis=0) / total_weight
            self.stdDev[i] = np.cov(X.T, aweights=(weight/total_weight).flatten(), bias=True)    

    def expectationStep(self, X):
        
        self.weights = self.calcProbability(X)
        self.prob = self.weights.mean(axis=0) # taking the avergae of the weights as the new value
    


    def fit(self, X):
        self.assign(X)
        
        for _ in range(self.reps):
            self.expectationStep(X)
            self.maximisationStep(X)
            
    def calcProbability(self, X):
        occurence = np.zeros( (self.row, self.k) )
        for i in range(self.k):
            normalDistr = multivariate_normal(
                mean=self.mean[i], 
                cov=self.stdDev[i])
            occurence[:,i] = normalDistr.pdf(X)
        
        n = occurence * self.prob
        d = n.sum(axis=1)[:, np.newaxis]
        j = n / d
        return j
    
    def predict(self, X):
        weights = self.calcProbability(X)
        return np.argmax(weights, axis=1)

data = pd.read_csv('CSE575-HW03-Data.csv')
inp = np.array(data)

gmm = GMM(k=2, reps=10)
gmm.fit(inp)
opt=gmm.predict(inp)



cl1 = [] 
cl2 = []

'''
classifying the input into two guassian mixtures obtained
'''

for i in range(len(inp)):
    if  opt[i] == 1:
        cl2.append(inp[i])
    else:
        cl1.append(inp[i])
cl1 = np.array(cl1)
cl2 = np.array(cl2)

'''
plotting the data 
'''

plt.scatter(cl1[:,1],cl1[:,2])
plt.scatter(cl2[:,1],cl2[:,2])

plt.show()
