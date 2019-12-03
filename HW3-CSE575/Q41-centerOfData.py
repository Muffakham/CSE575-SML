

import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt

class altKmeans:
    def __init__(self,X,K):
        self.X=X
        self.inpSize = X.shape
        self.Output={}
        self.center=np.array([[83.40, 3.5313,-4.1257,-4.2176,-7.00,-2.50,-2.2587,-2.2301,-5.3913,3.6305,5.6735,3.7018,0.3256]])
        self.center = self.center.T
        self.K=K
        self.m=self.X.shape[0]
        self.l,self.n=self.inpSize


   
    def fit(self,n_iter):
        
        for n in range(n_iter):
            ed=np.array([]).reshape(self.m,0)
            for k in range(self.K):
                t=np.sum((np.abs(self.X-self.center[:,k])),axis=1)
                ed=np.c_[ed,t]
            C=np.argmin(ed,axis=1)+1
            #adjust the centroids
            Y={}
            for k in range(self.K):
                Y[k+1]=np.array([]).reshape(self.n,0)
            for i in range(self.m):
                Y[C[i]]=np.c_[Y[C[i]],self.X[i]]
        
            for k in range(self.K):
                Y[k+1]=Y[k+1].T
            for k in range(self.K):
            	
            	ff = self.center
            	self.center[:,k]=self.calCenter(Y[k+1],k+1)
            	if(np.array_equal(ff,self.center)):
            		break 
                
                
            self.Output=Y
            
    
    def predict(self):
        return self.Output

    def calCenter(self,a,k):
    	n = a.shape[0]
    	ot = {}
    	for i in range(0,n):
    		temp = a[i]
    		ot[i] = np.sum((np.abs(temp-a[i,:])))
    	f = np.argmin(ot)
    	return a[f]


    
data = pd.read_csv("CSE575-HW03-Data.csv")
data = np.array(data)
km = altKmeans(data,1)
ft = km.fit(500)
op = km.predict()

class_1 = op[1]
class_1 = class_1[:,0:2]
plt.scatter(class_1[:,0],class_1[:,1])


print("Center of Cluster 1 is ",km.center[:,0])


plt.show()
