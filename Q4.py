# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 22:11:53 2019

@author: admin
"""

import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt
class Kmeans:
    def __init__(self,X,K):
        self.X=X
        self.shp = X.shape
        self.Output={}
        self.Centroids=np.array([[82.00, 3.9313,-4.6257,-5.2176,-7.00,-2.00,-2.2,-2.23,-4.3913,1.6305,2.6735,2.7018,0.3256],[88.77,-8.00,0.3902,-0.8573,-8.8,-3.1611,-2.5,0.2141,2.6793,6.5664,8.0879,4.8908,2.2093]])
        self.Centroids = self.Centroids.T
        self.K=K
        self.m=self.X.shape[0]
        self.l,self.n=self.shp


   
    def fit(self,n_iter):
        #randomly Initialize the centroids
        #self.Centroids=self.kmeanspp(self.X,self.K)
        
        """for i in range(self.K):
            rand=rd.randint(0,self.m-1)
            self.Centroids=np.c_[self.Centroids,self.X[rand]]"""
        
        #compute euclidian distances and assign clusters
        for n in range(n_iter):
            EuclidianDistance=np.array([]).reshape(self.m,0)
            for k in range(self.K):
                tempDist=np.sum((np.abs(self.X-self.Centroids[:,k])),axis=1)
                EuclidianDistance=np.c_[EuclidianDistance,tempDist]
            C=np.argmin(EuclidianDistance,axis=1)+1
            #adjust the centroids
            Y={}
            for k in range(self.K):
                Y[k+1]=np.array([]).reshape(self.n,0)
            for i in range(self.m):
                Y[C[i]]=np.c_[Y[C[i]],self.X[i]]
        
            for k in range(self.K):
                Y[k+1]=Y[k+1].T
            for k in range(self.K):
            	print("y is ",Y[k+1])
            	self.Centroids[:,k]=self.calCenter(Y[k+1],k+1) 
                
                
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
km = Kmeans(data,2)
ft = km.fit(50)
op = km.predict()

class_1 = op[1]
class_1 = class_1[:,0:2]
class_2 = op[2]
class_2 = class_2[:,0:2]
plt.scatter(class_1[:,0],class_1[:,1])
plt.scatter(class_2[:,0],class_2[:,1])

plt.show()