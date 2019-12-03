

import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as objplt
class KmALgo:
    def __init__(self,X,K):
        self.x=X
        self.Output={}
        self.center=np.array([]).reshape(self.x.shape[1],0)
        self.K=K
        self.rowsIn=self.x.shape[0]
        self.n=X.shape[1]
        self.loss = 0
        
    
    def fit(self,reps):
   
        vv=self.x[np.random.choice(self.rowsIn, self.K, replace=False)]
        self.center = vv.T
   
        
     
        for n in range(reps):
            ed=np.array([]).reshape(self.rowsIn,0)
            for k in range(self.K):
                dist=np.sum((self.x-self.center[:,k])**2,axis=1)
                ed=np.c_[ed,dist]
            C=np.argmin(ed,axis=1)+1
            for i in range(len(ed)):
              self.loss += ed[i][C[i]-1]
            #adjust the centroids
            Y={}
            for k in range(self.K):
                Y[k+1]=np.array([]).reshape(self.n,0)
            for i in range(self.rowsIn):
                Y[C[i]]=np.c_[Y[C[i]],self.x[i]]
        
            for k in range(self.K):
                Y[k+1]=Y[k+1].T
            for k in range(self.K):
                self.center[:,k]=np.mean(Y[k+1],axis=0)
                
            self.Output=Y
            
    
    def predict(self):
        return self.Output,self.loss
    
data = pd.read_csv("CSE575-HW03-Data.csv")
data = np.array(data)
km = KmALgo(data,2)
km.fit(50)
opt,loss = km.predict()

cl1 = opt[1]
cl1 = cl1[:,0:2]
cl2 = opt[2]
cl2 = cl2[:,0:2]
plt.scatter(cl1[:,0],cl1[:,1])
plt.scatter(cl2[:,0],cl2[:,1])


plt.show()

errors = []
K = [2,3,4,5,6,7,8,9]
for k in K:
  km = KmALgo(data,k)
  km.fit(50)
  opt,loss = km.predict()
  errors.append(loss)
objplt.plot(K, errors, 'bx-')
objplt.xlabel('k')
objplt.ylabel('Objective function')
objplt.show()
