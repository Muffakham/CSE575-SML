'''
ATTENTION ! MNIST needs to be installed inorder to run this code
### pip intsall mnist
'''



import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import mnist
import random
import math

random.seed(2)
np.random.seed(2)

import mnist
from matplotlib import pyplot as plt
import numpy as np
import random
import numpy.matlib
from collections import Counter
random.seed(42)
np.random.seed(42)


train_images = mnist.train_images()
train_images = train_images.reshape((train_images.shape[0], train_images.shape[1] * train_images.shape[2]))
train_labels = mnist.train_labels()
train_labels = train_labels.reshape(-1,1)

test_images = mnist.test_images()
test_images = test_images.reshape((test_images.shape[0], test_images.shape[1] * test_images.shape[2]))
test_labels = mnist.test_labels()
test_labels = test_labels.reshape(-1,1)

x_train = np.asarray(train_images).astype(np.float32)
y_train = np.asarray(train_labels).astype(np.int32)
x_test = np.asarray(test_images).astype(np.float32)
y_test = np.asarray(test_labels).astype(np.int32)


x_train = np.asarray(train_images).astype(np.float32)
y_train = np.asarray(train_labels).astype(np.int32)
x_test = np.asarray(test_images).astype(np.float32)
y_test = np.asarray(test_labels).astype(np.int32)


class knn():

    def __init__(self):
        pass

    def train(self, X, y):
        self.x_train = X
        self.y_train = y
    


    def distance(self, X):

        ab = np.dot(X, self.x_train.T)
        a2 = np.square(X).sum(axis = 1)
        b2 = np.square(self.x_train).sum(axis = 1)
        diff = np.sqrt(-2 * ab + b2 + np.matrix(a2).T) #fromula for (-2ab + a^2 + b^2) = (a-b)^2
        print(diff.shape)
        return(diff)

    def predict(self, X, k=1):
        diff = self.distance(X)
        
        inpShape = diff.shape[0]
        y_pred = np.zeros(inpShape)

        for i in range(inpShape):
            Klabels = []
            labels = self.y_train[np.argsort(diff[i,:])].flatten()
            Klabels = labels[:k]
            c = Counter(Klabels)
            y_pred[i] = c.most_common(1)[0][0]

        return(y_pred)



Kv = [1, 3, 5, 10, 20, 30, 40, 50, 60]
scores = []
for k in Kv:
  batch = 2000
  cl = knn()
  cl.train(x_train, y_train)
  p = []
  for i in range(int(len(x_test)/(batch))):
      predts = cl.predict(x_test[i * batch:(i+1) * batch], k)
      p = p + list(predts)
  
  count=0
  for i in range(len(p)):
    if y_test[i]==p[i]:
      count+=1
      
  scores.append((count/len(p)))
  print(count/len(p))







plt.plot(Kv, scores)
plt.xlabel('Values of k')
plt.ylabel('Prediction Accuracy')
plt.show()