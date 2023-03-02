import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets



def add_ones(X):

  # ADD YOUR CODES
  a=np.ones((X.shape[0],1))
  b=X[:,:]
  X_new=np.hstack((a,b))
  return X_new

def split_data(X,Y, train_size):
  # ADD YOUR CODES
  # shuffle the data before splitting it
  Y=Y.reshape((Y.shape[0],1))
  #print(Y.shape)
  data=np.hstack((X,Y))
  
  data1=np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data)

  size=int(train_size*X.shape[0])
  #print(np.shape(data1))
  X_train=data1[:size,:-1]
  X_test=data1[size:,:-1]
  Y_train=data1[:size,-1]
  Y_test=data1[size:, -1]

  return X_train, X_test, Y_train.reshape(Y_train.shape[0],1), Y_test.reshape(Y_test.shape[0],1)