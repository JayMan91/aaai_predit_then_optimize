import sys
sys.path.insert(0,'../..')
sys.path.insert(0,'../../QPTL/')
import pandas as pd
from get_energy import get_energy
import time,datetime
from melding_knapsack import *
from SPO_dp_lr import *
from sgd_learner import grid_search
from get_energy import *
import itertools
import time,datetime
import logging
import os
from torch import nn, optim
formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='weighted_QPTL.log', level=logging.INFO,format=formatter)


data = np.load('../../Data.npz')
X_1gtrain = data['X_1gtrain']
X_1gtest = data['X_1gtest']
y_train = data['y_train']
y_test = data['y_test']
X_1gvalidation = X_1gtest[0:2880,:]
y_validation = y_test[0:2880]
y_test= y_test[2880:]
X_1gtest = X_1gtest[2880:,:]
weights = [data['weights'].tolist()]
weights = np.array(weights)

##Weighted
clf = Pytorch_regression( weights=weights, epochs=35, optimizer= optim.SGD, lr=1e-2,momentum=0.1, capacity =[90],
store_result =True,verbose=True)
pdf = clf.fit(X_1gtrain,y_train,X_1gvalidation,y_validation,X_1gtest,y_test)
print(pdf.head())
clf = SGD_SPO_dp_lr( weights=weights, epochs=35, optimizer= optim.Adam, capacity =[90],store_result =True,verbose=True)
pdf = clf.fit(X_1gtrain,y_train,X_1gvalidation,y_validation,X_1gtest,y_test)

clf = qptl(90,weights,epochs=20,optimizer= optim.Adam,lr =0.0001, store_result =True,verbose=True)
pdf = clf.fit(X_1gtrain,y_train,X_1gvalidation,y_validation,X_1gtest,y_test)

## Unweighted
(X_1gtrain, y_train, X_1gtest, y_test) = get_energy("../../prices2013.dat")
X_1gvalidation = X_1gtest[0:2880,:]
y_validation = y_test[0:2880]
y_test= y_test[2880:]
X_1gtest = X_1gtest[2880:,:]
weights = [[1 for i in range(48)]]        
weights = np.array(weights)

clf = Pytorch_regression(weights=weights, epochs=35, optimizer= optim.SGD, lr=1e-1, capacity = [15],store_result =True,verbose=True)
pdf = clf.fit(X_1gtrain,y_train,X_1gvalidation,y_validation,X_1gtest,y_test)
clf = SGD_SPO_dp_lr( weights=weights, epochs=35, optimizer= optim.Adam, capacity =[15], store_result =True,verbose=True)
pdf = clf.fit(X_1gtrain,y_train,X_1gvalidation,y_validation,X_1gtest,y_test)

clf = qptl(15,weights,epochs=20,optimizer= optim.Adam,store_result =True,verbose=True )
pdf = clf.fit(X_1gtrain,y_train,X_1gvalidation,y_validation,X_1gtest,y_test)
