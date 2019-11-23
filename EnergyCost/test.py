from torch_SPO_updated import *
from ICON import *
import time,datetime
import pandas as pd
import logging

import sys
#sys.path.insert(0,'../../EnergyCost/')
sys.path.insert(0,"../")
from get_energy import get_energy

(X_1gtrain, y_train, X_1gtest, y_test) = get_energy("../prices2013.dat")
X_1gvalidation = X_1gtest[0:2880,:]
y_validation = y_test[0:2880]

y_test= y_test[2880:]
X_1gtest = X_1gtest[2880:,:]

file = "load1/day01.txt"
#filename = "../Results/Load1_SPO_warmstart.csv"
param_data = data_reading(file)
h= {"lr":1e-5,"momentum":0.01}
w= {'reset':False,'presolve':False,'warmstart':False}


clf = SGD_SPO_generic(solver= Gurobi_ICON,accuracy_measure=False,relax=True, 
                      validation_relax= True,verbose=True,param= param_data, maximize= False,epochs= 10,
                      timelimit=1000,**w,**h)

pdf = clf.fit(X_1gtrain,y_train,X_1gvalidation,y_validation,X_1gtest,y_test)