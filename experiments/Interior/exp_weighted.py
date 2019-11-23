import sys
sys.path.insert(0,'../..')
sys.path.insert(0,"../../Interior")
from intopt import *
from KnapsackSolving import *
from get_energy import *
import itertools
import scipy as sp
import numpy as np
import time,datetime
import pandas as pd
import logging

from get_energy import get_energy
import time,datetime
import logging

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
filename = "../Results/Interior/weighted_knapsack.csv"

H_combinations={30: [{'optimizer':optim.Adam,'lr':1e-2}],
        60: [{'optimizer':optim.Adam,'lr':1e-2}],
        90: [{'optimizer':optim.Adam,'lr':1e-2}],
        120: [{'optimizer':optim.Adam,'lr':1e-2}],
        150: [{'optimizer':optim.Adam,'lr':1e-2}],
        180:[{'optimizer':optim.Adam,'lr':1e-2}],
        210:[{'optimizer':optim.Adam,'lr':1e-2}]}

for r in range(10): 
    for capa in range(30,236,30 ):
        h_list = H_combinations[capa]
        for h in h_list:
            print("N : %s capacity:%d Time:%s \n" %(str(h),capa,datetime.datetime.now()))
            clf = pred_opt(np.array([capa]),weights,epochs= 12,verbose=True,**h)
            start = time.time()
            pdf = clf.fit(X_1gtrain,y_train,X_1gtest,y_test,X_1gvalidation,y_validation)
            end = time.time()
            pdf['capacity'] = capa
            pdf['hyperparams'] = [h for x in range(pdf.shape[0])]
            pdf['total_time'] = end-start
            with open(filename, 'a') as f:
                pdf.to_csv(f, mode='a', header=f.tell()==0,index=False)        
            del pdf
