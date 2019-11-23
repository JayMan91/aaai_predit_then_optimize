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
formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='unweighted_intopt.log', level=logging.INFO,format=formatter)

(X_1gtrain, y_train, X_1gtest, y_test) = get_energy("../../prices2013.dat")
X_1gvalidation = X_1gtest[0:2880,:]
y_validation = y_test[0:2880]

y_test= y_test[2880:]
X_1gtest = X_1gtest[2880:,:]
weights = [[1 for i in range(48)]]        
weights = np.array(weights)
filename = "../Results/Interior/Unweighted_knapsack.csv"

H_combinations={5: [{'optimizer':optim.Adam,'lr':1e-2}],
        10: [{'optimizer':optim.Adam,'lr':1e-2}],
        15: [{'optimizer':optim.Adam,'lr':1e-2}],
        20: [{'optimizer':optim.Adam,'lr':1e-2}],
        25: [{'optimizer':optim.Adam,'lr':1e-2}],
        30:[{'optimizer':optim.Adam,'lr':1e-2}],
        35:[{'optimizer':optim.Adam,'lr':1e-2}],
        40:[{'optimizer':optim.Adam,'lr':1e-2}],
        45:[{'optimizer':optim.Adam,'lr':1e-2}]
 }

for r in range(10): 
    for capa in range(5,46,5):
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
