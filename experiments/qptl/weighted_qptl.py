import sys
sys.path.insert(0,'../../QPTL/')
sys.path.insert(0,"../..")

import time,datetime
import pandas as pd
import logging

from melding_knapsack import *
from get_energy import get_energy
import time,datetime
import logging

formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='weighted_QPTL.log', level=logging.INFO,format=formatter)

filename = "../Results/weighted_QPTL.csv"

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

H_combinations={30: [{'optimizer':optim.Adam,'tau':1000},{'optimizer':optim.Adam,'tau':30000}],
60: [{'optimizer':optim.Adam,'tau':1000},{'optimizer':optim.Adam,'tau':30000}],
90: [{'optimizer':optim.Adam,'lr':1e-4,'tau':1000},{'optimizer':optim.Adam,'tau':30000}],
120: [{'optimizer':optim.Adam,'lr':1e-4,'tau':1000},{'optimizer':optim.Adam,'tau':30000}],
150: [{'optimizer':optim.Adam,'lr':1e-4,'tau':1000},{'optimizer':optim.Adam,'tau':30000}],
180:[{'lr':1e-4,'tau':1000},{'optimizer':optim.Adam,'tau':30000}],
210:[{'lr':1e-4,'tau':1000},{'optimizer':optim.Adam,'tau':30000}]}

for r in range(10): 
    for capa in range(30,222,30):
        h_list = H_combinations[capa]
        for h in h_list:
            print("N : %s capacity:%d Time:%s \n" %(str(h),capa,datetime.datetime.now()))
            clf = qptl(capa,weights,epochs=20,validation=True,verbose=True,**h)
            start = time.time()
            pdf = clf.fit(X_1gtrain,y_train,X_1gvalidation,y_validation,X_1gtest,y_test)
            end = time.time()
            pdf['capacity'] = capa
            pdf['hyperparams'] = [h for x in range(pdf.shape[0])]
            pdf['total_time'] = end-start
            with open(filename, 'a') as f:
                pdf.to_csv(f, mode='a', header=f.tell()==0,index=False)        
            del pdf
