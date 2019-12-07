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
logging.basicConfig(filename='Unweighted_QPTL.log', level=logging.INFO,format=formatter)

filename = "../Results/Unweighted_QPTL.csv"

(X_1gtrain, y_train, X_1gtest, y_test) = get_energy("../../prices2013.dat")
X_1gvalidation = X_1gtest[0:2880,:]
y_validation = y_test[0:2880]

y_test= y_test[2880:]
X_1gtest = X_1gtest[2880:,:]

weights = [[1 for i in range(48)]]        
weights = np.array(weights)
H_combinations={5: [{'lr':1e-3,'optimizer':optim.Adam}],
10: [{'optimizer':optim.Adam,'lr':1e-3,'tau':20000}],
15: [{'optimizer':optim.Adam,'lr':1e-4,'tau':500}],
20: [{'optimizer':optim.Adam,'lr':1e-4,'tau':75000}],
25: [{'optimizer':optim.Adam,'tau':1000,'lr':1e-4}],
30:[{'optimizer':optim.Adam,'lr':1e-3,'tau':100000}],
35:[{'optimizer':optim.Adam,'lr':1e-5,'tau':100000}],
40:[{'optimizer':optim.Adam,'lr':1e-3,'tau':75000}],
45:[{'optimizer':optim.Adam,'lr':1e-4,'tau':500}]
 }

for r in range(10): 
    for capa in range(5,46,5):
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
