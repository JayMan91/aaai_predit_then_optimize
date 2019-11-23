import sys
sys.path.insert(0,'../../QPTL_ICON/')
sys.path.insert(0,"../..")

import time,datetime
import pandas as pd
import logging

from qptl_model import *
from get_energy import get_energy
import time,datetime
import logging

formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='QPTL_Load6.log', level=logging.INFO,format=formatter)
logging.info('Started\n')



file = "../../EnergyCost/load6/day01.txt"
filename= "../Results/Load6_qptl.csv"
param_data = data_reading(file)
(X_1gtrain, y_train, X_1gtest, y_test) = get_energy("../../prices2013.dat")
X_1gvalidation = X_1gtest[0:2880,:]
y_validation = y_test[0:2880]

y_test= y_test[2880:]
X_1gtest = X_1gtest[2880:,:]
n_iter= 10
H_combinations=[ {'optimizer':optim.Adam,'lr':1e-2,'betas':(0.9, 0.6)}]
for i in range(n_iter):
	for h in H_combinations:
		print("hyperparams : %s  Time:%s \n" %(str(h),datetime.datetime.now()))
		start =  time.time()
		clf  =  qptl_ICON(epochs=12,param= param_data,verbose=True,validation=True,validation_relax=True,  **h )
		pdf = clf.fit(X_1gtrain,y_train,X_1gvalidation,y_validation ,X_1gtest,y_test)
		end= time.time()
		pdf['hyperparams'] = [h for x in range(pdf.shape[0])]
		
		pdf['validation_relax'] = True
		with open(filename, 'a') as f:
			pdf.to_csv(f, mode='a', header=f.tell()==0,index=False)

