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
logging.basicConfig(filename='QPTL_Hard0.log', level=logging.INFO,format=formatter)
logging.info('Started\n')



file = "../../EnergyCost/Hard_Instances/instance00/instance.txt"
modelPATH = str('../Results/Hard00/QPTL_hard00')
param_data = data_reading(file)
(X_1gtrain, y_train, X_1gtest, y_test) = get_energy()
X_1gvalidation = X_1gtest[0:2880,:]
y_validation = y_test[0:2880]

y_test= y_test[2880:]
X_1gtest = X_1gtest[2880:,:]
n_iter=2
H_combinations=[
{'optimizer':optim.Adam,'lr':1e-2,'tau':1e+5} ]
for i in range(n_iter):
	for h in H_combinations:
		model_name = str(modelPATH+"_"+str(i))
		print("hyperparams : %s  Time:%s \n" %(str(h),datetime.datetime.now()))
		start =  time.time()
		clf  =  qptl_ICON(epochs=10,param= param_data,model_save=True,model_name= model_name, **h )
		clf.fit(X_1gtrain,y_train)
		end= time.time()

