import sys
sys.path.insert(0,'../../EnergyCost/')
sys.path.insert(0,"../..")
sys.path.insert(0,"../../QPTL_ICON")
from torch_SPO_updated import *
from ICON import *
import time,datetime
import pandas as pd
import logging
import pandas as pd
from qptl_model import *
from get_energy import get_energy

formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='QPTL_Load2.log', level=logging.INFO,format=formatter)
logging.info('Started\n')


file = "../../EnergyCost/load2/day01.txt"
filename = "../Results/Hard00_SPO_timelimit.csv"

param_data = data_reading(file)
(X_1gtrain, y_train, X_1gtest, y_test) = get_energy()
X_1gvalidation = X_1gtest[0:2880,:]
y_validation = y_test[0:2880]

y_test= y_test[2880:]
X_1gtest = X_1gtest[2880:,:]
n_iter=10
H_combinations=[{'lr':1e-2},{'optimizer':optim.Adam,'lr':1e-2} ]
for i in range(n_iter):
	for h in H_combinations:
		print("hyperparams : %s  Time:%s \n" %(str(h),datetime.datetime.now()))
		start =  time.time()
		clf  =  qptl_ICON(epochs=1,param= param_data,verbose=True,validation=True,validation_relax=True,  **h )
		pdf = clf.fit(X_1gtrain,y_train,X_1gvalidation,y_validation,  X_1gtest,y_test)
		end= time.time()
		pdf['hyperparams'] = [h for x in range(pdf.shape[0])]
		pdf['total_time'] = end-start
		pdf['validation_relax'] = True
		with open(filename, 'a') as f:
			pdf.to_csv(f, mode='a', header=f.tell()==0,index=False)
