import sys
sys.path.insert(0,'..')
sys.path.insert(0,"../EnergyCost")
from torch_SPO_updated import *
from ICON import *
import time,datetime
import pandas as pd
import logging

formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='SPO_Load2.log', level=logging.INFO,format=formatter)
logging.info('Started\n')

file = "load2/day01.txt"
filename = "Load2_SPO_solutionobjcut.csv"
param_data = data_reading(file)
(X_1gtrain, y_train, X_1gtest, y_test) = get_energy()
X_1gvalidation = X_1gtest[0:2880,:]
y_validation = y_test[0:2880]

y_test= y_test[2880:]
X_1gtest = X_1gtest[2880:,:]
n_iter=10

h= {'lr':1e-5}
rslt= []
for i in range(5):
	print("N : %d Time:%s \n" %(i,datetime.datetime.now()))
	clf = SGD_SPO_generic(solver= Gurobi_ICON,accuracy_measure=False,relax=True, 
		validation_relax= True,verbose=True,param= param_data, maximize= False,
	                         epochs= 10,timelimit=21600,obj_cut=0, **h )
	logging.info('SPO')
	start =  time.time()
	pdf = clf.fit(X_1gtrain,y_train,X_1gvalidation,y_validation,X_1gtest,y_test)
	end = time.time()
	pdf['training_relaxation'] = True
	pdf['validation_relaxation'] = True
	pdf['hyperparams'] = [h for x in range(pdf.shape[0])]
	pdf['total_time'] = end- start
	pdf['obj_cut'] = 0
	with open(filename, 'a') as f:
		pdf.to_csv(f, mode='a', header=f.tell()==0,index=False)

	print("N : %d Time:%s \n" %(i,datetime.datetime.now()))
	clf = SGD_SPO_generic(solver= Gurobi_ICON,accuracy_measure=False,relax=True, 
		validation_relax= True,verbose=True,param= param_data, maximize= False,
	                         epochs= 10,timelimit=21600,obj_cut=10, **h )
	logging.info('SPO')
	start =  time.time()
	pdf = clf.fit(X_1gtrain,y_train,X_1gvalidation,y_validation,X_1gtest,y_test)
	end = time.time()
	pdf['training_relaxation'] = True
	pdf['validation_relaxation'] = True
	pdf['hyperparams'] = [h for x in range(pdf.shape[0])]
	pdf['total_time'] = end- start
	pdf['obj_cut'] = 10
	with open(filename, 'a') as f:
		pdf.to_csv(f, mode='a', header=f.tell()==0,index=False)