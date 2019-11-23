import sys
sys.path.insert(0,'../../EnergyCost/')
sys.path.insert(0,"../..")
from torch_SPO_updated import *
from ICON import *
import time,datetime
import pandas as pd
import logging
from get_energy import get_energy
formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='SPO_Load6.log', level=logging.INFO,format=formatter)
logging.info('Started\n')

file = "../../EnergyCost/load6/day01.txt"
filename = "../Results/Load6_SPO_warmstart_corrected.csv"
param_data = data_reading(file)
(X_1gtrain, y_train, X_1gtest, y_test) = get_energy("../../prices2013.dat")
X_1gvalidation = X_1gtest[0:2880,:]
y_validation = y_test[0:2880]

y_test= y_test[2880:]
X_1gtest = X_1gtest[2880:,:]
n_iter=10

H_list= [{'lr':1e-5,'momentum':0.1},{'lr':1e-5,'momentum':0.01}]
warmstart_hyperparams = [{'reset':True,'presolve':False,'warmstart':False},
{'reset':True,'presolve':True,'warmstart':False},
{'reset':False,'presolve':True,'warmstart':False},
{'reset':False,'presolve':True,'warmstart':True}]

for w in warmstart_hyperparams:
	for h in H_list:
		for i in range(n_iter): 
			print("N : %d Time:%s %s %s\n" %(i,datetime.datetime.now(),h,w))
			clf = SGD_SPO_generic(solver= Gurobi_ICON,accuracy_measure=False,relax=True, 
			validation_relax= True,verbose=True,param= param_data, maximize= False,
	                         epochs= 4,timelimit=21600,**h,**w )
			start =  time.time()
			pdf = clf.fit(X_1gtrain,y_train,X_1gvalidation,y_validation,X_1gtest,y_test)
			end = time.time()
			pdf['training_relaxation'] = True
			pdf['validation_relaxation'] = True
			pdf ['reset'] = w['reset']
			pdf['presolve'] = w['presolve']
			pdf['warmstart'] = w['warmstart']
			pdf['hyperparams'] = [h for x in range(pdf.shape[0])]
			with open(filename, 'a') as f:
				pdf.to_csv(f, mode='a', header=f.tell()==0,index=False)
