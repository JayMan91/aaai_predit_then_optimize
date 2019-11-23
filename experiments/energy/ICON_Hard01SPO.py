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
logging.basicConfig(filename='SPO_hard01.log', level=logging.INFO,format=formatter)
logging.info('Started\n')

file = "../../EnergyCost/Hard_Instances/instance01/instance.txt"
filename = "../Results/Hard01_SPO_timelimit.csv"
modelPATH = str('../Results/Hard01/SPO_hard01')
param_data = data_reading(file)
(X_1gtrain, y_train, X_1gtest, y_test) = get_energy()
X_1gvalidation = X_1gtest[0:2880,:]
y_validation = y_test[0:2880]

y_test= y_test[2880:]
X_1gtest = X_1gtest[2880:,:]

X_1gtest =  np.repeat(X_1gtest, 6,axis=0)
X_1gvalidation = np.repeat(X_1gvalidation,6,axis=0)
X_1gtrain = np.repeat(X_1gtrain,6,axis=0)
y_test = np.repeat(y_test,6)
y_train = np.repeat(y_train,6)
y_validation = np.repeat(y_validation,6)

h= {'lr':1e-5}
warmstart_hyperparams = {
'presolve':{'reset':True,'presolve':True,'warmstart':False},
'prestart':{'reset':False,'presolve':True,'warmstart':False},
'prebest':{'reset':False,'presolve':True,'warmstart':True}}
n_iter = 1
for k,w in warmstart_hyperparams.items(): 
	for i in range(n_iter): 
		model_name = str(modelPATH+"."+str(k)+"_"+str(i))

		clf = SGD_SPO_generic(solver= Gurobi_ICON,accuracy_measure=False,relax=True, 
		validation_relax= True,param= param_data, maximize= False,n_items=288,model_save=True,model_name= model_name,
	                         epochs= 10,timelimit=43200,lr=1e-5,**w )
		start =  time.time()
		clf.fit(X_1gtrain,y_train)
		end = time.time()
