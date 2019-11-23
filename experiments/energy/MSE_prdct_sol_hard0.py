import sys
sys.path.insert(0,'../../EnergyCost/')
sys.path.insert(0,"../..")
import math
from torch_SPO_updated import *
from ICON import *
import time,datetime
import pandas as pd
import logging
from get_energy import get_energy
formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='MSEpredictor_solutions.log', level=logging.INFO,format=formatter)
logging.info('Started\n')


modelPATH_list =['../Results/MSE_pred/MSE-prediction_test_epoch2.npy',
'../Results/MSE_pred/MSE-prediction_test_epoch4.npy',
'../Results/MSE_pred/MSE-prediction_test_epoch6.npy',
'../Results/MSE_pred/MSE-prediction_test_epoch8.npy',
'../Results/MSE_pred/MSE-prediction_test_epoch10.npy']#,
#'../Results/MSE_pred/MSE-prediction_test_epoch20.npy']


(X_1gtrain, y_train, X_1gtest, y_test) = get_energy("../../prices2013.dat")
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

i = 0
file = "../../EnergyCost/Hard_Instances/instance0"+str(i)+"/instance.txt"
filename = "../Results/Hard0"+str(i)+"/Hard"+str(i)+"_MSE_solutions.csv"
param_data = data_reading(file)
for modelPATH in modelPATH_list:
		print("Load :%d model %s Time:%s \n" %(i,modelPATH, datetime.datetime.now()))
		param_data = data_reading(file)
		y_pred = np.load(modelPATH)

		y_pred = np.repeat(y_pred,6)
		solutions = ICON_solution(y_pred,y_test,reset= True,presolve=True,
			relax=True,n_items=288,solver= Gurobi_ICON,method=2,**param_data)
		df = pd.DataFrame.from_dict(solutions)
		df['model'] = modelPATH
		with open(filename, 'a') as f:
				df.to_csv(f, mode='a', header=f.tell()==0,index=False)


