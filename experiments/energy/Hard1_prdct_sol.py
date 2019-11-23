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
logging.basicConfig(filename='predictor_solutions.log', level=logging.INFO,format=formatter)
logging.info('Started\n')

file = "../../EnergyCost/Hard_Instances/instance01/instance.txt"
filename = "../Results/Hard01/Hard01_pred_solutions_final.csv"
modelPATH_list =['../Results/Hard01/SPO_hard01.presolve_0_Epoch0_80.pth',
'../Results/Hard01/SPO_hard01.presolve_0_Epoch0_160.pth',
'../Results/Hard01/SPO_hard01.presolve_0_Epoch0_240.pth',
'../Results/Hard01/SPO_hard01.presolve_0_Epoch0_320.pth']

param_data = data_reading(file)

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
i=0
for modelPATH in modelPATH_list:
	i+=1
	model = LinearRegression(8,1)
	model.load_state_dict(torch.load(modelPATH))
	clf =  SGD_SPO_generic(solver= Gurobi_ICON,accuracy_measure=False,relax=True, model = model,
		validation_relax= True,param= param_data, maximize= False,n_items=288,reset= True,presolve=True,
	                         epochs= 0,timelimit=13600, lr=1e-5)
	logging.info('Dummy model set')
	clf.fit(X_1gtrain,y_train)
	y_pred= clf.predict(X_1gtest)
	logging.info('Prediction Obtained')
	solutions = ICON_solution(y_pred,y_test,relax=True,
		n_items=288,reset=True,presolve=True,solver= Gurobi_ICON,**param_data)
	df = pd.DataFrame.from_dict(solutions)
	df['model'] = modelPATH
	df['time'] = str(2*i)+"hour"
	with open(filename, 'a') as f:
		df.to_csv(f, mode='a', header=f.tell()==0,index=False)
