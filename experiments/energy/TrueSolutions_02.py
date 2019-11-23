import sys
sys.path.insert(0,'../../EnergyCost/')
sys.path.insert(0,"../..")
from torch_SPO_updated import *
from ICON import *
import time,datetime
import pandas as pd
import logging

formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='SPO_hard02_solutions.log', level=logging.INFO,format=formatter)
logging.info('Started\n')

file = "../../EnergyCost/Hard_Instances/instance02/instance.txt"
filename = "Hard02/Hard02_test_solutions.csv"

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

solutions = ICON_solution(y_test,y_test,relax=True,reset=True,presolve=True,n_items=288,solver= Gurobi_ICON,**param_data)
df = pd.DataFrame.from_dict(solutions)
with open(filename, 'a') as f:
	df.to_csv(f, mode='a', header=f.tell()==0,index=False)

	
