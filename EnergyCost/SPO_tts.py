from torch_SPO_updated import *
from ICON import *
import time
import pandas as pd

file = "load1/day01.txt"
param_data = data_reading(file)
(X_1gtrain, y_train, X_1gtest, y_test) = get_energy()
X_1gvalidation = X_1gtest[0:2880,:]
y_validation = y_test[0:2880]

y_test= y_test[2880:]
X_1gtest = X_1gtest[2880:,:]
n_iter= 10

h= {'lr':1e-5,'momentum':0.01}
clf = SGD_SPO_generic(solver= Gurobi_ICON,accuracy_measure=False,
	                        verbose=True,param= param_data, maximize= False,
	                         epochs= 1,timelimit=100, **h )

start = time.time()
df = clf.fit(X_1gtrain,y_train,X_1gvalidation,y_validation,X_1gtest,y_test)
end = time.time()
df.to_csv("test.csv",index=False)
print("Esecution time:%f, Model Time:%f"%(end-start,clf.timelimit))