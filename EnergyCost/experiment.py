from ICON import *
from get_energy import get_energy
import time
(X_1gtrain, y_train, X_1gtest, y_test) = get_energy()
file = "Hard_Instances/instance02/instance.txt"


param_data = data_reading(file)
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
start = time.time()
clf = Gurobi_ICON(relax=True,method=1, **param_data)
clf.make_model()
end = time.time()
print("Model Buliding Time",end-start)

start = time.time()
sol,tup,t = clf.solve_model(y_train[0:288],scheduling=True)
end = time.time()
print(end-start)

start = time.time()
sol,tup,t = clf.solve_model(y_train[576:864],scheduling=True,warmstart=tup)
end = time.time()
print(end-start)