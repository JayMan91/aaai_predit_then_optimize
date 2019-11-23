from ICON import *
from get_energy import get_energy
from torch_SPO_updated import *
import time
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

params = data_reading("load2/day01.txt")
start = time.time()
T= None

clf =  Gurobi_ICON(relax=True,reset=False,presolve=True,warmstart=True,verbose=True, **params)
clf.make_model()

start = time.time()
clf.solve_model(y_train[0:48])
end = time.time()

clf.solve_model(y_train[48:96])
clf.solve_model(y_train[96:144])


#print(clf.model.getAttr(GRB.Attr.X, clf.model.getVars()))
#print(clf.model.getAttr(GRB.Attr.VBasis, clf.model.getVars()))

#print(clf.model.getAttr(GRB.Attr.CBasis, clf.model.getConstrs()))

