import sys
sys.path.insert(0, '..')
from torch_SPO import *
from ICON import *

(X_1gtrain, y_train, X_1gtest, y_test) = get_energy()

file = "load1/day01.txt"
param_data = data_reading(file)


clf = Regression_generic(solver= ICON_scheduling,optimal_value= optimal_value,accuracy_measure=False,
                          param= param_data, getY= get_profits_ICON, getYpred = get_profits_pred_ICON,maximize= False,
                         epochs=5,lr= 1e-1 )
clf.fit(X_1gtrain,y_train,X_1gtest,y_test)

clf.test_score(X_1gtest,y_test)