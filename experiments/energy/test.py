import sys
sys.path.insert(0,'../..')
from get_energy import get_energy

(X_1gtrain, y_train, X_1gtest, y_test) = get_energy()
print("DONE")