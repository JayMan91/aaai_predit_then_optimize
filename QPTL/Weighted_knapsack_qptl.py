from melding_knapsack import *
from get_energy import get_energy

data = np.load('Data.npz')

X_1gtrain = data['X_1gtrain']
X_1gtest = data['X_1gtest']
y_train = data['y_train']
y_test = data['y_test']

X_1gvalidation = X_1gtest[0:2880,:]
y_validation= y_test[0:2880]

y_test= y_test[2880:]
X_1gtest = X_1gtest[2880:,:]

weights = [data['weights'].tolist()]

weights = np.array(weights)

for repitition  in range(2):
	for capa in range(30,220,30):
		clf = qptl(capa,weights,lr=1e-3,verbose=True,epochs=1,validation=True)
		pdf = clf.fit(X_1gtrain,y_train,X_1gtest,y_test,X_1gvalidation,y_validation)
		pdf['capacity'] = capa
		pdf_all = pd.read_csv("Weighted_QPTL.csv")
		pdf_all = pd.concat([pdf_all,pdf])
		pdf_all.to_csv("Weighted_QPTL.csv",index=False)
		del pdf_all
		del pdf