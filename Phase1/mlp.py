from fileread import x_train,y_train,x_test,y_test
import numpy as np
from sklearn import neural_network
clf = neural_network.MLPRegressor(hidden_layer_sizes=100,activation='logistic',solver ='sgd',alpha=10)


from sklearn.decomposition import PCA
pca = PCA(n_components=30)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)


clf.fit (x_train,y_train)	
y_out = clf.predict(x_test)
# print y_out
# print y_test
score = clf.score(x_test,y_test)
print score


residual = y_test - y_out
# print residual.shape

import matplotlib.pyplot as plt
# fig1 = plt.figure()
x_axis = np.array(range(1,len(residual)+1))
# print x_axis.shape
# print y_out
# print residual
plt.plot(y_out,residual,'rs')
plt.savefig('mlp_residual.png')