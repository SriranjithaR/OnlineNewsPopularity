import numpy as np
from sklearn import neural_network
clf = neural_network.MLPRegressor(hidden_layer_sizes=100,activation='logistic',solver ='sgd',alpha=10)



#Loading data
x_train = np.loadtxt('../Data/x_train.txt')
y_train = np.loadtxt('../Data/y_train.txt')
y_train_binary = np.loadtxt('../Data/y_train_binary.txt')
x_test = np.loadtxt('../Data/x_test.txt')
y_test = np.loadtxt('../Data/y_test.txt')
y_test_binary = np.loadtxt('../Data/y_test_binary.txt')
x_orig_train = np.loadtxt('../Data/x_orig_train.txt')
y_orig_train_binary = np.loadtxt('../Data/y_orig_train_binary.txt')


# Feature selection using PCA
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