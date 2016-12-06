import numpy as np
import time
import matplotlib.pyplot as plt

#Loading data
x_train = np.loadtxt('../Data/x_train.txt')
y_train = np.loadtxt('../Data/y_train.txt')
y_train_binary = np.loadtxt('../Data/y_train_binary.txt')
x_test = np.loadtxt('../Data/x_test.txt')
y_test = np.loadtxt('../Data/y_test.txt')
y_test_binary = np.loadtxt('../Data/y_test_binary.txt')
x_orig_train = np.loadtxt('../Data/x_orig_train.txt')
y_orig_train_binary = np.loadtxt('../Data/y_orig_train_binary.txt')

for i in range(1,59):
	plt.scatter(x_train[:,i],y_train,c='b')
	plt.savefig('feature_'+str(i)+'_vs_y')
	plt.clf()

# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)


# median = np.median(y_train)

# print median

# y_binary = y_train < median

# print y_binary

# color = ['r' if x == True  else 'b' for x in y_binary]


# plt.scatter(x_train[:,0],x_train[:,1],c=color)
# plt.show()
# # print y_tr	ain
# # plt.hist(y_train, 1000, normed=1, facecolor='green', alpha=0.5)
# # plt.show()
# plt.savefig('PCA_2_components')