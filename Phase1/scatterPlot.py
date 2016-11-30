from fileread import x_train,y_train,x_test,y_test
import numpy as np
import time
import matplotlib.pyplot as plt
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