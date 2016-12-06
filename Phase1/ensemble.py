import numpy as np
from sklearn import ensemble
n_e =100
alpha =0.01
clf = ensemble.AdaBoostRegressor(n_estimators =n_e, learning_rate = alpha,loss = 'square')



#Loading data
x_train = np.loadtxt('../Data/x_train.txt')
y_train = np.loadtxt('../Data/y_train.txt')
y_train_binary = np.loadtxt('../Data/y_train_binary.txt')
x_test = np.loadtxt('../Data/x_test.txt')
y_test = np.loadtxt('../Data/y_test.txt')
y_test_binary = np.loadtxt('../Data/y_test_binary.txt')
x_orig_train = np.loadtxt('../Data/x_orig_train.txt')
y_orig_train_binary = np.loadtxt('../Data/y_orig_train_binary.txt')


# Uncomment to Use Feature selection and put the number of features
num_features = 'all'
# from sklearn.decomposition import PCA
# pca = PCA(n_components=num_features)
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)


clf.fit (x_train,y_train)
y_out = clf.predict(x_test)
# print y_out
# print y_test
from sklearn.metrics import mean_squared_error
score = clf.score(x_test,y_test)
print "Regressinon Score: "+ str(score)
residual = y_test - y_out


from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics.pairwise import cosine_similarity as cs
print cs(y_out,y_test)

#Convert Into Binary Data
y_out_binary = y_out < 1400

# Results for classification
print prfs(y_test_binary,y_out_binary)

import matplotlib.pyplot as plt
# fig1 = plt.figure()
x_axis = np.array(range(1,len(residual)+1))
# print x_axis.shape
# print y_out
# print residual

plt.plot(y_out,residual,'rs')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title("AdaBoost : Features = "+str(num_features)+" ,Score : "+str(score)+" ,alpha = "+str(alpha)+" ,num = "+str(n_e))
plt.savefig('AdaBoost_residual_features_'+str(num_features)+"_alpha_"+str(alpha)+"_num_"+str(n_e)+'.png')
