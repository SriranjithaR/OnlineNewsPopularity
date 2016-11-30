from fileread import x_train,y_train,x_test,y_test
import numpy as np
from sklearn import tree
max_depth = None
clf = tree.DecisionTreeRegressor(criterion ='mse',splitter='best',max_depth = max_depth)
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
print score
residual = y_test - y_out


from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics.pairwise import cosine_similarity as cs
print cs(y_out,y_test)

y_out_new = y_out < 1400
print sum(y_out_new)
# print y_out_new
y_test_new = y_test < 1400
# print y_test_new
print prfs(y_test_new,y_out_new)

import matplotlib.pyplot as plt
# fig1 = plt.figure()
x_axis = np.array(range(1,len(residual)+1))
# print x_axis.shape
# print y_out
# print residual

plt.plot(y_out,residual,'rs')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title("Decision Tree Regression : Features = "+str(num_features)+" ,Score : "+str(score)+" ,depth = "+str(max_depth))
plt.savefig('decisionTree_residual_features_'+str(num_features)+"_depth_"+str(max_depth)+'.png')