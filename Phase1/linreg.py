from fileread import x_train,y_train,x_test,y_test
import numpy as np
from sklearn import linear_model
clf = linear_model.LinearRegression(normalize='false')
num_features = 10
from sklearn.decomposition import PCA
pca = PCA(n_components=num_features)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

clf.fit (x_train,y_train)
y_out = clf.predict(x_test)
# print y_out
# print y_test
from sklearn.metrics import mean_squared_error
score = clf.score(x_test,y_test)
print score
residual = y_test - y_out

import matplotlib.pyplot as plt
# fig1 = plt.figure()
x_axis = np.array(range(1,len(residual)+1))
# print x_axis.shape
# print y_out
# print residual

plt.plot(y_out,residual,'rs')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title("Linear Regression : Features = "+str(num_features)+" Score : "+str(score))
plt.savefig('linreg_residual_features_'+str(num_features)+'.png')
