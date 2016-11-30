from fileread import x_train,y_train,x_test,y_test
import numpy as np
from sklearn.svm import SVR
e = 0.5
C = 10
clf = SVR(C = C,epsilon =e,cache_size=10000,kernel='rbf')
num_features = 30
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

import matplotlib.pyplot as plt
# fig1 = plt.figure()
x_axis = np.array(range(1,len(residual)+1))
# print x_axis.shape
# print y_out
# print residual

plt.plot(y_out,residual,'rs')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title("SVR : Features = "+str(num_features)+" ,Score : "+str(score)+" ,E = "+str(e)+" ,C= "+str(C))
plt.savefig('SVR_residual_features_'+str(num_features)+"_E_"+str(e)+"_C_"+str(C)+'.png')
