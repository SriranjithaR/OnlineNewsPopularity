from fileread import x_train,y_train,x_test,y_test
import numpy as np
from sklearn.linear_model import LogisticRegression
e = 0.5
C = 1.0
clf = LogisticRegression(C = 0.01,solver ='sag')
# num_features = 1
# from sklearn.decomposition import PCA
# pca = PCA(n_components=num_features)
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)



median = np.median(y_train)

print median

y_train_binary = y_train < median

y_test_binary = y_test < median

clf.fit (x_train,y_train)

y_out = clf.predict(x_test)
# print y_out
# print y_test
# from sklearn.metrics import mean_squared_error
score = clf.score(x_test,y_test)
print score
# residual = y_test - y_out

# import matplotlib.pyplot as plt
# # fig1 = plt.figure()
# x_axis = np.array(range(1,len(residual)+1))
# # print x_axis.shape
# # print y_out
# # print residual

# plt.plot(y_out,residual,'rs')
# plt.xlabel('Predicted')
# plt.ylabel('Residuals')
# plt.title("SVR : Features = "+str(num_features)+" ,Score : "+str(score)+" ,E = "+str(e)+" ,C= "+str(C))
# plt.savefig('SVR_residual_features_'+str(num_features)+"_E_"+str(e)+"_C_"+str(C)+'.png')