import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from cross_validation import cross_validation as CV
import matplotlib.pyplot as plt



#Loading data
x_train = np.loadtxt('../Data/x_train.txt')
y_train_binary = np.loadtxt('../Data/y_train_binary.txt')
x_test = np.loadtxt('../Data/x_test.txt')
y_test_binary = np.loadtxt('../Data/y_test_binary.txt')
x_orig_train = np.loadtxt('../Data/x_orig_train.txt')
y_orig_train_binary = np.loadtxt('../Data/y_orig_train_binary.txt')

#Applying PCA
# num_features = 10
# from sklearn.decomposition import PCA
# pca = PCA(n_components=num_features)
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)

#Modeling classifier
from sklearn import svm
svr = svm.SVC(C=1.0, kernel='linear' )
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
clf = GridSearchCV(svr, parameters);
clf.fit (x_train,y_train_binary)
y_out = clf.predict(x_test)

#Printing scores
aScore = accuracy_score(y_test_binary,y_out)
print "Accuracy Score : ",aScore
score = clf.score(x_test,y_test_binary)
print "Score : ", score
print "Precision recall f-score support : " , prfs(y_test_binary,y_out)


# #Cross validation
# cval = CV()
# folds = 10
# print "10 fold cross validation score"
# cval.crossValidation(x_orig_train,y_orig_train_binary,clf,folds);
# scores = cross_val_score(clf, x_orig_train, y_orig_train_binary, cv=10)
#
# #Inbuilt CV function
# skf =  KFold(n_splits=folds,shuffle = False)
# skfscore = cross_val_score(clf, x_orig_train, y_orig_train_binary, cv=skf)
# print skfscore

# #Manual Parameter tuning
# for i in range(1,10):
#     clf = dtc(max_depth = i)
#     clf.fit (x_train,y_train_binary)
#     print 'Score for ',i,':',clf.score(x_test,y_test_binary)
#
# #Parameter tuning using grid search CV
# parameters = {'max_depth' : list(range(1,10)) }
# gscv = GridSearchCV(clf,parameters)
# gscv.fit(x_train,y_train_binary)
# print gscv.score(x_test,y_test_binary)
# print gscv.best_params_
# #print gscv.cv_results_
# means = gscv.cv_results_['mean_test_score']
#
#
# # PLotting figure
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# x_axis = np.array(range(1,10))
# y_axis = means
# plt.plot(x_axis,y_axis)
# plt.xlabel('Max_depth')
# plt.ylabel('Mean scores')
#
# plt.title("DecTreeClassifier ")
# plt.savefig('DecTreeClassifier.png')
