
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
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
clf = MLPClassifier(alpha=0.0001)
clf.fit (x_train,y_train_binary)
#y_out = clf.predict(x_test)
clf.fit (x_train,y_train_binary)

#Printing scores
# aScore = accuracy_score(y_test_binary,y_out)
# print "Accuracy Score : ",aScore
score = clf.score(x_test,y_test_binary)
print "Score : ", score
# print "Precision recall f-score support : " , prfs(y_test_binary,y_out)


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

#Manual Parameter tuning
max_depth_scores = []
print "Tuning based on alpha"
for i in [0.0001,0.001,0.01,0.1,1,10]:
    clf = MLPClassifier(alpha=i)
    clf.fit (x_train,y_train_binary)
    sc = clf.score(x_test,y_test_binary)
    print 'Score for ',i,':',sc
    max_depth_scores.append(sc)

#Parameter tuning using grid search CV
parameters = {'alpha' : [0.0001,0.001,0.01,0.1,1,10] }
gscv = GridSearchCV(clf,parameters)
gscv.fit(x_train,y_train_binary)
print gscv.score(x_test,y_test_binary)
print gscv.best_params_
#print gscv.cv_results_
means = gscv.cv_results_['mean_test_score']


# PLotting figure
fig = plt.figure()
ax = fig.add_subplot(111)

x_axis = [0.0001,0.001,0.01,0.1,1,10]
y_axis = max_depth_scores
plt.plot(x_axis,y_axis)
plt.xlabel('Alpha')
plt.ylabel('Scores')

plt.title("mlpc ")
plt.savefig('mlpc.png')
