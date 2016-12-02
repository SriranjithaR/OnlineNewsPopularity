import numpy as np
from sklearn.ensemble import AdaBoostClassifier as abc
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from cross_validation import cross_validation as CV
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from feature_selection import feature_selection
from sklearn.metrics import roc_curve, auc

#Loading data
x_train = np.loadtxt('../Data/x_train.txt')
y_train_binary = np.loadtxt('../Data/y_train_binary.txt')
x_test = np.loadtxt('../Data/x_test.txt')
y_test_binary = np.loadtxt('../Data/y_test_binary.txt')
x_orig_train = np.loadtxt('../Data/x_orig_train.txt')
y_orig_train_binary = np.loadtxt('../Data/y_orig_train_binary.txt')

#Modeling classifier
clf = abc()

#Calling feature selection methods
fs = feature_selection()
clf,x_train,x_test,y_out = fs.PCASelection(x_train,y_train_binary,x_test,y_test_binary,clf)
#clf,x_train,x_test,y_out = fs.KBest(x_train,y_train_binary,x_test,y_test_binary,clf)

#Printing scores
aScore = accuracy_score(y_test_binary,y_out)
print "Accuracy Score : ",aScore
score = clf.score(x_test,y_test_binary)
print "Score : ", score
print "Precision recall f-score support : " , prfs(y_test_binary,y_out)


#Cross validation
cval = CV()
folds = 2
print "\nManual ",folds," fold cross validation score"
cval.crossValidation(x_orig_train,y_orig_train_binary,clf,folds);
scores = cross_val_score(clf, x_orig_train, y_orig_train_binary, cv=10)

#Checking with inbuilt CV function
print "\nChecking with inbuilt function"
skf =  KFold(n_splits=folds,shuffle = False)
skfscore = cross_val_score(clf, x_orig_train, y_orig_train_binary, cv=skf)
print skfscore

#Manual Parameter tuning
print "\nManual parameter tuning"
learn_rate_scores = {}
for i in [0.1,1,2]:
    clf = abc(learning_rate = i)
    clf.fit (x_train,y_train_binary)
    sc = clf.score(x_test,y_test_binary)
    print 'Score for ',i,':',sc
    learn_rate_scores[i]=sc
opt_learn_rate = max(learn_rate_scores,key = learn_rate_scores.get)
print "Best parameter : ",opt_learn_rate
clf = abc(learning_rate = opt_learn_rate)
clf.fit (x_train,y_train_binary)

n_est_scores = {}
for i in range(20,70,5):
    clf = abc(n_estimators = i)
    clf.fit (x_train,y_train_binary)
    sc = clf.score(x_test,y_test_binary)
    print 'Score for ',i,':',sc
    n_est_scores[i]=sc
opt_n_est = max(n_est_scores,key = n_est_scores.get)
print "Best parameter : ",opt_n_est
clf = abc(n_estimators = opt_n_est)
# print "Best parameter : ",best_params_


# #Parameter tuning using grid search CV
# print "\nChecking with inbuilt parameter tuning function"
# parameters = {'learning_rate' : [0.1,1,2], 'n_estimators' : [20,30,40,50,60,70] }
# gscv = GridSearchCV(clf,parameters,cv=skf)
# gscv.fit(x_train,y_train_binary)
# print "Best score : ", gscv.score(x_test,y_test_binary)
# print "Best estimator : ",gscv.best_estimator_
# best_param = gscv.best_params_
# print "Best parameter : ",best_param
# opt_learn_rate = best_param['learning_rate']

#Printing final result
clf = abc(n_estimators = opt_n_est,learning_rate = opt_learn_rate)
clf.fit (x_train,y_train_binary)
sc = clf.score(x_test,y_test_binary)
print "\nMax score obtained using AdaBoost Classifier : ", sc


# Plotting figure
fig = plt.figure()
ax = fig.add_subplot(111)

x_axis = np.array(range(20,70,5))
y_axis = list(n_est_scores.values())
plt.plot(x_axis,y_axis)
plt.xlabel('learn_rate')
plt.ylabel('Mean scores')

plt.title("AdaBoostClassifier ")
plt.savefig('AdaBoostClassifier.png')

#Plotting ROC curve
from ROCCurves import ROCCurves as ROC
ROC().getROCCurves(clf,x_train,y_train_binary,x_test,y_test_binary)
