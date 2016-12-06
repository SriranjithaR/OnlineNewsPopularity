import numpy as np
from sklearn.ensemble import AdaBoostClassifier as abc
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from cross_validation import cross_validation as CV
import matplotlib.pyplot as plt
from feature_selection import feature_selection

#Loading data
x_train = np.loadtxt('../Data/x_train.txt')
y_train_binary = np.loadtxt('../Data/y_train_binary.txt')
x_test = np.loadtxt('../Data/x_test.txt')
y_test_binary = np.loadtxt('../Data/y_test_binary.txt')
x_orig_train = np.loadtxt('../Data/x_orig_train.txt')
y_orig_train_binary = np.loadtxt('../Data/y_orig_train_binary.txt')
x_final_test = np.loadtxt('../Data/x_final_test.txt')
y_final_test_binary = np.loadtxt('../Data/y_final_test_binary.txt')

#Modeling classifier
clf = abc()

#Calling feature selection methods
fs = feature_selection()
#clf,x_train,x_test,x_final_test,y_out = fs.PCASelection(x_train,y_train_binary,x_test,y_test_binary,x_final_test,clf)
#clf,x_train,x_test,x_final_test,y_out = fs.KBest(x_train,y_train_binary,x_test,y_test_binary,x_final_test,clf)
clf.fit (x_train,y_train_binary)
y_out = clf.predict(x_test)

#Printing scores
score = clf.score(x_test,y_test_binary)
print "Score : ", score
print "Precision recall f-score support : " , prfs(y_test_binary,y_out)


#Cross validation
folds = 2
print "\nManual ",folds," fold cross validation score"
CV(x_orig_train,y_orig_train_binary,clf,folds);
scores = cross_val_score(clf, x_orig_train, y_orig_train_binary, cv=10)

#Checking with inbuilt CV function
print "\nChecking with inbuilt function"
skf =  KFold(n_splits=folds,shuffle = False)
skfscore = cross_val_score(clf, x_orig_train, y_orig_train_binary, cv=skf)
print skfscore

#Manual Parameter tuning
print "\nManual parameter tuning"
print "Tuning learning rate :"
learn_rate_scores = {}
for i in [0.1,1,2]:
    clf = abc(learning_rate = i)
    clf.fit (x_train,y_train_binary)
    sc = clf.score(x_test,y_test_binary)
    print 'Score for ',i,':',sc
    learn_rate_scores[i]=sc
opt_learn_rate = max(learn_rate_scores,key = learn_rate_scores.get)
print "Best parameter : ",opt_learn_rate

print "Tuning number of estimators"
n_est_scores = {}
for i in range(20,70,5):
    clf = abc(n_estimators = i,learning_rate = opt_learn_rate)
    clf.fit (x_train,y_train_binary)
    sc = clf.score(x_test,y_test_binary)
    print 'Score for ',i,':',sc
    n_est_scores[i]=sc
opt_n_est = max(n_est_scores,key = n_est_scores.get)
print "Best parameter : ",opt_n_est


#Printing final result
clf = abc(n_estimators = opt_n_est,learning_rate = opt_learn_rate)
clf.fit (x_train,y_train_binary)
sc = clf.score(x_test,y_test_binary)
print "\nMax score obtained using AdaBoost Classifier : ", sc

# Plotting figure
fig = plt.figure(1)
x_axis = list(n_est_scores.keys())
y_axis = list(n_est_scores.values())
plt.scatter(x_axis,y_axis)
plt.xlabel('No. of estimators')
plt.ylabel('Scores')
plt.savefig('Figures/adaBoost/AdaBoostClassifier1.png')

fig = plt.figure(2)
learn_rate_scores = {}
for i in [0.1,1,2]:
    clf = abc(learning_rate = i,n_estimators=opt_n_est)
    clf.fit (x_train,y_train_binary)
    sc = clf.score(x_test,y_test_binary)
    learn_rate_scores[i]=sc
x_axis = list(learn_rate_scores.keys())
y_axis = list(learn_rate_scores.values())
plt.scatter(x_axis,y_axis)
plt.xlabel('Learning rate')
plt.ylabel('Scores')
plt.savefig('Figures/adaBoost/AdaBoostClassifier2.png')

#Plotting ROC curve
from ROCCurves import ROCCurves as ROC
ROC().getROCCurves(clf,x_train,y_train_binary,x_test,y_test_binary,"adaBoost")

#Dataset Vs Accuracy plot
from DatasetVsAccuracy import DatasetVsAccuracyPlot as dvap
dvap(clf,x_train,y_train_binary,x_test,y_test_binary,"adaBoost")

from final_test import finalTest as ft
ft(clf)