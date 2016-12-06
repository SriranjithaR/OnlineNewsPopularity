from sklearn.linear_model import LogisticRegression as LR
import numpy as np
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
x_test_copy = x_test
x_final_test = np.loadtxt('../Data/x_final_test.txt')
y_final_test_binary = np.loadtxt('../Data/y_final_test_binary.txt')


#Modeling the classifier
clf = LR(C = 1.0,solver ='sag')

#Calling feature selection methods
fs = feature_selection()
#clf,x_train,x_test,x_final_test,y_out = fs.PCASelection(x_train,y_train_binary,x_test,y_test_binary,x_final_test,clf)
clf,x_train,x_test,x_final_test,y_out = fs.KBest(x_train,y_train_binary,x_test,y_test_binary,x_final_test,clf)
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
print "Tuning C values"
c_scores = {}
for i in [0.01,0.1,1,10]:
    clf = LR(C = i)
    clf.fit (x_train,y_train_binary)
    sc = clf.score(x_test,y_test_binary)
    print 'Score for ',i,':',sc
    c_scores[i]=sc
opt_c = max(c_scores,key = c_scores.get)
print "Best parameter : ",opt_c
# print "Best parameter : ",best_params_

#Printing final result
clf =  LR(C = opt_c)
clf.fit (x_train,y_train_binary)
sc = clf.score(x_test,y_test_binary)
print "\nMax score obtained using Logistic classifier : ", sc

# PLotting figure
fig = plt.figure()
ax = fig.add_subplot(111)

x_axis = np.array([0.01,0.1,1,10])
y_axis = list(c_scores)
#plt.scatter(x_axis,y_axis)
plt.plot(x_axis,y_axis)
plt.xlabel('C')
plt.ylabel('Scores')

plt.title("Logistic Regression Classifier ")
plt.savefig('Figures/logistic/LogiisticRegression.png')

#Plotting ROC curve
from ROCCurves import ROCCurves as ROC
ROC().getROCCurves(clf,x_train,y_train_binary,x_test,y_test_binary,"logistic")

from final_test import finalTest as ft
ft(clf)