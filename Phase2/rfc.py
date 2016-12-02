import numpy as np
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
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


#Model fitting
clf = rfc()

#Calling feature selection methods
fs = feature_selection()
#clf,x_train,x_test,y_out = fs.PCASelection(x_train,y_train_binary,x_test,y_test_binary,clf)
clf,x_train,x_test,y_out = fs.KBest(x_train,y_train_binary,x_test,y_test_binary,clf)

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
n_est_scores = {}
for i in list(range(10,120,10)):
    clf = rfc(n_estimators = i)
    clf.fit (x_train,y_train_binary)
    sc = clf.score(x_test,y_test_binary)
    print 'Score for ',i,':',sc
    n_est_scores[i]=sc
opt_n_est = max(n_est_scores,key = n_est_scores.get)
print "Best parameter : ",opt_n_est
clf = rfc(n_estimators = opt_n_est)
clf.fit (x_train,y_train_binary)

print "\nManual parameter tuning"
max_depth_scores = {}
for i in list(range(5,30,5)):
    clf = rfc(n_estimators = opt_n_est, max_depth = i)
    clf.fit (x_train,y_train_binary)
    sc = clf.score(x_test,y_test_binary)
    print 'Score for ',i,':',sc
    max_depth_scores[i]=sc
opt_max_depth = max(max_depth_scores,key = max_depth_scores.get)
print "Best parameter : ",opt_max_depth
clf = rfc(n_estimators = opt_n_est, max_depth = opt_max_depth)
clf.fit (x_train,y_train_binary)

##Parameter tuning
#parameters = {'max_depth' : list(range(1,10)) }
#gscv = GridSearchCV(clf,parameters)
#gscv.fit(x_train,y_train_binary)
#print gscv.score(x_test,y_test_binary)
#print gscv.best_params_
##print gscv.cv_results_



#Printing final result
clf = rfc(n_estimators = opt_n_est,max_depth = opt_max_depth)
clf.fit (x_train,y_train_binary)
sc = clf.score(x_test,y_test_binary)
print "\nMax score obtained using decision tree : ", sc


## Plotting figure
#fig = plt.figure()
#ax = fig.add_subplot(111)
#
#x_axis = np.array(range(1,10))
#y_axis = means
#plt.plot(x_axis,y_axis)
#plt.xlabel('learn_rate')
#plt.ylabel('Mean scores')
#
#plt.title("AdaBoostClassifier ")
#plt.savefig('AdaBoostClassifier.png')

from ROCCurves import ROCCurves as ROC
ROC().getROCCurves(clf,x_train,y_train_binary,x_test,y_test_binary)
