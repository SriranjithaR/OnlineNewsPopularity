import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.tree import DecisionTreeClassifier as dtc
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

#Modeling classifier
clf = dtc(max_depth = 3)

#Calling feature selection methods
fs = feature_selection()
#clf,x_train,x_test,y_out = fs.PCASelection(x_train,y_train_binary,x_test,y_test_binary,clf)
clf,x_train,x_test,y_out = fs.KBest(x_train,y_train_binary,x_test,y_test_binary,clf)

#Printing scores
clf.fit (x_train,y_train_binary)
y_out = clf.predict(x_test)
aScore = accuracy_score(y_test_binary,y_out)
print "Accuracy Score : ",aScore
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
print "Tuning max depth"
max_depth_scores = {}
for i in range(2,20,2):
    clf = dtc(max_depth = i)
    clf.fit (x_train,y_train_binary)
    sc = clf.score(x_test,y_test_binary)
    print 'Score for ',i,':',sc
    max_depth_scores[i]=sc
opt_max_depth = max(max_depth_scores,key = max_depth_scores.get)
print "Best parameter : ",opt_max_depth

#Printing final result
clf = dtc(max_depth = opt_max_depth)
clf.fit (x_train,y_train_binary)
sc = clf.score(x_test,y_test_binary)
print "\nMax score obtained using decision tree : ", sc


# PLotting figure
fig = plt.figure()
ax = fig.add_subplot(111)

x_axis = np.array(range(2,20,2))
y_axis = list(max_depth_scores.values())
plt.plot(x_axis,y_axis)
plt.xlabel('Max_depth')
plt.ylabel('Scores')

plt.title("DecTreeClassifier ")
plt.savefig('DecTreeClassifier.png')

#Plotting ROC curve
from ROCCurves import ROCCurves as ROC
ROC().getROCCurves(clf,x_train,y_train_binary,x_test,y_test_binary,'decTree')

from DatasetVsAccuracy import DatasetVsAccuracyPlot as dvap
dvap(clf,x_train,y_train_binary,x_test,y_test_binary,"decTree")
