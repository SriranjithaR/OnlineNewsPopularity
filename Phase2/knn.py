import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from cross_validation import cross_validation as CV
import matplotlib.pyplot as plt
from feature_selection import feature_selection
from sklearn.neighbors import KNeighborsClassifier as knn

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
clf = knn()

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
print "Tuning number of neighbours"
n_nei_scores = {}
for i in range(1,10):
    clf = knn(n_neighbors = i)
    clf.fit (x_train,y_train_binary)
    sc = clf.score(x_test,y_test_binary)
    print 'Score for ',i,':',sc
    n_nei_scores[i]=sc
opt_n_nei = max(n_nei_scores,key = n_nei_scores.get)
print "Best parameter : ",opt_n_nei


#Printing final result
clf = knn(n_neighbors = opt_n_nei)
clf.fit (x_train,y_train_binary)
sc = clf.score(x_test,y_test_binary)
print "\nMax score obtained using KNN Classifier : ", sc


# Plotting figure
fig = plt.figure()
ax = fig.add_subplot(111)

x_axis = np.array(range(1,10))
y_axis = list(n_nei_scores.values())
#plt.scatter(x_axis,y_axis)
plt.plot(x_axis,y_axis)
plt.xlabel('No of neighbours')
plt.ylabel('Mean scores')

plt.title("KNN Classifier ")
plt.savefig('Figures/knn/KNNClassifier.png')

#Plotting ROC curve
from ROCCurves import ROCCurves as ROC
ROC().getROCCurves(clf,x_train,y_train_binary,x_test,y_test_binary,"knn")

from final_test import finalTest as ft
ft(clf,x_final_test,y_final_test_binary)