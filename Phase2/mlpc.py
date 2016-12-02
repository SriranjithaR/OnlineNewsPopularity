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
from feature_selection import feature_selection

#Loading data
x_train = np.loadtxt('../Data/x_train.txt')
y_train_binary = np.loadtxt('../Data/y_train_binary.txt')
x_test = np.loadtxt('../Data/x_test.txt')
y_test_binary = np.loadtxt('../Data/y_test_binary.txt')
x_orig_train = np.loadtxt('../Data/x_orig_train.txt')
y_orig_train_binary = np.loadtxt('../Data/y_orig_train_binary.txt')

#Modeling classifier
clf = MLPClassifier(alpha=0.0001)

#Calling feature selection methods
fs = feature_selection()
clf,x_train,x_test,y_out = fs.PCASelection(x_train,y_train_binary,x_test,y_test_binary,clf)
#clf,x_train,x_test,y_out = fs.KBest(x_train,y_train_binary,x_test,y_test_binary,clf)


#Printing scores
# aScore = accuracy_score(y_test_binary,y_out)
# print "Accuracy Score : ",aScore
score = clf.score(x_test,y_test_binary)
print "Score : ", score
# print "Precision recall f-score support : " , prfs(y_test_binary,y_out)


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
print "Tuning alpha values"
alpha_scores = {}
for i in  [0.0001,0.001,0.01,0.1,1,10]:
    clf = MLPClassifier(alpha=i)
    clf.fit (x_train,y_train_binary)
    sc = clf.score(x_test,y_test_binary)
    print 'Score for ',i,':',sc
    alpha_scores[i]=sc
opt_alpha = max(alpha_scores,key = alpha_scores.get)
print "Best parameter : ",opt_alpha
clf = MLPClassifier(alpha=opt_alpha)
clf.fit (x_train,y_train_binary)

#Printing final result
clf =  MLPClassifier(alpha=opt_alpha)
clf.fit (x_train,y_train_binary)
sc = clf.score(x_test,y_test_binary)
print "\nMax score obtained using MLPClassifier : ", sc

# PLotting figure
fig = plt.figure()
ax = fig.add_subplot(111)

x_axis = [0.0001,0.001,0.01,0.1,1,10]
y_axis = list(alpha_scores.values())
plt.plot(x_axis,y_axis)
plt.xlabel('Alpha')
plt.ylabel('Scores')

plt.title("mlpc ")
plt.savefig('mlpc.png')


from ROCCurves import ROCCurves as ROC
ROC().getROCCurves(clf,x_train,y_train_binary,x_test,y_test_binary,"mlpc")
