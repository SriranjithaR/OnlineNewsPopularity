import numpy as np
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from cross_validation import cross_validation as CV
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif




#Loading data
x_train = np.loadtxt('../Data/x_train.txt')
y_train_binary = np.loadtxt('../Data/y_train_binary.txt')
x_test = np.loadtxt('../Data/x_test.txt')
y_test_binary = np.loadtxt('../Data/y_test_binary.txt')
x_orig_train = np.loadtxt('../Data/x_orig_train.txt')
y_orig_train_binary = np.loadtxt('../Data/y_orig_train_binary.txt')


#Model fitting
e = 0.5
C = 1.0
clf = rfc(n_estimators = 100 , max_depth=5)

#Applying PCA
pca_scores = {}
for nf in list(range(45,50)):
    pca = PCA(n_components= nf )
    x_train_new = pca.fit_transform(x_train)
    x_test_new = pca.transform(x_test)
    clf.fit (x_train_new,y_train_binary)
    y_out = clf.predict(x_test_new)
    score = clf.score(x_test_new,y_test_binary)
    print "Score  for " ,nf," features : ",  score
    pca_scores[nf] = score

max_score_nf = max(pca_scores,key=pca_scores.get)
print "Max score was obtained for : ",max_score_nf," features"
pca = PCA(n_components=  max_score_nf)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)


clf.fit (x_train,y_train_binary)
y_out = clf.predict(x_test)
# print y_out
# print y_test

clf.fit (x_train,y_train_binary)
print clf.feature_importances_
y_out = clf.predict(x_test)

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

#Parameter tuning
parameters = {'max_depth' : list(range(1,10)) }
gscv = GridSearchCV(clf,parameters)
gscv.fit(x_train,y_train_binary)
print gscv.score(x_test,y_test_binary)
print gscv.best_params_
#print gscv.cv_results_



#Printing final result
clf = abc(n_estimators = opt_n_est,learning_rate = opt_learn_rate)
clf.fit (x_train,y_train_binary)
sc = clf.score(x_test,y_test_binary)
print "\nMax score obtained using decision tree : ", sc


# # Plotting figure
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# x_axis = np.array(range(1,10))
# y_axis = means
# plt.plot(x_axis,y_axis)
# plt.xlabel('learn_rate')
# plt.ylabel('Mean scores')
#
# plt.title("AdaBoostClassifier ")
# plt.savefig('AdaBoostClassifier.png')
