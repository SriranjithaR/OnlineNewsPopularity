fileread - changed, now stores values in files

linreg
ensemble, with some warnings
ridge, with some warnings
mlp
DecisionTreeRegressor


cl - knn
cl - random forest
cl - SVC - won't even run
cl - nbc - no params to tune
cl - MLPClassifier -- best accuracy so far
cl - DecisionTreeClassifier - PCA doesn't work well with this
cl - AdaBoostClassifier -- also good accuracy
cl - Stochastic gradient descent

wrote the test and training data to files.

Problems :-
1. mlpc : Scores not the same for same alpha values :|
            Different values on each run
