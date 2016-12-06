import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA

class feature_selection:
    #Applying SelectKBest
    def KBest(self,x_train,y_train_binary,x_test,y_test_binary
    ,clf):
        print "\nApplying SelectKBest"
        skbest_scores = {}
        for nf in list(range(1,60)):
            skbest = SelectKBest(f_classif, k = nf )
            x_train_new = skbest.fit_transform(x_train,y_train_binary)
            x_test_new = skbest.transform(x_test)
            clf.fit (x_train_new,y_train_binary)
            score = clf.score(x_test_new,y_test_binary)
            print "Score  for " ,nf," features : ",  score
            skbest_scores[nf] = score

        max_score_nf = max(skbest_scores,key=skbest_scores.get)
        print "Max score was obtained for : ",max_score_nf," features"

        #Fitting to best no of features
        skbest = SelectKBest(f_classif, k = max_score_nf )
        x_train = skbest.fit_transform(x_train,y_train_binary)
        x_test = skbest.transform(x_test)
        clf.fit (x_train,y_train_binary)
        y_out = clf.predict(x_test)
        return (clf,x_train,x_test,y_out)

    def PCASelection(self,x_train,y_train_binary,x_test,y_test_binary
    ,clf):
        #Applying PCA
        print "\nApplying PCA"
        pca_scores = {}
        for nf in list(range(1,60)):
            pca = PCA(n_components= nf )
            x_train_new = pca.fit_transform(x_train)
            x_test_new = pca.transform(x_test)
            clf.fit (x_train_new,y_train_binary)
            score = clf.score(x_test_new,y_test_binary)
            print "Score  for " ,nf," features : ",  score
            pca_scores[nf] = score

        max_score_nf = max(pca_scores,key=pca_scores.get)
        #max_score_nf = 59
        print "Max score was obtained for : ",max_score_nf," features"
        pca = PCA(n_components=  max_score_nf)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)

        clf.fit (x_train,y_train_binary)
        y_out = clf.predict(x_test)
        return (clf,x_train,x_test,y_out)
