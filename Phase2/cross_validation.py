
import math as m
import numpy as np

from sklearn.metrics import accuracy_score

class cross_validation:

    def crossValidation(self,x_train,y_train,clf,k):

        n = x_train.shape[0];
        d = x_train.shape[1];

        #Split the data across k iterations
        partSize = m.floor(n/k);
        for i in xrange(1,k+1):

            sIndex = m.floor(n/k*(i-1))

            fIndex = sIndex + partSize
            if (n-fIndex < partSize):
                fIndex = n;

            x_cv = x_train[sIndex:fIndex]
            y_cv = y_train[sIndex:fIndex]

            x_t = np.concatenate((x_train[0:max(0,sIndex)],x_train[fIndex:n]),axis=0);
            y_t = np.concatenate((y_train[0:max(0,sIndex)],y_train[fIndex:n]),axis=0);

            clf.fit(x_t,y_t)
            y_out = clf.predict(x_cv)

            #print "0",sIndex,fIndex-1,n,x_t.shape[0], x_cv.shape[0], x_t.shape[0]+x_cv.shape[0], x_train.shape[0]
            print accuracy_score(y_cv,y_out)
