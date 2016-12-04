
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from itertools import cycle
from scipy import interp
import numpy as np

class ROCCurves:
    def getROCCurves(self,clf,x_train,y_train_binary,x_test,y_test_binary,classifierName):
        # Plot of a ROC curve for a specific class
        preds = clf.predict_proba(x_test)[:,1]
        fpr = []
        tpr = []
        roc_auc = []
        fpr, tpr, _ = roc_curve(y_test_binary, preds)
        roc_auc = auc(fpr, tpr)
        
        savename = 'Figures/' + classifierName + '/'+ classifierName +'ROC.png'
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example') 
        plt.savefig(savename)
        plt.legend(loc="lower right")
        plt.show()
        print 'ROC curve Area = '+ str(roc_auc)
            
        
        # Run classifier with cross-validation and plot ROC curves
        cv = KFold(n_splits=6)
        classifier = clf
        
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        
        colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
        lw = 2

        print 'AUC for CV:' 
        i = 0
        for (train, test), color in zip(cv.split(x_train, y_train_binary), colors):
            probas_ = classifier.fit(x_train[train], y_train_binary[train]).predict_proba(x_train[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y_train_binary[test], probas_[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=lw, color=color,
                     label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
            print "ROC fold "+ str(i) +"(area = "+ str(roc_auc) + ")" 
            i += 1
        plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
                 label='Luck')
        
        mean_tpr /= cv.get_n_splits(x_train, y_train_binary)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
                 label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
        
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        savename = 'Figures/' + classifierName + '/'+ classifierName +'ROCKFold.png'
        plt.savefig(savename)
        plt.legend(loc="lower right")
        plt.show()
