# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 23:39:23 2016

@author: Sriranjitha
"""
import numpy as np
import matplotlib.pyplot as plt

def DatasetVsAccuracyPlot(clf,x_train,y_train,x_test,y_test,classifierName):
    
    #Calculate scores based on dataset size
    plot = {}
    for i in np.arange(0.1,1.1,0.1):
        msk = np.random.rand(len(y_train))<i    
        x_train_subset = x_train[msk]
        y_train_subset = y_train[msk]
        msk = np.random.rand(len(y_test))<i
        x_test_subset = x_test[msk]
        y_test_subset = y_test[msk]
        
        clf.fit(x_train_subset,y_train_subset)
        sc = clf.score(x_test_subset,y_test_subset)
        plot[i] = sc
        
    #Plot graph
    x_axis = list(plot.keys())
    y_axis = list(plot.values())
    plt.scatter(x_axis,y_axis)
    plt.xlabel('Percentage of dataset')
    plt.ylabel('Scores')
    
    savename = 'Figures/' + classifierName + '/'+ classifierName + 'DVAP.png'
    
    plt.title("Dataset size Vs Accuracy plot ")
    plt.savefig(savename)
        
        
        