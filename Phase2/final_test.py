# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 23:39:23 2016

@author: Shikhar
"""
from sklearn.cross_validation import train_test_split
from pandas import DataFrame, read_csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #this is how I usually import pandas
import sys #only needed to determine Python version number
import matplotlib #only needed to determine Matplotlib version number

def finalTest(clf):
    fileLocation = '../OnlineNewsPopularity/test_data.csv'
    df = pd.read_csv(fileLocation)
    print df.shape

    # Deleting first column
    del df['url']

    # Original data for cross validation
    x_test = df.as_matrix()
    y_test = x_orig_train[:,60]
    x_test = np.delete(x_orig_train,60,1)
    x_test = np.delete(x_orig_train,0,1)
    median = 1400.0
    y_test_binary = y_test < median
    finalScore = clf.score(x_test)
    return finalScore    