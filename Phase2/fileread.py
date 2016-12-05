from sklearn.cross_validation import train_test_split
from pandas import DataFrame, read_csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #this is how I usually import pandas
import sys #only needed to determine Python version number
import matplotlib #only needed to determine Matplotlib version number
fileLocation = '../OnlineNewsPopularity/training_data.csv'
df = pd.read_csv(fileLocation)
print df.shape

# Deleting first column
del df['url']

# Original data for cross validation
x_orig_train = df.as_matrix()
y_orig_train = x_orig_train[:,60]
x_orig_train = np.delete(x_orig_train,60,1)
x_orig_train = np.delete(x_orig_train,0,1)

# Dividing Train:Test::80:20 into
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

#x_train, x_test, y_train, y_test = train_test_split(x_orig_train, y_orig_train, test_size=0.8)


# Train data
x_train = train.as_matrix()
y_train = x_train[:,60]
x_train = np.delete(x_train,60,1)
x_train = np.delete(x_train,0,1)

# Test data
x_test = test.as_matrix()
y_test = x_test[:,60]
x_test = np.delete(x_test,60,1)
x_test = np.delete(x_test,0,1)


median = np.median(y_train)

y_train_binary = y_train < median
y_test_binary = y_test < median
y_orig_train_binary = y_orig_train < median

#Reading the final testset file
fileLocation = '../OnlineNewsPopularity/test_data.csv'
df = pd.read_csv(fileLocation)
print df.shape

# Deleting first column
del df['url']

x_final_test = df.as_matrix()
y_final_test = x_final_test[:,60]
x_final_test = np.delete(x_final_test,60,1)
x_final_test = np.delete(x_final_test,0,1)
y_final_test_binary = y_final_test < median

np.savetxt('../Data/x_train.txt',x_train, fmt = '%g')
np.savetxt('../Data/y_train.txt',y_train, fmt = '%g')
np.savetxt('../Data/x_test.txt',x_test, fmt = '%g')
np.savetxt('../Data/y_test.txt',y_test, fmt = '%g')
np.savetxt('../Data/x_orig_train.txt',x_orig_train, fmt = '%g')
np.savetxt('../Data/y_orig_train.txt',y_orig_train, fmt = '%g')
np.savetxt('../Data/y_train_binary.txt',y_train_binary, fmt = '%g')
np.savetxt('../Data/y_test_binary.txt',y_test_binary, fmt = '%g')
np.savetxt('../Data/x_final_test.txt',x_final_test, fmt = '%g')
np.savetxt('../Data/y_final_test_binary.txt',y_final_test_binary, fmt = '%g')


from sklearn import preprocessing
normalizer = preprocessing.Normalizer().fit(x_train)
x_train_new = normalizer.transform(x_train)
x_test_new = normalizer.transform(x_test)
x_train = x_train_new
x_test = x_test_new
# feature selection
# 11
# features_to_ignore = [1, 2, 4, 5, 6, 7, 8, 9 ,10 ,11 ,12 ,13 ,14 ,15 ,16 ,17 ,18 , 19 , 21, 25, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
# 21
# features_to_ignore = [4,5,6,11,13,14,15,16,17,18,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58]
# 31
# features_to_ignore=[4,    13,    15 ,  16 ,   17 ,   32 ,  33  ,  34 ,   35   , 36 ,   37 ,   38,    39,    40,  43,    44,    45  ,  46  ,  47  ,  48   , 49 ,   50,   51   , 53,    55 ,   56  ,  57 ,   58]
# 33

# features_to_ignore=[  4  ,  13  , 16  ,  17  ,  32   , 33  ,  34    ,35,    36 ,  37 ,   39 ,   40,    43,    44 ,   45,    46 ,   47 ,   48 ,   49  ,  50,    51 ,   53,   55 ,   57, 58]
# 41
# features_to_ignore=[3,   33,    36,    39,    40,    45,    46,    47,    51]
# features_to_ignore =[]
# print len(features_to_ignore)
# x_train = np.delete(x_train,features_to_ignore,1)
# x_test = np.delete(x_test,features_to_ignore,1)
print x_train.shape , x_test.shape


# print y_train
# print x_train[:,15]
print 'File Read'
