# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 19:21:24 2019

Working example of preprocessing data

@author: FawazM
"""

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values# first parameter is rows, second parameter is columns. : means take all the rows. :-1 means takes all the columns except the last one.
y = dataset.iloc[:, 3].values # all rows, only the 3rd column (0-indexed)

df_X = pd.DataFrame(X) #convert to dataframes for QOL - possible to view in variable explorer
df_y = pd.DataFrame(y) #convert to dataframes for QOL - possible to view in variable explorer

# Missing Data
from sklearn.preprocessing import Imputer

'''
Create Imputer object. Params:
	missing_values is the identifier for what counts as a missing value in your data set
	strategy is how the Imputer will handle missing data
	axis = 0 if taking mean of column, = 1 if taking mean of row. 
'''
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) 

'''
Fit imputer to data. The parameter is the array of data we want to fit.

X[:, 1:3] is all rows, and columns 1 and 2. Important: Upper bound is excluded when indexing arrays. 
'''
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3]) #transform relevant part of X to replace missing data. Now, X will have its missing values replaced by the mean of the columns


# Categorical Data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

## Encode independent by encoding then creating dummy variables
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0]) # fit labelEncoder to first column and transform the data
### must use oneHot because there is no relational order between the variables
oneHotEncoder = OneHotEncoder(categorical_features = [0]) #categorical_features param is what column we are creating dummy variables for
X = oneHotEncoder.fit_transform(X).toarray() #use entire X because we specify columns to encode when creating object

### X will now have the categorical variable column(s) replaced by dummy variable columns. One dummy variable will be created for each option


## Encode depdendent. Don't need to one hot encode because model will assume there's no relational order between dependent variable
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)



# Splitting Data into Test & Training Sets
from sklearn.model_selection import train_test_split
'''
# train_test_split defines all 4 matrices at same time. 
Param 1 X, y is the sequence of arrays of same length. 
Param 2 is the percentage of the data we want to allocate to test set. 0.2 is a good rule of thumb. Max 0.4
Param 3 is train_size, leave blank for 1-test_size
Param 4 is random_state. Psuedo-random number generator state for random sampling. Leave blank unless you want to compare your results to someone else (then use same random sate variable)

'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 


#Feature Scaling

'''
Variables may not have the same scale (they usually won't). 
Since many machine learning models are based on Euclidean distance, 
numbers that have larger scales will dominate numbers that have smaller scales. 
E.g. if age (small-scale) and salary (large-scale) are variables, then most models will see age as being insignificant
since salary will contribute much more to the distance. 
Therefore, we must scale them so that each feature contributes proportionally to the final distance.

Two common methods are standardization and normalization 
'''

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc.X.transform(X_test)

## Don't need to scale dependent variable as its categorical






