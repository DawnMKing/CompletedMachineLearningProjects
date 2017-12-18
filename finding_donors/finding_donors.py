#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 09:07:43 2017

@author: dking
"""

import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames
# Import sklearn.preprocessing.StandardScaler (MinMaxScaler) for normalizing data
from sklearn.preprocessing import MinMaxScaler
# Import train_test_split
from sklearn.model_selection import train_test_split

# Import supplementary visualization code visuals.py
import visuals as vs

# Pretty display for notebooks
#%matplotlib inline

# Load the Census dataset
data = pd.read_csv("census.csv")

# Success - Display the first record
display(data.head(n=1))
# TODO: Total number of records
n_records = data['income'].count()

# TODO: Number of records where individual's income is more than $50,000
n_greater_50k = data[data.income==">50K"].income.count()

# TODO: Number of records where individual's income is at most $50,000
n_at_most_50k = data[data.income=="<=50K"].income.count()

# TODO: Percentage of individuals whose income is more than $50,000
greater_percent = (float(n_greater_50k)/n_records)*100

# Print the results
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent))

# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualize skewed continuous features of original data
#vs.distribution(data)
# Log-transform the skewed features to make more manageable
skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
#vs.distribution(features_raw, transformed = True)


# Initialize a scaler for normalization, then apply it to the features
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

# Show an example of a record with scaling applied
display(features_raw.head(n = 100))


# TODO: One-hot encode the 'features_raw' data using pandas.get_dummies()
columns=['workclass','education_level','marital-status','occupation','relationship','race','sex','native-country']
features = pd.get_dummies(features_raw[columns])

# TODO: Encode the 'income_raw' data to numerical values
income = pd.get_dummies(income_raw)
print(income)
# Print the number of features after one-hot encoding... this gives the column names of each new encoded feature
encoded = list(features.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# Uncomment the following line to see the encoded feature names
#print(encoded)


# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

# TODO: Calculate accuracy

prediction = np.asarray([1]*n_records)
accuracy = float(n_greater_50k)/n_records

# TODO: Calculate F-score using the formula above for 


#prediction = np.asarray([1]*n_records)
val=income['>50K'].value_counts()
TN=val[0];TP=val[1];FN=0;FP=n_records-TP
prec=TP/(TP+FP)
recall=TP/(TP+FN)
beta=0.5
fscore = (1+beta**2)*((prec*recall)/(prec*beta**2 + recall))


# Print the results 
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))
# Check against sklearn calcs
from sklearn.metrics import accuracy_score, fbeta_score
prediction = np.asarray([1]*n_records)

print ( "SKLearn Accuracy: ", accuracy_score(income['>50K'], prediction), "SKLearn F-score", fbeta_score(income['>50K'], prediction, beta) )

# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import accuracy_score, fbeta_score
def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
   # print("S",sample_size)
    # TODO: Fit the learner to the training data using slicing with 'sample_size'
    start = time() # Get start time
   # print("X",y_train[:sample_size]['>50K'])
  
    learner.fit(X_train[:sample_size],y_train[:sample_size]['>50K'])
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = end - start

        
    # TODO: Get the predictions on the test set,
    #       then get predictions on the first 300 training samples
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
   

    # Calculate the total prediction time
    results['pred_time'] = end - start

   # Compute accuracy on the first 300 training samples
    results['acc_train'] = accuracy_score(y_train[:300]['>50K'], predictions_train)
        
    # Compute accuracy on test set
    results['acc_test'] = accuracy_score(y_test['>50K'], predictions_test)
    
    # Compute F-score on the the first 300 training samples. We use previous beta
    results['f_train'] = fbeta_score(y_train[:300]['>50K'], predictions_train, beta)
        
    # Compute F-score on the test set. We use previous beta
    #print("ytest",y_test)
    results['f_test'] = fbeta_score(y_test['>50K'], predictions_test, beta)
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results

# TODO: Import the three supervised learning models from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.svm import SVC
# TODO: Initialize the three models
clf_A = GaussianNB()
clf_B = tree.DecisionTreeClassifier(criterion="entropy",random_state=0)
clf_C = SVC()

# TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
samples_1 = int(len(X_train)/100)
samples_10 = int(len(X_train)/10)
samples_100 = len(X_train)

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
print("FFFF",fscore)        
vs.evaluate(results, accuracy, fscore)