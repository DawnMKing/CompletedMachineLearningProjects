# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# Import supplementary visualization code visuals.py
import visuals as vs

# Load the Census dataset
data = pd.read_csv("census.csv")

#Display the n records
display(data.head(n=2))
#display(data['income'])

#Total number of records
n_records = data['income'].count()

#Number of records where individual's income is more than $50,000
n_greater_50k = data[data.income==">50K"].income.count()

#Number of records where individual's income is at most $50,000
n_at_most_50k = data[data.income=="<=50K"].income.count()

#Percentage of individuals whose income is more than $50,000
greater_percent = (float(n_greater_50k)/n_records)*100

# Print the results
print "Total number of records: {}".format(n_records)
print "Individuals making more than $50,000: {}".format(n_greater_50k)
print "Individuals making at most $50,000: {}".format(n_at_most_50k)
print "Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent)
# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)
#print(income_raw)
#print(features_raw)
# Visualize skewed continuous features of original data
vs.distribution(data)
# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
vs.distribution(features_raw, transformed = True)



# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

# Show an example of a record with scaling applied
display(features_raw.head(n = 1))
# One-hot encode the 'features_raw' data using pandas.get_dummies()
features = pd.get_dummies(features_raw)
#Encode the 'income_raw' data to numerical values
income = pd.get_dummies(income_raw)
# Print the number of features after one-hot encoding
encoded = list(features.columns)
print "{} total features after one-hot encoding.".format(len(encoded))

# Uncomment the following line to see the encoded feature names
#print encoded


# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0)
# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])

# Before ML accuracy and f-beta calc
accuracy = float(n_greater_50k)/n_records

val=income['>50K'].value_counts()
TN=val[0];TP=val[1];FN=0;FP=n_records-TP
prec=TP/(TP+FP)

recall=TP/(TP+FN)
beta=0.5
fscore = (1+beta**2)*((prec*recall)/(prec*beta**2 + recall))


print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))

# Import two metrics from sklearn - fbeta_score and accuracy_score
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
    #Fit the learner to the training data using slicing with 'sample_size'
    start = time() # Get start time
    learner.fit(X_train[:sample_size],y_train[:sample_size]['>50K'])
    end = time() # Get end time
    
    #Calculate the training time
    results['train_time'] = end - start

        
    #Get the predictions on the test set,
    #then get predictions on the first 300 training samples
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

# Import the three supervised learning models from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.svm import SVC
# Initialize the three models
clf_A = GaussianNB()
clf_B = tree.DecisionTreeClassifier(random_state=0)
clf_C = SVC(random_state=0)

# Calculate the number of samples for 1%, 10%, and 100% of the training data
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
vs.evaluate(results, accuracy, fscore)
for i in results.items():
    print i[0]
    display(pd.DataFrame(i[1]).round(4).rename(columns={0:'1%', 1:'10%', 2:'100%'}))

#FINE Tune achosen algorithm
# Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from sklearn import cross_validation
import pandas as pd

#  Initialize the classifier
clf = tree.DecisionTreeClassifier(criterion="gini",random_state=0)
#clf = SVC(random_state=0)
# Create the parameters list you wish to tune
parameters = {'max_depth':[2,3,4,5,6,8], 'splitter':['best','random'] ,'min_samples_split':[0.001,0.002,0.003,0.005,0.006,.01,.02],'min_samples_leaf':[0.001,0.002,0.003,0.005,0.006,.01,.02] }
#parameters = {'C':[1,6],'kernel':['linear','poly','rbf','sigmoid'],'degree':[1,6]}
#  Make an fbeta_score scoring object
scorer = make_scorer(fbeta_score, beta=0.5)
#cv = cross_validation.StratifiedShuffleSplit(y_train['>50K'], 2, random_state = 42)
# Perform grid search on the classifier using 'scorer' as the scoring method
grid_obj = GridSearchCV(clf, parameters,scoring=scorer)

# Fit the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(X_train, y_train['>50K'])

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train['>50K'])).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print( "Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test['>50K'], predictions)))
print( "F-score on testing data: {:.4f}".format(fbeta_score(y_test['>50K'], predictions, beta = 0.5)))
print( "\nOptimized Model\n------")
print( "Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test['>50K'], best_predictions)))
print( "Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test['>50K'], best_predictions, beta = 0.5)))

df = pd.DataFrame(grid_fit.grid_scores_).sort_values('mean_validation_score',ascending=False).tail()

print(df)

# Import a supervised learning model that has 'feature_importances_'
from sklearn.tree import DecisionTreeClassifier
# Train the supervised model on the training set 
model = tree.DecisionTreeClassifier(criterion="gini",random_state=0)
model.fit(X_train, y_train['>50K'])
# Extract the feature importances
importances = model.feature_importances_

# Plot
vs.feature_plot(importances, X_train, y_train['>50K'])     

# Import functionality for cloning a model
from sklearn.base import clone

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the "best" model found from grid search earlier
clf = (clone(best_clf)).fit(X_train_reduced, y_train['>50K'])

# Make new predictions
reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print("Final Model trained on full data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test['>50K'], best_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test['>50K'], best_predictions, beta = 0.5)))
print("\nFinal Model trained on reduced data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test['>50K'], reduced_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test['>50K'], reduced_predictions, beta = 0.5)))