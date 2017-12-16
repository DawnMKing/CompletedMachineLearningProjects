# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 00:12:09 2017

@author: Dawn
"""

# Import libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import ShuffleSplit, train_test_split
from sklearn.metrics import r2_score as r2, make_scorer
from sklearn.tree import DecisionTreeRegressor as DT
from sklearn.grid_search import  GridSearchCV 
# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
#%matplotlib inline

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
# Success
print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)
#print features
#print prices
#Data preprocessing

for col in features.columns:

    fig, ax = plt.subplots()
    fit = np.polyfit(features [col], prices, deg=1) # Using a linear fit to compute the trendline
    ax.scatter(features [col],  prices)
    plt.plot(features [col], prices, 'o', color='black')
    ax.plot(features[col], fit[0] * features[col] + fit[1], color='blue', linewidth=3) # This plots a trendline with the regression parameters computed earlier. We should plot this after the dots or it will be covered by the dots themselves
    plt.title('PRICES vs  '+ str(col)) # title here
    plt.xlabel(col) # features here
    plt.ylabel('PRICES') # label here
#statistics of the target data

minimum_price = np.min(prices)
maximum_price = np.max(prices)
mean_price = np.mean(prices)
median_price = np.median(prices)
std_price = np.std(prices)

# Show the calculated statistics
print "Statistics for Boston housing dataset:\n"
print "Minimum price: ${:,.2f}".format(minimum_price)
print "Maximum price: ${:,.2f}".format(maximum_price)
print "Mean price: ${:,.2f}".format(mean_price)
print "Median price ${:,.2f}".format(median_price)
print "Standard deviation of prices: ${:,.2f}".format(std_price)

#Define the performance metric using the r2 score importes sklearn.metrics import r2_score 
def performance_metric(labels, prediction):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # Calculate the performance score between 'y_true' and 'y_predict'
    score = r2(labels, prediction)
     # Return the score
    return score
 #Calculate the performance of this model, import 
#Testing if functionworks on a test set#####################################################################
#score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
#print "Model has a coefficient of determination, R^2, of {:.3f}.".format(score)
##############################################################################   
# Shuffle and split the data into training and testing subsets from sklearn.cross_validation--new version wil use model_selection
X_train, X_test, y_train, y_test = train_test_split(features['RM'], prices, test_size = 0.20, random_state=1)

# from visuals.py Produce learning curves for varying training set sizes and maximum depths
vs.ModelLearning(features, prices)
#from visual.py produce complexity plot of maximum depth v. score of traing and testing curves
vs.ModelComplexity(features, prices)
#==============================================================================
#Implement code to fit model... this case will be the decsion tree algoritm.
def fit_model(features, prices):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data to define how to split 
    #and how many test runs of each split on data 
    cv_sets = ShuffleSplit(features.shape[0], n_iter=10, test_size=0.2, random_state=1)

    # TODO: Create a decision tree regressor object
    regressor = DT()

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth': (1,2,3,4,5,6,7,8,9,10)}

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric,greater_is_better=True)

    # Create the grid search object--RandomizedSearchCV is another option
    grid = GridSearchCV(regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(features, prices)

    # Return the optimal model after fitting the data
    return grid.best_estimator_
    
    
# Fit the training data to the model using grid search
reg = fit_model(features, prices)

# Produce the value for 'max_depth'
print "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth'])
# Produce a matrix for client data
client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print "Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price)
#using visuals.py code predict ten different model fits of a particular clients data
#client_data = [[5, 17, 15]], # Client 1
               #[4, 32, 22], # Client 2
               #[8, 3, 12]]  # Client 3

vs.PredictTrials(features, prices, fit_model, client_data)