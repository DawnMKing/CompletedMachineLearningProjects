# CompletedMachineLearningProjects
Machine Learning Projects for the completion of Udacity's Machine Learning Nanodegree. Each project contians an html file for easy viewing of the project, the jupyter ipy notebook used to edit the project for submission, and Python code. Description of each follows:

## Predicting Boston Housing Prices
Purpose is to predict boston housing prices based on 490 data points and 3 features. 

**Features**
1.  `RM`: average number of rooms per dwelling
2. `LSTAT`: percentage of population considered lower status
3. `PTRATIO`: pupil-student ratio by town

**Target Variable**
4. `MEDV`: median value of owner-occupied homes

![alt text](https://github.com/DawnMKing/CompletedMachineLearningProjects/blob/master/boston_housing/ScatterPlots.PNG)

### Key Highlights

Supervised Learning: Decision Tree Regressor

Implemented GridSearchCV with maxdepth parameter, r2 scoring, and ShuffleSplit (number of splits=10) to determine optimal model.

#### Visualizations

*Learning Curves* show high bias when tree depth is too low, and high variance when tree depth is to high.
![alt text](https://github.com/DawnMKing/CompletedMachineLearningProjects/blob/master/boston_housing/LearningCurves.png)

*Complexity Plot* shows how the training and validation set score as a function of tree depth.
![alt text](https://github.com/DawnMKing/CompletedMachineLearningProjects/blob/master/boston_housing/Complexity.png)

#### Outcome

Optimal depth was determined to be at 4. Reasonable predictions were made on: 

Client 1 (#rooms=5,%poverty level=17,student-teacher ratio=15)

Predicted selling price for Client 1's home: $408,800.00


Client 2 (#rooms=4,%poverty level=32,student-teacher ratio=22)

Predicted selling price for Client 2's home: $231,253.45


Client 3 (#rooms=8,%poverty level=3,student-teacher ratio=12)

Predicted selling price for Client 3's home: $938,053.85



## Finding Donors

This project employs several supervised algorithms to accurately model individuals' income using data collected from the 1994 U.S. Census. After analysis of initial implementation of several different algorithms, one is chosen that is further optimized. This **goal** of this project is to construct a model that accurately predicts whether an individual makes more than $50,000, with theunderstanding that an individual's income can help a non-profit better understand how large of a donation to request, or whether or not they should reach out to begin with. The project will infer a person's individual income this value from other publically available features.

### Exploring the Data/ Preprocessing

Features: age, workclass, education_level, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country

Census Records:

Total number of records: 45222

Individuals making more than $50,000: 11208

Individuals making at most $50,000: 34014

Percentage of individuals making more than $50,000: 24.78%

**Features with Skewed Data Distribution**
![alt text](https://github.com/DawnMKing/CompletedMachineLearningProjects/blob/master/finding_donors/Skewed.png)

**Lorgaritmic Transformation**
![alt text](https://github.com/DawnMKing/CompletedMachineLearningProjects/blob/master/finding_donors/Log.png)

All numerical data was normalized using MinMaxScaler, and categorical features where transformed using one-hot encoding via pandas get_dummies function.

### Exporing Supervised Models

**Guassian Naive Bayes**(red), **Support Vector Classifier**(blue), **Decision Tree Classifier**(green)

![alt text](https://github.com/DawnMKing/CompletedMachineLearningProjects/blob/master/finding_donors/AllLearners.png)

![alt text](https://github.com/DawnMKing/CompletedMachineLearningProjects/blob/master/finding_donors/features.png)
## Customer Segments


## Smartcab
