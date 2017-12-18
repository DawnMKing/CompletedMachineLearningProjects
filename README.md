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


## Customer Segments


## Smartcab
