#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:59:13 2018

@author: patrickorourke
"""

# Assignment for the dataset "Titanic - Functions"

'''
classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
	AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()]
'''

import pandas as pd  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score

file_train = '/Users/patrickorourke/Documents/Titanic/train.csv'
file_test = '/Users/patrickorourke/Documents/Titanic/test.csv'
file_compare = '/Users/patrickorourke/Documents/Titanic/gender_submission.csv'

enc = preprocessing.LabelEncoder()
model = KNeighborsClassifier(n_neighbors=3, algorithm='auto', )

pd.options.mode.chained_assignment = None

def import_data(file):
    data = pd.read_csv(file)
    return data

def missing(data):
    # Check for missing data
    missing = data.isnull().sum()
    return missing

# For "Age", replace missing values with the mean age of the column
def missing_age(data):
    data["Age"] = data["Age"].fillna(data["Age"].mean())
    return data

# For "Embarked", replace 2 missing values with the most common value   
def missing_embarked(data):
    pd.options.mode.chained_assignment = None
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode().values[0])
    return data
 
def missing_cabin(data):
    # Select only necessary columns
    data = data[['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']]
    # Fill all blank entries with 0
    data = data.fillna(0)
    # Isolate only the 204 rows in the FULL data where "Cabin" HAS values
    Cabin_HAS = data[data['Cabin'] != 0]
    # Isolate only the 204 rows in the train data where "Cabin" HAS values
    trainX = Cabin_HAS[['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Embarked']]
    # Isolate only the 204 rows in the train data where "Cabin" HAS values
    trainY = Cabin_HAS[['Cabin']]
    # Fit a KNN Classifier on the traning X and Y data
    model.fit(trainX, trainY.values.ravel()) 
    # Isolate only the 687 rows in the FULL data where "Cabin" has NO values
    Cabin_HAS_NO = data[data['Cabin'] == 0]
    # Isolate only the 687 rows in the test data where "Cabin" has NO values
    testX=  Cabin_HAS_NO[['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Embarked']]
    # Isolate only the 687 rows in the train data where "Cabin" has NO values
    # testY = Cabin_HAS_NO[['Cabin']]
    # Using the fitted KNN modell, predicted the missing values the Pandas' Series, "testY" 
    predY = model.predict(testX)
    cabins = list(trainY.values.flatten()) + list(predY.flatten())
    data['Cabin'] = cabins
    return data
         
# Encode "Name" column based on surnames
# NOTE - CANNOT use "len()" when iterating over a Pandas column
def encode_name(data):
    for i in range(data["Name"].count()):
        st = data["Name"][i].split(",")
        data["Name"][i] = st[0]
    enc.fit(data["Name"])
    data["Name"]  = enc.transform(data["Name"]) 
    return data
 
# Encode "Sex" column to [0,1]
def encode_sex(data):
    data["Sex"] = data['Sex'].replace({'female':0,'male':1})
    return data
    
# Encode "Ticket" column
def encode_ticket(data):
    enc.fit(data["Ticket"])
    data["Ticket"]  = enc.transform(data["Ticket"]) 
    return data
    
# Encode "Embarked" column
def encode_embarked(data):
    enc.fit(data["Embarked"])
    data["Embarked"]  = enc.transform(data["Embarked"])
    return data
    
# Encode "Cabin" column
def encode_cabin(data):
    enc.fit(data["Cabin"])
    data["Cabin"]  = enc.transform(data["Cabin"])
    return data
      
data_train = import_data(file_train)
#train = missing(train)
train = missing_age(data_train)
train = missing_embarked(train)
train = encode_name(train)
train = encode_sex(train)
train = encode_ticket(train)
train = encode_embarked(train)   
train = missing_cabin(train) 
train = encode_cabin(train)

data_test = import_data(file_test)
#train = missing(train)
test = missing_age(data_test)
test = missing_embarked(test)
test = encode_name(test)
test = encode_sex(test)
test = encode_ticket(test)
test = encode_embarked(test)   
test = missing_cabin(test) 
test = encode_cabin(test) 

data_compare = import_data(file_compare)


trainY = data_train['Survived']
trainX = train[['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']]
testX = test[['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']]
#testX = data2.iloc[:,2:11]
nav = GaussianNB()
nav.fit(trainX, trainY)
PredY_nav = nav.predict(testX)

rndFor = DecisionTreeClassifier(random_state=0)
rndFor.fit(trainX, trainY)
PredY_rndFor = rndFor.predict(testX)

svM = svm.SVC(gamma='auto')
svM.fit(trainX, trainY)
PredY_SVM = svM.predict(testX)

# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(trainX, trainY)
PredY_RF = rf.predict(testX)

# Create and fit an AdaBoosted decision tree
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME",n_estimators=200)
ada.fit(trainX, trainY)
PredY_ada = ada.predict(testX)


parameters = {
    "loss":["deviance"],
    "learning_rate": [0.01],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    #"max_depth":[3,5,8],
    #"max_features":["log2","sqrt"],
    #"criterion": ["friedman_mse",  "mae"],
    #"subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    #"n_estimators":[10]
    }
# GridSearchCV scores different combinations of hyperparameters and finds the combination which gives 
#the best results
grad = GridSearchCV(GradientBoostingClassifier(), parameters, cv=10, n_jobs=-1)
grad.fit(trainX, trainY)
PredY_grad = grad.predict(testX)

# Compare testX['PredY'] amd data_compare['Survived']
accu_nav = accuracy_score(PredY_nav, data_compare['Survived'])
accu_svm = accuracy_score(PredY_SVM, data_compare['Survived'])
accu_rndFor = accuracy_score(PredY_rndFor, data_compare['Survived'])
accu_RF = accuracy_score(PredY_RF, data_compare['Survived'])
accu_ada = accuracy_score(PredY_ada, data_compare['Survived'])
accu_grad = accuracy_score(PredY_grad, data_compare['Survived'])

print("Accuracy score with Naive Bayes: ", accu_nav)
print("Accuracy score with SVM: ", accu_svm)
print("Accuracy score with Decision Tree Classifiert: ", accu_rndFor)
print("Accuracy score with Random Forest Classifier: ", accu_RF)
print("Accuracy score with Two-Class AdaBoost Classifier: ", accu_ada)
print("Accuracy score with Gradient-Boosting Classifier: ", accu_grad * 100)

submission = pd.DataFrame({
        "PassengerId": data_test["PassengerId"],
        "Survived": PredY_grad
    })
submission.to_csv('/Users/patrickorourke/Documents/Titanic_Solution1.csv', index=False)
