"""
Code pertaining to the starting Kaggle competition 
"Titanic - Machine Learning from Disaster"
https://www.kaggle.com/competitions/titanic/overview

My submission: https://www.kaggle.com/code/eleonoraricci/model-screening-hp-opt-and-feature-engineering

"""

import numpy as np 
import pandas as pd 
import os
import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib
import warnings
warnings.filterwarnings('ignore')

# Imports from utility scripts
from imports import *
from data_processing import DataProcessing
from utils import model_score, return_regressors_list, return_classifiers_list 

random_state = 17

# Read in the datasets
dataset_for_modelling_fe = pd.read_csv("data/train.csv")
submission_set_fe = pd.read_csv("data/test.csv")

data_fe = DataProcessing(dataset_for_modelling_fe, submission_set_fe)
data_fe.data_info(data_fe.dataset_for_modelling_fe)

# Report which columns of the data are missing values and require actions
data_fe.check_missing_values()

""" TEST 3: feature engineering  """

""" 3.1 - Train "Age" regressor """

# Rapid model and features list screening for Age prediction

from sklearn import preprocessing

model_names = return_regressors_list()

# Lists of features to drop
tests = [["Name", "Ticket", "Cabin", "PassengerId", "Embarked"], 
         ["Name", "Ticket", "Cabin", "PassengerId", "Embarked", "Fare", "Sex"], 
         ["Name", "Ticket", "Cabin", "PassengerId", "Embarked", "Fare", "Sex", "Parch"]]
results = []

for test in tests:

    age_prediction = dataset_for_modelling_fe[dataset_for_modelling_fe["Age"].isnull()]
    feature_encoder = preprocessing.LabelEncoder()
    age_prediction["Sex"] = feature_encoder.fit_transform(age_prediction["Sex"])
    age_prediction = age_prediction.drop(columns=test + ["Survived", "Age"]) 
    X_submission = age_prediction.values

    data_test = DataProcessing(dataset_for_modelling_fe, submission_set_fe)

    encode_list = ["Sex", "Embarked"]
    data_test.encode(encode_list)
    
    preprocessing_dict = {
        "median" : ["Fare"],
        "zeros" : [],
        "category" : [["Cabin" , "UUU"],
                      ["Embarked" , "U"]],
        "drop" : ["Age"]
        }
    data_test.minimal_preprocessing(preprocessing_dict)

    data_test.drop(test)
    data_test.dataset_for_modelling.drop(columns=["Survived"], inplace = True)
    data_test.get_dummies()

    label = "Age"

    X_train, X_test, y_train, y_test, _ = data_test.get_split(label, test_size=0.25, 
                                                      shuffle=False, random_state=random_state)
    # Run the tests
    start = time.time()

    models_scores = [model_score(name, X_train, X_test, y_train, y_test, X_submission, submission_set, save_to_csv=False) for name in model_names]  

    end = time.time()   
    print("\nTraining time required:", (end-start) , " s") 
    
    models_scores_df_fe = pd.DataFrame(np.array(models_scores), columns=['Model', 
                                                                         'Cross Validation', 
                                                                         'Accuracy'])
    models_scores_df_fe.sort_values(['Cross Validation'], ascending=False, inplace=True)
    models_scores_df_fe = models_scores_df_fe.set_index('Model')
    results.append(models_scores_df_fe)
    
for i, result in enumerate(results):
    result.rename(columns={'Cross Validation': 'Cross Validation - ' + str(i+1)}, inplace=True)
    results[i].drop(columns=["Accuracy"], inplace = True)

res = results[0].copy()
res = res.merge(results[1], how='inner', on='Model')
res = res.merge(results[2], how='inner', on='Model')
res.astype('float').style.background_gradient(axis=0, vmax=0.3, vmin=0., cmap='RdYlGn')

# Predict Age using the best model and features list

dataset_for_modelling = pd.read_csv("data/train.csv")
submission_set = pd.read_csv("data/test.csv")

age_prediction_modl_set = dataset_for_modelling[dataset_for_modelling["Age"].isnull()]
age_prediction_subm_set = submission_set[submission_set["Age"].isnull()]

age_prediction = age_prediction_modl_set.copy()
age_prediction = age_prediction.drop(columns=["Name", "Ticket", "Cabin", "PassengerId", "Embarked", "Fare", "Sex"] + ["Survived", "Age"])
X_age = age_prediction.values

age_prediction_subm = age_prediction_subm_set.copy()
age_prediction_subm = age_prediction_subm.drop(columns=["Name", "Ticket", "Cabin", "PassengerId", "Embarked", "Fare", "Sex"] + ["Age"])
X_age_sumb = age_prediction_subm.values

data_test_age = DataProcessing(dataset_for_modelling, submission_set)

encode_list = ["Sex", "Embarked"]
data_test_age.encode(encode_list)

preprocessing_dict = {
    "median" : ["Fare"],
    "zeros" : [],
    "category" : [["Cabin" , "UUU"],
                  ["Embarked" , "U"]],
    "drop" : ["Age"]
    }
data_test_age.minimal_preprocessing(preprocessing_dict)

data_test_age.drop(["Name", "Ticket", "Cabin", "PassengerId", "Embarked", "Fare", "Sex"])
data_test_age.dataset_for_modelling.drop(columns=["Survived"], inplace = True)
data_test_age.get_dummies()

label = "Age"

X_train, X_test, y_train, y_test, _ = data_test_age.get_split(label, test_size=0.25, 
                                                  shuffle=False, random_state=random_state)
from sklearn.ensemble._gb import GradientBoostingRegressor
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
age = model.predict(X_age)
age_submission = model.predict(X_age_sumb)


# Test with new Age feature
dataset_for_modelling = pd.read_csv("data/train.csv")
submission_set = pd.read_csv("data/test.csv")
data = DataProcessing(dataset_for_modelling, submission_set)

data.check_missing_values()

preprocessing_dict = {
    "median" : ["Fare"],
    "zeros" : [],
    "category" : [["Cabin" , "UUU"],
                  ["Embarked" , "U"]],
    "drop" : []
    }
data.minimal_preprocessing(preprocessing_dict)

# Encode categorical features
encode_list = ["Sex", "Embarked"]
data.encode(encode_list)

# Features to drop
drop_list = ["Name", "Ticket", "Cabin", "PassengerId"]
data.drop(drop_list)

# Fill with predicted Age values
fill_mod = pd.DataFrame(index = dataset_for_modelling.index[dataset_for_modelling["Age"].isnull()], data = age, columns = ["Age"])
data.dataset_for_modelling.fillna(fill_mod, inplace = True)
fill_submission = pd.DataFrame(index = submission_set.index[submission_set["Age"].isnull()], data = age_submission, columns = ["Age"])
data.submission_set.fillna(fill_submission, inplace = True)

data.get_dummies()

label = 'Survived'

X_train, X_test, y_train, y_test, X_submission = data.get_split(label, test_size=0.25, 
                                                  shuffle=False, random_state=random_state)

model_names = return_classifiers_list(random_state, X_train, y_train)

# Run the tests
start = time.time()

models_scores = [model_score(name, X_train, X_test, y_train, y_test, X_submission, submission_set, save_to_csv=False) for name in model_names]  

end = time.time()   
print("\nTraining time required:", (end-start) , " s") 

models_scores_df = pd.DataFrame(np.array(models_scores), columns=['Model', 'Cross Validation', 'Accuracy'])
models_scores_df.sort_values(['Accuracy'], ascending=False, inplace=True)
models_scores_df = models_scores_df.set_index('Model')
# Uncomment below to save output - optional
# models_scores_df.to_csv("results_with_new_age.csv")
models_scores_df.astype('float').style.background_gradient(axis=0, vmin=0.34, vmax=0.9, cmap='RdYlGn')


""" 3.2 - Features manipulation """

dataset_for_modelling = pd.read_csv("data/train.csv")
submission_set = pd.read_csv("data/test.csv")
data = DataProcessing(dataset_for_modelling, submission_set)
preprocessing_dict = {
    "median" : ["Fare", "Age"],
    "zeros" : [],
    "category" : [["Cabin" , "UUU"],
                  ["Embarked" , "U"]],
    "drop" : []
    }
data.minimal_preprocessing(preprocessing_dict)

encode_list = ["Sex", "Embarked"]
data.encode(encode_list)

# Define function to extract titles from passenger names
import re

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
# Create a new feature Title, containing the titles of passenger names
for dataset in [data.dataset_for_modelling, data.submission_set]:
    dataset['Title'] = dataset['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"
for dataset in [data.dataset_for_modelling, data.submission_set]:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 
                                                 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
encode_list = ["Title",]
data.encode(encode_list)

for dataset in [data.dataset_for_modelling, data.submission_set]:
    dataset['CombinedFamily'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['Alone'] = pd.cut(dataset['CombinedFamily'], bins=[0, 1, 10], labels=['One','Many'])

encode_list = ["Alone",]
data.encode(encode_list)

## create Range for Age and Fare features
for dataset in [data.dataset_for_modelling, data.submission_set]:
    dataset['AgeRange'] = pd.cut(dataset['Age'], bins=[0, 15, 18, 29, 35, 120], labels=['Children','Teenage','YoungAdult', 'Adult', 'Elder'])

for dataset in [data.dataset_for_modelling, data.submission_set]:
    dataset['FareRange'] = pd.cut(dataset['Fare'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 300], 
                                  labels=[ "10", "20", "30", "40", "50", "60", "70", "80", "90", "100", "150", "300"])

encode_list = ["AgeRange", "FareRange"]
data.encode(encode_list)

# Pearson correlation coefficient between features and survival rate
data.dataset_for_modelling.corr().sort_values(by='Survived', ascending=False).style.background_gradient(axis=0, vmax = 0.6, cmap='RdYlGn')

# Test 1: more features, models with optimal hyperpatamers
# Features to drop - test 1
drop_list = ["Name", "Ticket", "Cabin", "PassengerId"]
data.drop(drop_list)
data.get_dummies()

label = 'Survived'

X_train, X_test, y_train, y_test, X_submission = data.get_split(label, test_size=0.25, 
                                                  shuffle=False, random_state=random_state)

estimator = RandomForestClassifier(random_state=random_state, 
                                   ccp_alpha = 0.010771, 
                                   criterion ='entropy',
                                   max_features = 'sqrt', 
                                   n_estimators = 152, 
                                   bootstrap = False).fit(X_train, y_train)

estimator_kn = LogisticRegression(solver = 'newton-cg', C = 84).fit(X_train, y_train)
             
estimators = [('rf', estimator),
            ('kn', estimator_kn)]

model_names = [('MLPClassifier', '(early_stopping = True, \
                                   activation = \'logistic\', \
                                   alpha = 0.0001, \
                                   batch_size = 20, \
                                   hidden_layer_sizes = 10, \
                                   random_state = 17, \
                                   solver = \'lbfgs\', \
                                   max_iter = 1000 )'),
               ('HistGradientBoostingClassifier','(early_stopping = True, \
                                                   l2_regularization = 0.5, \
                                                   random_state = 17, \
                                                   min_samples_leaf = 27, \
                                                   max_iter = 41, \
                                                   max_leaf_nodes = 83)'),
              ('StackingClassifier', f'(estimators={estimators})')]
            
models_scores = [model_score(name, X_train, X_test, y_train, y_test, X_submission, submission_set, save_to_csv=True) for name in model_names]  
print(models_scores)

# Test 2: less features, models with optimal hyperpatamers
# Features to drop - test 2
# drop_list = ["Name", "Ticket", "Cabin", "PassengerId"] Already dropped
drop_list = ["CombinedFamily", "Age", "Parch", "SibSp", "Title"]
data.drop(drop_list)
data.get_dummies()

label = 'Survived'

X_train, X_test, y_train, y_test, X_submission = data.get_split(label, test_size=0.25, 
                                                  shuffle=False, random_state=random_state)

estimator = RandomForestClassifier(random_state=random_state, 
                                   ccp_alpha = 0.010771, 
                                   criterion ='entropy',
                                   max_features = 'sqrt', 
                                   n_estimators = 152, 
                                   bootstrap = False).fit(X_train, y_train)

estimator_kn = LogisticRegression(solver = 'newton-cg', C = 84).fit(X_train, y_train)
             
estimators = [('rf', estimator),
            ('kn', estimator_kn)]

model_names = [('MLPClassifier', '(early_stopping = True, \
                                   activation = \'logistic\', \
                                   alpha = 0.0001, \
                                   batch_size = 20, \
                                   hidden_layer_sizes = 10, \
                                   random_state = 17, \
                                   solver = \'lbfgs\', \
                                   max_iter = 1000 )'),
               ('HistGradientBoostingClassifier','(early_stopping = True, \
                                                   l2_regularization = 0.5, \
                                                   random_state = 17, \
                                                   min_samples_leaf = 27, \
                                                   max_iter = 41, \
                                                   max_leaf_nodes = 83)'),
              ('StackingClassifier', f'(estimators={estimators})')]
            
models_scores = [model_score(name, X_train, X_test, y_train, y_test, X_submission, submission_set, save_to_csv=True) for name in model_names]  
print(models_scores)