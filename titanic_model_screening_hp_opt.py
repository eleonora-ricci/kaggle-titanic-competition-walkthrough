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
dataset_for_modelling = pd.read_csv("data/train.csv")
submission_set = pd.read_csv("data/test.csv")

data = DataProcessing(dataset_for_modelling, submission_set)
data.data_info(data.dataset_for_modelling)

# Report which columns of the data are missing values and require actions
data.check_missing_values()

""" TEST 1: Model screening  """

# Decide preprocessing actions. Four options are implemented in the DataProcessing class:
# 1. fill with the median
# 2. fill with zeros
# 3. fill with a categorical placeholder
# 4. drop rows with missing or Nan values
# For each option (dictionary keys), the columns to treat in this way 
# are supplied as a list. 
# In the case of categorical placeholders each list element must be itself 
# a list containing Column Name, Value 
preprocessing_dict = {
    "median" : ["Age", "Fare"],
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
data.get_dummies()

label = 'Survived'

X_train, X_test, y_train, y_test, X_submission = data.get_split(label, test_size=0.25, 
                                                  shuffle=False, random_state=random_state)

model_names = return_classifiers_list(random_state, X_train, y_train)

start = time.time()

models_scores = [model_score(name, X_train, X_test, y_train, y_test, X_submission, submission_set, save_to_csv=False) for name in model_names]  

end = time.time()   
print("\nTraining time required:", (end-start) , " s") 

#Results
models_scores_df = pd.DataFrame(np.array(models_scores), columns=['Model', 'Cross Validation', 'Accuracy'])
models_scores_df.sort_values(['Accuracy'], ascending=False, inplace=True)
models_scores_df = models_scores_df.set_index('Model')
models_scores_df.replace("nan", 0.77009, inplace=True) #average values over 4 cross-validation folds for CategoricalNB, as the 5th resulted in nan
# Value in the the following list were obtained by submitting the predicitons for evaluation
models_scores_df.loc[:,'Submission'] = pd.Series([0.77990,0.67794,0.76555,0.76076,0.74641,0.74641,0.79186,0.76555,0.76555,0.76555,0.76555,0.76555,0.76555,0.76315,0.77272,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], index=models_scores_df.index)
models_scores_df.astype('float').style.background_gradient(axis=0, vmin=0.34, vmax=0.9, cmap='RdYlGn')


""" TEST 2: Hyperparameter tuning  """

from skopt import BayesSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

test_models_names = ["LogisticRegressionCV", "MLPClassifier", "HistGradientBoostingClassifier", "RandomForestClassifier"]

def explore_model_params(test_models_names):

    for model_name in test_models_names:
        model = None
        name = model_name
        if type(model_name) is tuple:
            model = eval(model_name[0] + model_name[1])
            name = str(model).split('(')[0]
        else:
            try:
                model = eval(model_name + '(random_state=17)')
            except:
                model = eval(model_name + '()')

        print("Hyperparameters for estimator: ", name)
        hp = model.get_params()
        print(hp)
        print("\n")
        
explore_model_params(test_models_names)

test_models_names = ["LogisticRegressionCV", "MLPClassifier", "HistGradientBoostingClassifier", "RandomForestClassifier"]

@ignore_warnings(category=ConvergenceWarning)
def optimize_hp(test_models_names, X_train, X_test, y_train, y_test, X_submission, submission_set):
    start = time.time()
    for model_name in test_models_names:
        model = None
        name = model_name
        if type(model_name) is tuple:
            model = eval(model_name[0] + model_name[1])
            name = str(model).split('(')[0]        
        else:
            try:
                model = eval(model_name + '(random_state=17)')
            except:
                model = eval(model_name + '()')
            
        params = dict()
        if isinstance(model, LogisticRegressionCV):            
            params['Cs'] = (50, 100)
            params['solver'] = ['lbfgs', 'liblinear', 'newton-cg']
            params['max_iter'] = [500]
            params['random_state'] = [17]

        if isinstance(model, MLPClassifier):
            params['activation'] = ['logistic', 'tanh', 'relu']
            params['solver'] = ['lbfgs', 'sgd', 'adam']
            params['alpha'] = [0.0001, 0.001]
            params['batch_size'] = (20, 100)
            params['early_stopping'] = [True]
            params['hidden_layer_sizes'] = (10, 100)
            params['max_iter'] = [1000]
            params['random_state'] = [17]

        if isinstance(model, HistGradientBoostingClassifier):
            params['max_iter'] = (20, 200)
            params['max_leaf_nodes'] = (10, 100)
            params['min_samples_leaf'] = (5, 50)
            params['random_state'] = [17]
            params['early_stopping'] = [True]
            params['l2_regularization'] = [0.0, 0.5]
        
        if isinstance(model, RandomForestClassifier):
            params['random_state'] = [17]
            params['n_estimators'] = (20, 200)
            params['criterion'] = ["gini", "entropy"]
            params['max_features'] = ["sqrt", "log2", None]
            params['bootstrap'] = [True, False]
            params['ccp_alpha'] = [0.0, 0.5]

        if isinstance(model, StackingClassifier):
            params['random_state'] = [17]
            params['C'] = (0.1, 10)
            params['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
            params['gamma'] = ['scale', 'auto']
            params['probability'] = [True, False]
            params['cache_size'] = [500]

        # define evaluation
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=17)
        # define the search
        search = BayesSearchCV(estimator=model, search_spaces=params, n_jobs=-1, cv=cv,  n_iter=50)
        # perform the search
        search.fit(X = np.concatenate((X_train,X_test)), y = np.concatenate((y_train,y_test)))
        # report the best result
        print("Best results for estimator: ", name)
        print("Best score on cross-validation: ",search.best_score_)
        print("Best parameters: ", search.best_params_)
        print("Results on validation set: ",search.best_estimator_.score(X_test, y_test).round(5))

        # Calculate model scores
        y_final = search.best_estimator_.predict(X_submission)
        y_final = y_final.reshape(y_final.shape[0],)
        final_df = pd.DataFrame({'PassengerId': submission_set['PassengerId'], 'Survived': y_final})

        filename = str(model).split('(')[0]
        final_df.to_csv(f'{filename}_hp_opt.csv', index=False)
        print("\n")
        end = time.time()   
        print("HP optimization time required for ", name, ":", (end-start) , " s") 

optimize_hp(test_models_names, X_train, X_test, y_train, y_test, X_submission, submission_set)