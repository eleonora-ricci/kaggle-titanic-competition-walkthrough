import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from imports import * 

def model_score(model_name, X_train, X_test, y_train, y_test, X_submission, submission_df, save_to_csv=True):
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

    print("Training: ", name)
    if name == 'ClassifierChain' or name == 'MultiOutputClassifier':
        return model_scores_special(model, X_train, X_test, y_train, y_test, X_submission, submission_df, save_to_csv=save_to_csv)
    
    model.fit(X_train, y_train)
    
    y_final = model.predict(X_submission)
    
    if save_to_csv:
        final_df = pd.DataFrame({'PassengerId': submission_df['PassengerId'], 'Survived': y_final})
        final_df.to_csv(f'{name}.csv', index=False)
    
    model_score = model.score(X_test, y_test).round(5)
    cv_test = cross_val_score(model, np.concatenate((X_train, X_test)), np.concatenate((y_train,y_test)), cv=5).mean().round(5)

    return [name, cv_test, model_score]
    
    
def model_scores_special(model, X_train, X_test, y_train, y_test, X_submission, submission_df, save_to_csv=True):
    
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    model.fit(X_train, y_train)
    
    # Calculate model scores
    y_final = model.predict(X_submission)
    y_final = np.concatenate(y_final).astype(int)
    y_final = y_final.reshape(y_final.shape[0],)

    model_score = model.score(X_test, y_test).round(5)    
    cv_test = cross_val_score(model, np.concatenate((X_train, X_test)), np.concatenate((y_train,y_test)), cv=5).mean().round(5)
    name = str(model).split('(')[0]

    if save_to_csv:
        final_df = pd.DataFrame({'PassengerId': submission_df['PassengerId'], 'Survived': y_final})
        final_df.to_csv(f'{name}.csv', index=False)
    
    return [name, cv_test, model_score]

def return_classifiers_list(random_state, X_train, y_train):
    solver = "liblinear"
    estimator = RandomForestClassifier(random_state=random_state).fit(X_train, y_train)
    estimator_kn = LogisticRegression(solver = solver).fit(X_train, y_train)
    estimators = [('rf', estimator),
                  ('kn', estimator_kn)]

    model_names = [
        'AdaBoostClassifier',
        'BaggingClassifier',
        'BernoulliNB',
        'CalibratedClassifierCV',
        ('CategoricalNB', '(min_categories=75)'),
        ('ClassifierChain', f'(base_estimator={estimator_kn})'),    
        'ComplementNB',
        'DecisionTreeClassifier',
        ('DummyClassifier', '(strategy="most_frequent", random_state=17)'),
        'ExtraTreeClassifier',
        'ExtraTreesClassifier',
        'GaussianNB',
        'GaussianProcessClassifier',
        'GradientBoostingClassifier',
        'HistGradientBoostingClassifier',
        'KNeighborsClassifier',
        'LabelPropagation',
        'LabelSpreading',
        'LinearDiscriminantAnalysis',
        'LinearSVC',
        ('LogisticRegression', '(solver="liblinear")'),
        ('LogisticRegressionCV', '(solver="liblinear")'),
        'MLPClassifier',
        ('MultiOutputClassifier', f'(estimator={estimator})'),
        'MultinomialNB',
        'NearestCentroid',
        'NuSVC',
        ('OneVsOneClassifier', f'(estimator={estimator})'),
        ('OneVsRestClassifier', f'(estimator={estimator})'),
        ('OutputCodeClassifier', f'(estimator={estimator})'),
        'PassiveAggressiveClassifier',
        ('Perceptron', '(shuffle=False, random_state=17)'),
        'QuadraticDiscriminantAnalysis',
        ('RadiusNeighborsClassifier', '(radius=50, outlier_label=0)'),
        'RandomForestClassifier',
        'RidgeClassifier',
        'RidgeClassifierCV',
        'SGDClassifier',
        'SVC',
        ('StackingClassifier', f'(estimators={estimators})'),
        ('VotingClassifier', f'(estimators={estimators})'),
        ]

    return model_names

def return_regressors_list():
    model_names = [
        'ARDRegression',
        'AdaBoostRegressor',
        'BaggingRegressor',
        'BayesianRidge',
        'CCA',
        'DecisionTreeRegressor',
        'ElasticNet',
        'ElasticNetCV',
        'ExtraTreeRegressor',
        'ExtraTreesRegressor',
        'GammaRegressor',
        'GradientBoostingRegressor',
        'HistGradientBoostingRegressor',
        'HuberRegressor',
        'KNeighborsRegressor',
        'KernelRidge',
        'Lars',
        'LarsCV',
        'Lasso',
        'LassoCV',
        'LassoLars',
        'LassoLarsCV',
        'LassoLarsIC',
        'LinearRegression',
        'LinearSVR',
        'MLPRegressor',
        'NuSVR',
        'OrthogonalMatchingPursuit',
        'OrthogonalMatchingPursuitCV',
        'PLSCanonical',
        'PLSRegression',
        'PassiveAggressiveRegressor',
        'PoissonRegressor',
        'QuantileRegressor',
        'RANSACRegressor',
        'RandomForestRegressor',
        'Ridge',
        'RidgeCV',
        'SVR',
        'TheilSenRegressor',
        'TransformedTargetRegressor',
        'TweedieRegressor',  
        ]
    return model_names