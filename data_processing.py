import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class DataProcessing():
    """Class for the cleaning and preprocessing of the datasets"""
    def __init__(self, dataset_for_modelling, submission_set):
        """The initialization receives the datasets as pandas dataframes and prints
        out some information on the dataset to be used to build the model.
      
        Parameters
        ----------
        dataset_for_modelling: pandas.DataFrame
            The dataset containing features and labels to build the model. 

        submission_set: pandas.DataFrame
            The dataset containing the features for which labels have to 
            be predicted for the submission.

        """
        self.dataset_for_modelling = dataset_for_modelling.copy()
        self.submission_set = submission_set.copy()

    def data_info(self, dataset):
        """Use pandas.DataFrame methods to print out information on the dataset"""
        print("DATASET SUMMARY INFORMATION")
        # Take case of null values
        print("\n")
        print("Dataset shape: ", dataset.shape)
        print("\n")
        print(dataset.head(10))
        print("\n")
        print(dataset.info())
        print("\n")
        print(dataset.describe())
        print("\n")
   
    def check_missing_values(self):
        """Check which columns are missing values""" 

        print("Missing values in training set : ")
        print(self.dataset_for_modelling.isna().sum())
        print("\n")
        print("Missing values in submission set : ")
        print(self.submission_set.isna().sum())
        print("\n")
      
    def minimal_preprocessing(self, preprocessing_dict):
        """Deal with the minimum preprocessing to make the data usable.

         Parameters
         ----------
         preprocessing_dict: dict
            Dictionary of instructions on how to preprocess various columns.
            The keys are the keywords to trigger the related methods: 
            "median", "zeros", "category" or "drop".
            The value for each key is a list of the column names to be
            processed in that fashion. For "category" is a list of lists:
            Each sublist contains the name of the column and the categorical 
            ID to use when filling for empty values.

         Example:
         >>> preprocessing_dict = {
               "median" : ["Age", "Fare"],
               "zeros" : [],
               "category" : [["Cabin" , "CCC"],
                              ["Embarked" , "U"]], 
               "drop" : []               
               }
      """

        if "median" in preprocessing_dict.keys():
            for column in preprocessing_dict["median"]:
                self.fill_with_median(column)

        if "zeros" in preprocessing_dict.keys():
            for column in preprocessing_dict["zeros"]:
                self.fill_with_zero(column)

        if "category" in preprocessing_dict.keys():
            for column in preprocessing_dict["category"]:
                self.fill_with_category(column[0], column[1])
                
        if "drop" in preprocessing_dict.keys() and len(preprocessing_dict['drop'])>0:
            self.dropna(preprocessing_dict["drop"])
      
        print("After preprocessing\n")
        self.check_missing_values()
        
    def dropna(self, columns_list):
        """Drop rows with missing values"""
        self.dataset_for_modelling.dropna(how='all', subset=columns_list, inplace=True)
        self.submission_set.dropna(how='all', subset=columns_list, inplace=True)
   
    def fill_with_median(self, column):
        """Fill the missing values for both the modelling 
        and submission set using the median value"""
        self.dataset_for_modelling[column].fillna(self.dataset_for_modelling[column].median(), inplace = True)
        self.submission_set[column].fillna(self.submission_set[column].median(), inplace = True)
   
    def fill_with_zero(self, column):
        """Fill the missing values for both the modelling 
        and submission set using zeros"""
        self.dataset_for_modelling[column].fillna(0, inplace = True)
        self.submission_set[column].fillna(0, inplace = True)
   
    def fill_with_category(self, column, name):      
        """Fill the missing values for both the modelling and 
        submission set using the categorical ID specified in "name" """
        self.dataset_for_modelling[column].fillna(name, inplace = True)
        self.submission_set[column].fillna(name, inplace = True)

    def encode(self, features_to_encode):
        """Encode string-based categorical features"""
        feature_encoder = preprocessing.LabelEncoder()
        for feature_to_encode in features_to_encode:
            self.dataset_for_modelling[feature_to_encode] = feature_encoder.fit_transform(self.dataset_for_modelling[feature_to_encode])
            self.submission_set[feature_to_encode] = feature_encoder.fit_transform(self.submission_set[feature_to_encode])

    def drop(self, features_to_drop):
        """Drop unwanted features"""
        self.dataset_for_modelling = self.dataset_for_modelling.drop(columns=features_to_drop)
        self.submission_set = self.submission_set.drop(columns=features_to_drop)

    def get_dummies(self):
        """Drop unwanted features"""
        self.dataset_for_modelling = pd.get_dummies(self.dataset_for_modelling)
        self.submission_set = pd.get_dummies(self.submission_set)

    def get_values(self, label):
        """Extract labels and feature values, for subsequent splitting"""
        X = self.dataset_for_modelling.drop(columns=[label]).values
        y = self.dataset_for_modelling[label].values
        X_submission = self.submission_set.values
        return X, y, X_submission

    def get_split(self, label, test_size=0.25, shuffle=False, random_state=17):
        """Create train/test splits of the datasets"""
        X, y, X_submission = self.get_values(label)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle, random_state=random_state)
        return X_train, X_test, y_train, y_test, X_submission

