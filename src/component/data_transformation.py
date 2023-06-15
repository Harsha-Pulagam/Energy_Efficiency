# Importing Libraries:
import os
import sys

import numpy as np
import pandas as pd 

from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer # It is used to fill out missing values...
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder # OneHotEncoder might needed for Orientation Column. 

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object 
from dataclasses import dataclass

# Data Transformation Configuration:
@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

# Data Transformation: 
class DataTransformation: 
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # Getting Preprocessor:
    def get_data_transformation_object(self):
        try: 
            logging.info('Data transformation initiated...')

            # Separate features based on Categorical and Numerical features:
            categorical_col = ['X6']
            numerical_col = ['X1', 'X2', 'X3', 'X4', 'X5', 'X7', 'X8', 'Y1', 'Y2']

            logging.info('Pipeline Inited...')

            ## Numerical Pipeline: 
            num_pipeline = Pipeline(
                steps= [
                    ('scaler', StandardScaler())
                ]
            )

            ## Categorical Pipeline: 
            cat_pipeline = Pipeline(
                steps= [
                    ('onehotencoding', OneHotEncoder())
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_col)
                ('cat_pipeline', cat_pipeline, categorical_col)
            ])

            logging.info('Pipeline Completed')

            return preprocessor
            
        except Exception as e:
            logging.error('Error in data Transformation step')
            raise CustomException(e, sys)
        
# Data Transformation using the Preprocessor:

def initiate_data_transformation(self, train_path, test_path):
    try:
        # loading train and test data
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # logging few details: 
        logging.info('Read train and test data completed')
        logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
        logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

        logging.info('Obtaining preprocessing object')

        # Creating preprocessing object:
        preprocessing_obj = self.get_data_transformation_object()

        # Separating dependent and independent features:
        target_column_name = 'Y1'

        input_feature_train_df = train_df.drop(columns=target_column_name, axis=1)
        target_feature_train_df = train_df[target_column_name]

        input_feature_test_df = test_df.drop(columns=target_column_name, axis=1)
        target_feature_test_df = test_df[target_column_name]

        ## Transforming using preprocessor obj:
        input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
        input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

        logging.info("Applying preprocessing object on training and testing datasets.")

        train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
        test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

        save_object(
            file_path = self.data_transformation_config.preprocessor_obj_file_path, 
            obj = preprocessing_obj
        )

        logging.info('Preprocessing pickle file saved')

        return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path
        )
    
    except Exception as e:
        logging.info('Exception occured in the initiate ')