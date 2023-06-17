import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features, model_path):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            #model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred= model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occurred in prediction")
            raise CustomException(e,sys)

# Taking input from user:

class CustomData:
    def __init__(self,
                 X1:float,
                 X2:float,
                 X3:float,
                 X4:float,
                 X5:float,
                 X6:int,
                 X7:float,
                 X8:int):
        
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.X4 = X4
        self.X5 = X5
        self.X6 = X6
        self.X7 = X7
        self.X8 = X8



    def get_data_as_dataframe(self):
        try:
            CustomData = {
                'X1': [self.X1],
                'X2': [self.X2],
                'X3': [self.X3],
                'X4': [self.X4],
                'X5': [self.X5],
                'X6': [self.X6],
                'X7': [self.X7],
                'X8': [self.X8]
            }

            df = pd.DataFrame(CustomData)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occurred in prediction pipeline')
            raise CustomException(e, sys)
