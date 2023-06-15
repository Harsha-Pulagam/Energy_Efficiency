import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)

# Taking input from user:

class CustomData:
    def __init__(self,
                 Relative_Compactness: float,
                 Surface_Area: float,
                 Wall_Area: float,
                 Roof_Area: float,
                 Overall_Height: float,
                 Orientation: int,
                 Glazing_Area: float,
                 Glazing_Area_Distribution: int):
        
        self.X1 = Relative_Compactness
        self.X2 = Surface_Area
        self.X3 = Wall_Area
        self.X4 = Roof_Area
        self.X5 = Overall_Height
        self.X6 = Orientation
        self.X7 = Glazing_Area
        self.X8 = Glazing_Area_Distribution


    def get_data_as_dataframe(self):
        try:
            CustomData = {
                'Relative Compactness': [self.X1],
                'Surface Area': [self.X2],
                'Wall Area': [self.X3],
                'Roof Area': [self.X4],
                'Overall Height': [self.X5],
                'Orientation': [self.X6],
                'Glazing Area': [self.X7],
                'Glazing Area Distribution': [self.X8]
            }

            df = pd.DataFrame(CustomData)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occurred in prediction pipeline')
            raise CustomException(e, sys)
