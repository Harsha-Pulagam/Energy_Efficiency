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
                 Glazing_Area_Distribution: int,
                 Heating_Load: float,
                 Cooling_Load: float):
        
        self.Relative_Compactness = Relative_Compactness
        self.Surface_Area = Surface_Area
        self.Wall_Area = Wall_Area
        self.Roof_Area = Roof_Area
        self.Overall_Height = Overall_Height
        self.Orientation = Orientation
        self.Glazing_Area = Glazing_Area
        self.Glazing_Area_Distribution = Glazing_Area_Distribution
        self.Heating_Load = Heating_Load
        self.Cooling_Load = Cooling_Load

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Relative Compactness': [self.Relative_Compactness],
                'Surface Area': [self.Surface_Area],
                'Wall Area': [self.Wall_Area],
                'Roof Area': [self.Roof_Area],
                'Overall Height': [self.Overall_Height],
                'Orientation': [self.Orientation],
                'Glazing Area': [self.Glazing_Area],
                'Glazing Area Distribution': [self.Glazing_Area_Distribution],
                'Heating Load': [self.Heating_Load],
                'Cooling Load': [self.Cooling_Load]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occurred in prediction pipeline')
            raise CustomException(e, sys)
