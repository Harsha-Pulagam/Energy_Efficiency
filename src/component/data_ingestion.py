# Importing Libraries and Modules
import os 
import sys # os and sys used for working with file management and system operations. 
from src.logger import logging
from src.exception import CustomException
import pandas as pd 
from sklearn.model_selection import train_test_split 
from dataclasses import dataclass # class that is primarily used to store data

# Initializing the data Ingestion Configuration:

@dataclass 
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'raw.csv')

# Creating a class for the Data Ingestion:

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Initiating Data Ingestion process...")
        try:
            df = pd.read_excel(os.path.join('notebooks/data', 'ENB2012_data.xlsx')) # notebooks/data /ENB2012_data.xlsx
            logging.info("Dataset loaded as Pandas DataFrame Successfully")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True) # If exist_ok = False then it will raise error.
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info("Train Test split in progress...")
            train_set, test_set = train_test_split(df, test_size=0.30, random_state= 42) # random state works like anchor to create random samples

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Data Ingestion Completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("Exception Occurred at Data Ingestion Stage.")
            raise CustomException(e, sys)
        
        

# #---------------------------------
# # Run Test for Data Ingestion:

# if __name__ == "__main__":
#     obj = DataIngestion()
#     train_data, test_data = obj.initiate_data_ingestion()

# # test successful. 