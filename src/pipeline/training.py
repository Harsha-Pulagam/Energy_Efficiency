
import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.component.data_ingestion import DataIngestion
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer


if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path= obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    # for Heating Load:
    train_arr, test_arr, _ =data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    model_trainer=ModelTrainer()
    model_trainer.initiate_model_training(train_arr,test_arr)
    # FOR Cooling load:
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation_for_cooling(train_data_path,test_data_path)
    model_trainer=ModelTrainer()
    model_trainer.initiate_model_training_for_cooling(train_arr,test_arr)




