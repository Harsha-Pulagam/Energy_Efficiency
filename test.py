from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from typing import Tuple
import pandas as pd
import numpy as np

class DataLoader:
    def _init_(self, path: str):
        self.path = path
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        data = pd.read_excel(self.path)
        X = data.iloc[:, :8].values
        y = data.iloc[:, 8].values
        return X, y

class ModelTrainer:
    def _init_(self, model):
        self.model = model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        return r2, mse

class ModelComparer:
    def _init_(self, model1, model2):
        self.model1 = model1
        self.model2 = model2
    
    def compare(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        trainer1 = ModelTrainer(self.model1)
        trainer2 = ModelTrainer(self.model2)
        trainer1.train(X_train, y_train)
        trainer2.train(X_train, y_train)
        r2_1, mse_1 = trainer1.evaluate(X_test, y_test)
        r2_2, mse_2 = trainer2.evaluate(X_test, y_test)
        print(f"Model 1: r2 = {r2_1}, mse = {mse_1}")
        print(f"Model 2: r2 = {r2_2}, mse = {mse_2}")

if _name_ == "_main_":
    data_loader = DataLoader("/home/omkar/Omkar Pawar/Data Science/Projects/Energy_Efficiency/notebooks/data/ENB2012_data.xlsx")
    X, y = data_loader.load_data()
    X_train, X_test = X[:600], X[600:]
    y_train, y_test = y[:600], y[600:]
    model1 = LinearRegression()
    model2 = RandomForestRegressor()
    comparer = ModelComparer(model1, model2)
    comparer.compare(X_train, y_train, X_test, y_test)