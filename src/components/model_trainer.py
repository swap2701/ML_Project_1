import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging


from src.utlis import save_obj,evaluate_model

@dataclass
class ModelTrainerConfig():
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting Training and Test Input Data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K Neighbors":KNeighborsRegressor(),
                "XGB Regressor":XGBRegressor(),
                "CatBoost Regressor":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor":AdaBoostRegressor()

            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
        
            ### To get best model Score from Dictionary
            best_model_score=max(sorted(model_report.values()))

            ### To get the best model name from Dictionary
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No Best Model found")
            logging.info("Best found model on both training and testing dataset")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            Score_r2= r2_score(y_test,predicted)
            return Score_r2
        
        except Exception as e:
            raise CustomException(e,sys)