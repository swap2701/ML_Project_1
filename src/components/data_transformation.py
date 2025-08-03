import sys
from dataclasses import dataclass
import os


import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


from src.exception import CustomerException
from src.logger import logging

from src.utlis import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        """
        This Function is responsible for Data Transformation
        """
    
        logging.info("Entered the Data Transformation Component")
        try:
            numerical_features=["reading_score","writing_score"]
            categorical_features=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            numerical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("Scaler",StandardScaler())
                    ]   
                )
            
        
            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
                    ("Scaler",StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f"Categorical Features: {categorical_features}")
            logging.info(f"Numerical Features: {numerical_features}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",numerical_pipeline,numerical_features),
                    ("cat_pipeline",categorical_pipeline,categorical_features)
                ]
            )
            logging.info("Numerical Columns standard scaling Completed")
            logging.info("Categorical Columns encoding Completed")

            return preprocessor
        
        except Exception as e:
            raise CustomerException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read Train and Test Completed")

            logging.info("Obtaining Preprocessor Object")

            preprocessor_obj=self.get_data_transformation_object()

            target_column_name="math_score"
            numerical_features=["reading_score","writing_score"]

            input_feature_train_df=train_df.drop(columns=target_column_name,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=target_column_name,axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")


            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info("Saved Preprocessing Objects")

            save_obj(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomerException(e,sys)

