import sys 
from dataclasses import dataclass
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """ This class is responsible for data transformation configuration
    """
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    """ This class is responsible for data transformation
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        """ This function is responsible for data transformation
        """
        try:
            logger.info("Creating data transformer object")

            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = ['gender',"race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logger.info(f"Numerical columns: {numerical_columns}")
            logger.info(f"Categorical columns: {categorical_columns}")
            logger.info("Numerical and categorical pipelines created")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            logger.info("Data transformer object created successfully")
            return preprocessor
        except Exception as e:
            logger.error("Error in creating data transformer object")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        """ This function is responsible for initiating data transformation
        """
        try:
            logger.info("Starting data transformation process")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info("Train and test data read successfully")

            logger.info("Obtaining preprocessor object")
            preprocessor = self.get_data_transformer_object()

            target_column_name = 'math_score'
            numerical_columns = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logger.info("Applying preprocessing object on training and testing data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logger.info("Preprocessing applied successfully")

            logger.info("Saving preprocessor object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            logger.info("Preprocessor object saved successfully")


            logger.info("Data transformation process completed")
            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            logger.error("Error occurred during data transformation")
            raise CustomException(e, sys)
        
