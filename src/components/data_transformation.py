import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
#from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_objects



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class  DataTransformation:
    def __init__(self):
        self.data_transformation_config =  DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            numerical_columns =  ['Hours_Studied','Attendance','Sleep_Hours','Previous_Scores','Tutoring_Sessions','Physical_Activity']
            categorical_columns = ['Parental_Involvement','Access_to_Resources','Extracurricular_Activities','Motivation_Level','Internet_Access','Family_Income','Teacher_Quality','School_Type','Peer_Influence','Learning_Disabilities','Parental_Education_Level','Distance_from_Home','Gender']

            num_pipeline = Pipeline([
                ('scaler', StandardScaler(with_mean= False))
            ])
            cat_pipeline = Pipeline(
                steps= [
                    ('imputer', SimpleImputer(strategy =  "most_frequent")),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown= 'ignore')),
                    ('scaler', StandardScaler(with_mean= False))
                
            ])

            logging.info("Numerical column scaling completed")
            logging.info("catgorical_column encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df =pd.read_csv(test_path)

            logging.info(" Read train data from csv completed")
            logging.info("obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = "Exam_Score"
            numerical_columns =  ['Hours_Studied','Attendance','Sleep_Hours','Previous_Scores','Tutoring_Sessions','Physical_Activity']
            input_feature_train_df =  train_df.drop(columns =[target_column_name], axis =  1)
            target_feature_train_df =  train_df[target_column_name]

            input_feature_test_df =  test_df.drop(columns =[target_column_name], axis =  1)
            target_feature_test_df =  test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training datarfame and testing dataframe")


            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr =  np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            logging.info(f"saved preprocessing object")
            save_objects(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj =preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )



        except Exception as e:
            raise CustomException(e,sys)
            

 



