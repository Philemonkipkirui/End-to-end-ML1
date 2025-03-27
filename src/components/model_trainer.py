import logging
import os


from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from catboost import CatBoostRegressor

import sys

from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException


from src.utils import save_objects, evaluate_models

@dataclass
class ModelTrainerConfig: 
    trained_model_file_path =  os.path.join('artifacts', "model.pkl")

class ModelTrainer():
    def __init__(self):
    
        self.model_Trainer_config =  ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr, preprocessor_path):
        try:
           logging.info ("Split training and test input data")
           X_train, y_train, X_test, y_test = (
               
               train_arr[:,:-1],
               train_arr[:, -1],
               test_arr[:, :-1],
               test_arr[:, -1]
           )
           models = {
               
                   'Linear Regression': LinearRegression(),
                    'Lasso': Lasso(),
                    'Ridge': Ridge(),
                    'ElasticNet': ElasticNet(),
                    'DecisionTreeRegressor': DecisionTreeRegressor(),
                    'RandomForestRegressor': RandomForestRegressor(),
                    'GradientBoostingRegressor': GradientBoostingRegressor(),
                    'AdaBoostRegressor': AdaBoostRegressor(),
                    'CatBoostRegressor': CatBoostRegressor(verbose=0),

           }

           model_report:dict = evaluate_models(X_train = X_train, X_test = X_test , y_train = y_train, y_test= y_test, models = models)


           ##  To get the model score from dict

           best_model_score =  max(model_report.values())

           ## To get teh best model ame from dict

           best_model_name = next(name for name, score in model_report.items() if score == best_model_score)
           best_model = models[best_model_name]


           if best_model_score<0.6:
               raise CustomException("No best model found")
           

           logging.info(f"Best model determination terminated")

           save_objects(
               file_path = self.model_Trainer_config.trained_model_file_path,
               obj  = best_model
           )


           predicted =  best_model.predict(X_test)

           r2 = r2_score(y_test, predicted)
           return r2

        except Exception as e:
            raise CustomException(e, sys)
            

