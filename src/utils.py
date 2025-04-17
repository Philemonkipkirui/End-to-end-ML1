import os

#import pandas as pd 
import sys

#import numpy as np
import dill
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


#from src.logger import logger
def save_objects(obj, file_path):
    try:
        dir_path =  os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
     raise CustomException(e, sys)


def evaluate_models(X_train, X_test, y_test, y_train, models, param):
   try:
        report ={}

        # Iterate over each model in the dictionary
        for model_name, model in models.items():
            # Get the corresponding parameter grid (default to {} if not present)
            param_grid = param.get(model_name, {})

            # If the parameter grid is empty, fit the model directly.
            if not param_grid:
                model.fit(X_train, y_train)
                best_model = model
            else:
                gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=0, error_score=np.nan)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
                
                # Ensure that best_model is fitted; if not, fit it directly.
                try:
                    check_is_fitted(best_model)
                except NotFittedError:
                    best_model.fit(X_train, y_train)
                    

            y_train_predict = best_model.predict(X_train)
            y_test_predict = best_model.predict(X_test)


            train_model_score = r2_score(y_train, y_train_predict)
            test_model_score = r2_score(y_test, y_test_predict)

            report[model_name] = test_model_score

        return  report

      
   except Exception as e:
      raise CustomException(e, sys)
def load_objects(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)   

 