import os

#import pandas as pd 
import sys

#import numpy as np
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException
#from src.logger import logger
def save_objects(obj, file_path):
    try:
        dir_path =  os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
     raise CustomException(e, sys)


def evaluate_models(X_train, X_test, y_test, y_train, models):
   try:
      report ={}

      for i in range (len(list(models))):
        model = list(models.values())[i]
        model.fit(X_train,y_train)

        y_train_predict = model.predict(X_train)

        y_test_predict = model.predict(X_test)

        train_model_score = r2_score(y_train, y_train_predict)

        test_model_score = r2_score(y_test, y_test_predict)

        report[list(models.keys())[i]] =  test_model_score

        return  report

      
   except Exception as e:
      raise CustomException(e, sys)


