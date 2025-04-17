import os
import sys
# add project root to path so that src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import logging
from math import sqrt
from dataclasses import dataclass

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from catboost import CatBoostRegressor

from src.logger import logging as log
from src.exception import CustomException
from src.utils import save_objects, evaluate_models

# bring in ingestion and transformation components for standalone run
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            log.info("Split training and test input data")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # candidate models
            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet(),
                'DecisionTree': DecisionTreeRegressor(),
                'RandomForest': RandomForestRegressor(),
                'GradientBoosting': GradientBoostingRegressor(),
                'AdaBoost': AdaBoostRegressor(),
                'CatBoost': CatBoostRegressor(verbose=0)
            }

            # hyperparameter grids
            params = {
                'LinearRegression': {},
                'Lasso': {},
                'Ridge': {},
                'ElasticNet': {},
                'DecisionTree': {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error']
                },
                'RandomForest': {
                    'n_estimators': [8, 32, 64, 128]
                },
                'GradientBoosting': {
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [50, 100]
                },
                'AdaBoost': {
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [50, 100]
                },
                'CatBoost': {
                    'depth': [6, 8],
                    'iterations': [50, 100],
                    'learning_rate': [0.1, 0.01]
                }
            }

            # evaluate all models and hyperparameters
            report = evaluate_models(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                models=models,
                param=params
            )

            # select best model
            best_score = max(report.values())
            best_name = next(k for k, v in report.items() if v == best_score)
            best_model = models[best_name]

            if best_score < 0.6:
                raise CustomException(f"No model achieved R2 >= 0.6, highest was {best_score:.3f}")

            log.info(f"Best model: {best_name} with R2={best_score:.3f}")

            # save best model
            os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)
            save_objects(
                file_path=self.config.trained_model_file_path,
                obj=best_model
            )

            return best_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # 1. ingest data
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # 2. transform data
    transformer = DataTransformation()
    train_arr, test_arr, _ = transformer.initiate_data_transformation(train_path, test_path)

    # 3. train and evaluate
    trainer = ModelTrainer()
    best_r2 = trainer.initiate_model_trainer(train_arr, test_arr)

    print(f"✅ Training complete — best model R² = {best_r2:.3f}")

            

