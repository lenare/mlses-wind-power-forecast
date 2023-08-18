''' XGBoost model class '''
from typing import Tuple
from matplotlib import pyplot as plt

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

from utils.config import Config as config


class XGBoost():
    ''' XGBoost model class that wraps three XGBoost models for next step, next hour and next day prediction '''
    model_type: str = 'XGBOOST'
    data_type: str
    next_step_model: xgb.XGBRegressor
    next_hour_model: xgb.XGBRegressor
    next_day_model: xgb.XGBRegressor
    mae_next_step: float
    mae_next_hour: float
    mae_next_day: float
    rmse_next_step: float
    rmse_next_hour: float
    rmse_next_day: float

    def __init__(self, data_type: str):
        ''' Init model '''
        self.next_step_model = xgb.XGBRegressor(**config.XGBOOST_MODEL_PARAMS)
        self.next_day_model = xgb.XGBRegressor(**config.XGBOOST_MODEL_PARAMS)
        self.next_hour_model = xgb.XGBRegressor(**config.XGBOOST_MODEL_PARAMS)
        self.data_type = data_type

    def train_models(self, X_train: pd.DataFrame(), y_train_next_step: pd.DataFrame(), y_train_next_hour: pd.DataFrame(), y_train_next_day: pd.DataFrame(), X_val: pd.DataFrame(), y_val_next_step: pd.DataFrame(), y_val_next_hour: pd.DataFrame(), y_val_next_day: pd.DataFrame(), verbose: bool = False):
        ''' Train models '''
        self.next_step_model = self.next_step_model.fit(
            X_train, y_train_next_step, eval_set=[(X_val, y_val_next_step)], verbose=verbose)
        self.next_hour_model = self.next_hour_model.fit(
            X_train, y_train_next_hour, eval_set=[(X_val, y_val_next_hour)], verbose=verbose)
        self.next_day_model = self.next_day_model.fit(
            X_train, y_train_next_day, eval_set=[(X_val, y_val_next_day)], verbose=verbose)

    def predict_models(self, X_test: pd.DataFrame()) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        ''' Make predictions '''
        return self.next_step_model.predict(X_test), self.next_hour_model.predict(X_test), self.next_day_model.predict(X_test)

    def plot_feature_importances(self, feature_names: list, top_n: int = 10, model_type: str = 'next_step'):
        ''' Plot feature importance '''
        # top_n features must be <= than the number of features of the model
        if top_n > len(self.next_step_model.feature_importances_):
            raise ValueError(
                f"top_n ({top_n}) must be <= than the number of features ({len(self.next_step_model.feature_importances_)})!")

        # map model_type to model
        model_type_map = {
            'next_step': self.next_step_model,
            'next_hour': self.next_hour_model,
            'next_day': self.next_day_model
        }
        if model_type not in model_type_map.keys():
            raise ValueError(
                f"model_type ({model_type}) must be one of {model_type_map.keys()}!")

        # create feature importance dataframe
        feature_importance_df = pd.DataFrame(
            {'feature': feature_names, 'importance': model_type_map[model_type].feature_importances_}).sort_values(by='importance', ascending=False)
        # plot top_n most important features with plt
        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance')
        plt.xlabel('Relative Importance')
        plt.ylabel('Features')
        plt.barh(feature_importance_df[:top_n]['feature'],
                 feature_importance_df[:top_n]['importance'])
        plt.show()

    def evaluate_model(self, y_test: pd.DataFrame(), y_pred: pd.DataFrame()) -> Tuple[float, float]:
        ''' Evaluate the next step predictions using MAE and RMSE '''
        return mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred) ** 0.5

    def evaluate_models(self, y_test_next_step: pd.DataFrame(), y_test_next_hour: pd.DataFrame(), y_test_next_day: pd.DataFrame(), y_pred_next_step: pd.DataFrame(), y_pred_next_hour: pd.DataFrame(), y_pred_next_day: pd.DataFrame()) -> Tuple[float, float, float, float, float, float]:
        ''' Evaluate the next step, next hour and next day predictions using MAE and RMSE '''
        self.mae_next_step, self.rmse_next_step = self.evaluate_model(
            y_test_next_step, y_pred_next_step)
        self.mae_next_hour, self.rmse_next_hour = self.evaluate_model(
            y_test_next_hour, y_pred_next_hour)
        self.mae_next_day, self.rmse_next_day = self.evaluate_model(
            y_test_next_day, y_pred_next_day)
        benchmark_df = pd.read_csv('../benchmark_wind.csv', index_col=3)
        benchmark_df = benchmark_df[(benchmark_df.data_type == self.data_type)]
        eval_df = pd.DataFrame(
                    {
                        'MAE': [self.mae_next_step, self.mae_next_hour, self.mae_next_day],
                        'benchmark_MAE': benchmark_df.MAE.values,
                        'RMSE': [self.rmse_next_step, self.rmse_next_hour, self.rmse_next_day],
                        'benchmark_RMSE': benchmark_df.RMSE.values,
                    },
                    index=['next_step', 'next_hour', 'next_day'],
                )
        return eval_df

    def save_prediction_plot(self, y_test: pd.DataFrame(), y_pred: pd.DataFrame(), name: str, dir_path: str = './plots') -> str:
        ''' Save a plot of the actual vs predicted values for a single model '''
        plt.figure(figsize=(20, 10))
        plt.plot(y_test.values, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title(f'{self.data_type} {name} Predictions')
        plt.ylabel('Power (kW)')
        plt.legend()
        # save plot
        plt.savefig(
            f"{dir_path}/{self.model_type}_{self.data_type}_{name}_predictions.png")

    def save_prediction_plots(self, y_test_next_step: pd.DataFrame(), y_test_next_hour: pd.DataFrame(), y_test_next_day: pd.DataFrame(), y_pred_next_step: pd.DataFrame(), y_pred_next_hour: pd.DataFrame(), y_pred_next_day: pd.DataFrame()):
        ''' Save plots of the actual vs predicted values for all models '''
        return (
            self.save_prediction_plot(
                y_test_next_step, y_pred_next_step, f"next_step"),
            self.save_prediction_plot(
                y_test_next_hour, y_pred_next_hour, f"next_hour"),
            self.save_prediction_plot(
                y_test_next_day, y_pred_next_day, f"next_day"),
        )

    def save_models(self, model_path: str):
        ''' Save models '''
        self.next_step_model.save_model(f"{model_path}/next_step_model.json")
        self.next_hour_model.save_model(f"{model_path}/next_hour_model.json")
        self.next_day_model.save_model(f"{model_path}/next_day_model.json")

        print(f"Saved models to {model_path}!")
        return model_path
