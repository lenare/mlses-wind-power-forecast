''' XGBoost model class '''
from typing import Tuple
from matplotlib import pyplot as plt

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb


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
        self.next_step_model = xgb.XGBRegressor(n_estimators=500, max_depth=5, eta=0.1, subsample=0.7, colsample_bytree=0.8,)
        self.next_day_model = xgb.XGBRegressor(n_estimators=500, max_depth=5, eta=0.1, subsample=0.7, colsample_bytree=0.8,)
        self.next_hour_model = xgb.XGBRegressor(n_estimators=500, max_depth=5, eta=0.1, subsample=0.7, colsample_bytree=0.8,)
        self.data_type = data_type

    def split_features_labels(self, data: pd.DataFrame(), features: list(), target_next_step: str, target_next_hour: str, target_next_day: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        ''' Split features and labels '''
        # Prepare the training and test data for different horizons
        X_data, y_next_step, y_next_hour, y_next_day = data[features], data[target_next_step], data[target_next_hour], data[target_next_day]
        return X_data, y_next_step, y_next_hour, y_next_day

    def train_models(self, X_train: pd.DataFrame(), y_train_next_step: pd.DataFrame(), y_train_next_hour: pd.DataFrame(), y_train_next_day: pd.DataFrame(), X_val: pd.DataFrame(), y_val_next_step: pd.DataFrame(), y_val_next_hour: pd.DataFrame(), y_val_next_day: pd.DataFrame()):
        ''' Train models '''
        self.next_step_model = self.next_step_model.fit(
            X_train, y_train_next_step, eval_set=[(X_val, y_val_next_step)], early_stopping_rounds=50, verbose=True)
        self.next_hour_model = self.next_hour_model.fit(
            X_train, y_train_next_hour, eval_set=[(X_val, y_val_next_hour)], early_stopping_rounds=50, verbose=True)
        self.next_day_model = self.next_day_model.fit(
            X_train, y_train_next_day, eval_set=[(X_val, y_val_next_day)], early_stopping_rounds=50, verbose=True)

    def predict_models(self, X_test: pd.DataFrame()) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        ''' Make predictions '''
        return self.next_step_model.predict(X_test), self.next_hour_model.predict(X_test), self.next_day_model.predict(X_test)

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
        with open(f"{self.model_type}_{self.data_type}_evaluation.md", "w") as f:
            f.write(f"# {self.model_type} Evaluation\n")
            f.write(
                f"Mean Absolute Error (Next Step): {self.mae_next_step}\n\n")
            f.write(f"RMSE (Next Step): {self.rmse_next_step}\n\n")
            f.write(
                f"Mean Absolute Error (Next Hour): {self.mae_next_hour}\n\n")
            f.write(f"RMSE (Next Hour): {self.rmse_next_hour}\n\n")
            f.write(f"Mean Absolute Error (Next Day): {self.mae_next_day}\n\n")
            f.write(f"RMSE (Next Day): {self.rmse_next_day}\n\n")
        return self.mae_next_step, self.rmse_next_step, self.mae_next_hour, self.rmse_next_hour, self.mae_next_day, self.rmse_next_day

    def save_prediction_plot(self, y_test: pd.DataFrame(), y_pred: pd.DataFrame(), name: str) -> str:
        ''' Save a plot of the actual vs predicted values for a single model '''
        plt.figure(figsize=(20, 10))
        plt.plot(y_test.values, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title('Actual vs Predicted')
        plt.ylabel('Power (kW)')
        plt.legend()
        # save plot
        plt.savefig(f"{self.model_type}_{name}_predictions.png")

    def save_prediction_plots(self, y_test_next_step: pd.DataFrame(), y_test_next_hour: pd.DataFrame(), y_test_next_day: pd.DataFrame(), y_pred_next_step: pd.DataFrame(), y_pred_next_hour: pd.DataFrame(), y_pred_next_day: pd.DataFrame()):
        ''' Save plots of the actual vs predicted values for all models '''
        return self.save_prediction_plot(y_test_next_step, y_pred_next_step, f"next_step_{self.data_type}"), self.save_prediction_plot(y_test_next_hour, y_pred_next_hour, f"next_hour_{self.data_type}"), self.save_prediction_plot(y_test_next_day, y_pred_next_day, f"next_day_{self.data_type}")

    def save_models(self, model_path: str):
        ''' Save models '''
        self.next_step_model.save_model(f"{model_path}/next_step_model.json")
        self.next_hour_model.save_model(f"{model_path}/next_hour_model.json")
        self.next_day_model.save_model(f"{model_path}/next_day_model.json")

        print(f"Saved models to {model_path}!")
        return model_path
