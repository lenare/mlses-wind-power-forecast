''' Data Preprocessor class for relevant preprocessing steps '''
from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from xarray import Dataset

from utils.config import Config as config


class DataPreprocessor():
    train_ratio: float
    valid_ratio: float
    test_ratio: float
    model_type: str

    def __init__(self, train_ratio: float = 0.8, valid_ratio: float = 0, test_ratio: float = 0.2, model_type: str = "XGBOOST"):
        ''' Init with train/val/test ratio '''
        if train_ratio + valid_ratio + test_ratio != 1:
            raise ValueError(
                'train_ratio + valid_ratio + test_ratio must equal 1!')
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        if model_type != "XGBOOST":
            raise ValueError('Only XGBOOST model_type currently supported!')
        self.model_type = model_type

    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        ''' Rename columns to snake_case '''
        column_map = {
            column: column.strip().lower().replace(" ", "_").replace("-", "_") for column in df.columns
        }
        result_df = df.rename(columns=column_map)
        return result_df

    # def fill_na(self, df: pd.DataFrame) -> pd.DataFrame():
    #     ''' Fill missing/NaN values '''
    #     # TODO: improve selection and handling of missing values
    #     df = df['06-01-2016':]
    #     return df.fillna(method='ffill')

    def shift_target_variable(self, df: pd.DataFrame, target_variable: str) -> pd.DataFrame:
        ''' Shift data for different time horizons '''
        shifted_df = df.copy()
        # Shift the target variable for different horizons
        shifted_df['power_next_step'] = df[target_variable].shift(-1)
        shifted_df['power_next_hour'] = df[target_variable].shift(-6)
        shifted_df['power_next_day'] = df[target_variable].shift(-6*24)
        # Drop rows with NaN values in the shifted columns
        shifted_df.dropna(
            subset=['power_next_step', 'power_next_hour', 'power_next_day'], inplace=True)
        return shifted_df

    def train_val_test_split(self, df) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] or Tuple[pd.DataFrame, pd.DataFrame]:
        ''' Split data into train, (valid) and test set '''
        timestamp_delta = df.index[-1] - df.index[0]
        timestamp_step_delta = df.index[1] - df.index[0]
        print(f"timestamp_delta: {timestamp_delta}")
        print(f"timestamp_step_delta: {timestamp_step_delta}")

        # Train set
        start_train = df.index[0]
        end_train = (start_train + timestamp_delta * self.train_ratio)
        train = df[start_train:end_train.floor(timestamp_step_delta)]

        # Validation set
        end_valid = end_train + timestamp_delta * self.valid_ratio
        valid = df[end_train.ceil(timestamp_step_delta)                   :end_valid.floor(timestamp_step_delta)]

        # Test set
        end_test = end_valid + timestamp_delta * self.test_ratio
        test = df[end_valid.ceil(timestamp_step_delta):df.index[-1]]

        print(f"train range: {start_train}/{end_train.floor(timestamp_step_delta)}" +
              f"\nvalid range: {end_train.ceil(timestamp_step_delta)}/{end_valid.floor(timestamp_step_delta)}" +
              f"\ntest range: {end_valid.ceil(timestamp_step_delta)}/{end_test.ceil(timestamp_step_delta)}")

        # Raise exception if last timestamp of df is not equal to last timestamp of test set
        if df.index[-1] != end_test.ceil(timestamp_step_delta):
            raise ValueError(
                'end_test.ceil(timestamp_step_delta) must equal df.index[-1]!')

        print(f"train range: {start_train}/{end_train.floor(timestamp_step_delta)}" +
              f"\nvalid range: {end_train.ceil(timestamp_step_delta)}/{end_valid.floor(timestamp_step_delta)}" +
              f"\ntest range: {end_valid.ceil(timestamp_step_delta)}/{end_test.ceil(timestamp_step_delta)}")

        if self.valid_ratio > 0:
            return train, valid, test
        else:
            return train, test

    def plot_data_split(self, train_data: pd.DataFrame(), val_data: pd.DataFrame(), test_data: pd.DataFrame()) -> None:
        ''' Plot the data split into training, validation and testing '''
        figure, ax = plt.subplots(figsize=(20, 5))
        train_data.resample('D').mean().plot(
            ax=ax, label="Training", y="power_next_step")
        val_data.resample('D').mean().plot(
            ax=ax, label="Validation", y="power_next_step")
        test_data.resample('D').mean().plot(
            ax=ax, label="Testing",  y="power_next_step", title="Power")
        plt.show()

    def create_rolling_window_statistics(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        ''' Create rolling window statistics for features '''
        df['power_1h_mean'] = df[column].rolling(window=6).mean()
        df['power_12h_mean'] = df[column].rolling(window=6*12).mean()
        df['power_24h_mean'] = df[column].rolling(window=6*24).mean()
        df['power_1h_std'] = df[column].rolling(window=6).std()
        df['power_12h_std'] = df[column].rolling(window=6*12).std()
        df['power_24h_std'] = df[column].rolling(window=6*24).std()
        df['power_1h_max'] = df[column].rolling(window=6).max()
        df['power_12h_max'] = df[column].rolling(window=6*12).max()
        df['power_24h_max'] = df[column].rolling(window=6*24).max()
        df['power_1h_min'] = df[column].rolling(window=6).min()
        df['power_12h_min'] = df[column].rolling(window=6*12).min()
        df['power_24h_min'] = df[column].rolling(window=6*24).min()
        return df

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ''' Create time features '''
        df["season"] = df.index.month % 12 // 3
        df["month"] = df.index.month
        df['hour'] = df.index.hour
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        df['day'] = df.index.day
        return df

    def encode_cyclic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ''' Encode time features as sin and cos and add to dataframe '''
        def encode(data, col, max_val):
            data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
            data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
            return data

        df = encode(df, 'hour', 24)
        df = encode(df, 'month', 12)
        df = encode(df, 'quarter', 4)
        df = encode(df, 'season', 4)
        df = encode(df, 'day', 31)
        return df
    
    def get_features_labels(self, df: pd.DataFrame, target_variable: str) -> Tuple[list, str, str, str]:
        ''' Split dataframe into features and labels '''
        # TODO: improve
        features = list(df.columns)
        features.remove(target_variable) if target_variable in features else None
        features.remove(target_variable + "_min") if target_variable + "_min" in features else None
        features.remove(target_variable + "_max") if target_variable + "_max" in features else None
        features.remove('power_next_step') if 'power_next_step' in features else None
        features.remove('power_next_hour') if 'power_next_hour' in features else None
        features.remove('power_next_day') if 'power_next_day' in features else None
        features.extend(["season", "month", "hour", "quarter", "year", "power_1h_mean", "power_12h_mean", "power_24h_mean", "power_1h_std", "power_12h_std", "power_24h_std", "power_1h_max", "power_12h_max", "power_24h_max", "power_1h_min", "power_12h_min", "power_24h_min"])
        features = list(set(features))

        target_next_step = 'power_next_step'
        target_next_hour = 'power_next_hour'
        target_next_day = 'power_next_day'

        return features, target_next_step, target_next_hour, target_next_day


class UKDataPreprocessor(DataPreprocessor):
    ''' Data preprocessor for UK data '''
    data_type: str = "UK"

    def select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        ''' Select which columns of dataframe columns to keep'''
        # TODO: improve which columns to select
        # drop all columns in df where either 'max' or 'min' or 'stddev' or 'standard_deviation' is in the column name
        selected_df = df.loc[:, ~df.columns.str.contains(
            'max|min|stddev|standard_deviation|lost_production|time_based|data_availability')]

        return selected_df

    def set_timestamp_index(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        ''' Set column as timestamp index and rename '''
        df = df.set_index(column)
        df.index = pd.to_datetime(df.index).rename("timestamp")
        df = df['06-2016':]
        # sort index if not already sorted
        df = df.sort_index()
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame():
        ''' Preprocess data combining several steps defined in this class '''
        # Rename columns, set timestamp as index
        renamed_df = self.rename_columns(df)
        reindexed_df = self.set_timestamp_index(
            renamed_df, '#_date_and_time')
        preprocessed_data = self.select_columns(reindexed_df)

        return preprocessed_data


class BrazilDataPreprocessor(DataPreprocessor):
    ''' Data preprocessor for Brazil data '''
    data_type: str = "Brazil"

    def select_turbine(self, ds: Dataset, turbine_number: int) -> pd.DataFrame():
        ''' Select which turbine to keep'''
        if turbine_number not in ds.Turbine:
            raise ValueError(
                f"Turbine {turbine_number} not in dataset! Please choose one of {ds.Turbine.values.tolist()}")
        else:
            df = ds.sel(Turbine=2).to_dataframe()
            # drop column Turbine
            df = df.drop(columns=["Turbine"])

        return df

    def handle_height_coordinate(self, df: pd.DataFrame) -> pd.DataFrame():
        ''' Handle height coordinate '''
        # TODO: test out other strategies than mean
        return df.groupby(level=0).mean()

    def rename_index(self, df: pd.DataFrame) -> pd.DataFrame():
        ''' Rename index '''
        df.index.rename("timestamp", inplace=True)
        # sort index
        df.sort_index(inplace=True)
        return df

    def select_columns(self, df: pd.DataFrame) -> pd.DataFrame():
        ''' Select which columns of dataframe columns to keep'''
        # TODO: improve which columns to drop
        if len(config.UK_COLUMNS) > 0:
            # only use columns specified in config
            selected_df = df[config.UK_COLUMNS]
        else:
            # columns with less than specified percentage missing values
            is_missing = df.isna().sum()/len(df)
            selected_df = df[list(is_missing[is_missing <= 0.04].index)]

            # TODO: unclear, improve!!
            # only use columns with 75% of values greater than 0
            selected_columns = selected_df.columns[selected_df.quantile(
                0.75) > 0]
            selected_df = selected_df[selected_columns]

        return selected_df

    def preprocess_data(self, ds: Dataset, turbine_number: int) -> pd.DataFrame():
        ''' Preprocess data combining several steps defined in this class '''
        # Select turbine, rename columns to snake case, set timestamp as index, select columns
        selected_turbine_df = self.select_turbine(ds, turbine_number)
        grouped_df = self.handle_height_coordinate(selected_turbine_df)
        renamed_df = self.rename_columns(grouped_df)
        preprocessed_data = self.rename_index(renamed_df)
        # selected_df = self.select_columns(reindexed_df)

        return preprocessed_data

