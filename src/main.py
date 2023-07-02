""" 
    Main file of the project executing the following steps:
        - loads data from file_path
        - preprocesses data
        - splits data into train, (valid) and test set
        - trains model
        - evaluates model
        - makes predictions for horizons next 10 min, next hour, next day
"""
from data.data_loader import DataLoader
from data.data_preprocessor import BrazilDataPreprocessor, UKDataPreprocessor
from utils.config import Config as config
from models.xgboost import XGBoost


def main():
    # Load data for UK Turbine 2 and Brazil UEBB
    data_loader = DataLoader()
    print(
        f"Reading following files into uk_data dataframe: {config.UK_RAW_FILES}")
    uk_data_raw = data_loader.load_data(config.UK_RAW_FILES, skiprows=9)
    print(
        f"Reading following files into brazil_data dataset: {config.BRAZIL_RAW_FILES}")
    brazil_data_raw = data_loader.load_data(config.BRAZIL_RAW_FILES)

    # Preprocess data
    print("Preprocessing data...")
    uk_data_preprocessor = UKDataPreprocessor(
        train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1)
    brazil_data_preprocessor = BrazilDataPreprocessor(
        train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1)
    uk_preprocessed_data = uk_data_preprocessor.preprocess_data(uk_data_raw)
    brazil_preprocessed_data = brazil_data_preprocessor.preprocess_data(
        brazil_data_raw, 2)
    print(brazil_preprocessed_data.head())

    # Feature engineering
    # Shift data by 1 timestep to predict next timestep
    print("Shifting data...")  # lag
    uk_data = uk_data_preprocessor.shift_target_variable(
        uk_preprocessed_data, config.UK_TARGET_VARIABLE)
    brazil_data = brazil_data_preprocessor.shift_target_variable(
        brazil_preprocessed_data, config.BRAZIL_TARGET_VARIABLE)
    print("Creating rolling window statistics...")
    uk_data = uk_data_preprocessor.create_rolling_window_statistics(
        uk_data, config.UK_TARGET_VARIABLE)
    brazil_data = brazil_data_preprocessor.create_rolling_window_statistics(
        brazil_data, config.BRAZIL_TARGET_VARIABLE)
    print("Creating time features...")
    uk_data = uk_data_preprocessor.create_time_features(uk_data)
    brazil_data = brazil_data_preprocessor.create_time_features(brazil_data)
    print("Encode cyclic time features...")
    uk_data = uk_data_preprocessor.encode_cyclic_features(uk_data)
    brazil_data = brazil_data_preprocessor.encode_cyclic_features(brazil_data)

    # Split data into train, (valid) and test sets
    print("Splitting data into train, (valid) and test sets...")
    uk_train, uk_val, uk_test = uk_data_preprocessor.train_val_test_split(
        uk_data)
    brazil_train, brazil_val, brazil_test = brazil_data_preprocessor.train_val_test_split(
        brazil_data)

    # # Save preprocessed full, train, (valid,) test data (optional)
    # print("Saving preprocessed data...")
    # data_loader.save_data(uk_preprocessed_data, config.UK_DATA_PATH + "processed/preprocessed.csv", index=True)
    # data_loader.save_data(uk_train, config.UK_DATA_PATH + "processed/train.csv", index=True)
    # data_loader.save_data(uk_val, config.UK_DATA_PATH + "processed/val.csv", index=True)
    # data_loader.save_data(uk_test, config.UK_DATA_PATH + "processed/test.csv", index=True)

    # data_loader.save_data(brazil_preprocessed_data, config.BRAZIL_DATA_PATH + "processed/preprocessed.csv", index=True)
    # data_loader.save_data(brazil_train, config.BRAZIL_DATA_PATH + "processed/train.csv", index=True)
    # data_loader.save_data(brazil_val, config.BRAZIL_DATA_PATH + "processed/val.csv", index=True)
    # data_loader.save_data(brazil_test, config.BRAZIL_DATA_PATH + "processed/test.csv", index=True)

    # Getting features and labels
    uk_features, uk_target_next_step, uk_target_next_hour, uk_target_next_day = uk_data_preprocessor.get_features_labels(
        uk_data, config.UK_TARGET_VARIABLE)
    brazil_features, brazil_target_next_step, brazil_target_next_hour, brazil_target_next_day = brazil_data_preprocessor.get_features_labels(
        brazil_data, config.BRAZIL_TARGET_VARIABLE)

    # Initialize model
    print("Initializing model...")
    uk_xgboost = XGBoost(data_type='uk')
    brazil_xgboost = XGBoost(data_type='brazil')

    # Split into features and labels
    print("Splitting data into features and labels...")
    uk_X_train, uk_y_train_next_step, uk_y_train_next_hour, uk_y_train_next_day = uk_xgboost.split_features_labels(
        uk_train, uk_features, uk_target_next_step, uk_target_next_hour, uk_target_next_day)
    uk_X_val, uk_y_val_next_step, uk_y_val_next_hour, uk_y_val_next_day = uk_xgboost.split_features_labels(
        uk_val, uk_features, uk_target_next_step, uk_target_next_hour, uk_target_next_day)
    uk_X_test, uk_y_test_next_step, uk_y_test_next_hour, uk_y_test_next_day = uk_xgboost.split_features_labels(
        uk_test, uk_features, uk_target_next_step, uk_target_next_hour, uk_target_next_day)

    brazil_X_train, brazil_y_train_next_step, brazil_y_train_next_hour, brazil_y_train_next_day = brazil_xgboost.split_features_labels(
        brazil_train, brazil_features, brazil_target_next_step, brazil_target_next_hour, brazil_target_next_day)
    brazil_X_val, brazil_y_val_next_step, brazil_y_val_next_hour, brazil_y_val_next_day = brazil_xgboost.split_features_labels(
        brazil_val, brazil_features, brazil_target_next_step, brazil_target_next_hour, brazil_target_next_day)
    brazil_X_test, brazil_y_test_next_step, brazil_y_test_next_hour, brazil_y_test_next_day = brazil_xgboost.split_features_labels(
        brazil_test, brazil_features, brazil_target_next_step, brazil_target_next_hour, brazil_target_next_day)

    # Train xgboost models
    print("Training models...")
    uk_xgboost.train_models(uk_X_train, uk_y_train_next_step, uk_y_train_next_hour,
                            uk_y_train_next_day, uk_X_val, uk_y_val_next_step, uk_y_val_next_hour, uk_y_val_next_day)
    brazil_xgboost.train_models(brazil_X_train, brazil_y_train_next_step, brazil_y_train_next_hour,
                                brazil_y_train_next_day, brazil_X_val, brazil_y_val_next_step, brazil_y_val_next_hour, brazil_y_val_next_day)

    # Make predictions for next step, next hour, next day
    print("Making predictions...")
    uk_y_pred_next_step, uk_y_pred_next_hour, uk_y_pred_next_day = uk_xgboost.predict_models(
        uk_X_test)
    brazil_y_pred_next_step, brazil_y_pred_next_hour, brazil_y_pred_next_day = brazil_xgboost.predict_models(
        brazil_X_test)

    # Print evaluation metrics of models as markdown file '{MODEL_NAME}_evaluation.md'
    print("Evaluating models...")
    uk_xgboost.evaluate_models(uk_y_test_next_step, uk_y_test_next_hour, uk_y_test_next_day,
                               uk_y_pred_next_step, uk_y_pred_next_hour, uk_y_pred_next_day)
    brazil_xgboost.evaluate_models(brazil_y_test_next_step, brazil_y_test_next_hour, brazil_y_test_next_day,
                                   brazil_y_pred_next_step, brazil_y_pred_next_hour, brazil_y_pred_next_day)

    # Save prediction plots
    print("Saving prediction plots...")
    next_step_path, next_hour_path, next_day_path = uk_xgboost.save_prediction_plots(
        uk_y_test_next_step, uk_y_test_next_hour, uk_y_test_next_day, uk_y_pred_next_step, uk_y_pred_next_hour, uk_y_pred_next_day)
    print(
        f"Saved UK prediction plots to {next_step_path}, {next_hour_path}, {next_day_path}")
    next_step_path, next_hour_path, next_day_path = brazil_xgboost.save_prediction_plots(
        brazil_y_test_next_step, brazil_y_test_next_hour, brazil_y_test_next_day, brazil_y_pred_next_step, brazil_y_pred_next_hour, brazil_y_pred_next_day)
    print(
        f"Saved Brazil prediction plots to {next_step_path}, {next_hour_path}, {next_day_path}")

    # Save models
    print("Saving models...")
    model_path = uk_xgboost.save_models(config.UK_DATA_PATH + "models")
    print(f"Saved UK models to {model_path}")
    model_path = brazil_xgboost.save_models(config.BRAZIL_DATA_PATH + "models")
    print(f"Saved Brazil models to {model_path}")


if __name__ == "__main__":
    main()
