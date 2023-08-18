class Config:
    # UK data related
    UK_DATA_PATH = '../data/uk/'
    UK_RAW_FILES = [f'{UK_DATA_PATH}raw/Kelmarsh_SCADA_2020_3086/Turbine_Data_Kelmarsh_2_2020-01-01_-_2021-01-01_229.csv', f'{UK_DATA_PATH}raw/Kelmarsh_SCADA_2021_3087/Turbine_Data_Kelmarsh_2_2021-01-01_-_2021-07-01_229.csv', f'{UK_DATA_PATH}raw/Kelmarsh_SCADA_2018_3084/Turbine_Data_Kelmarsh_2_2018-01-01_-_2019-01-01_229.csv',
                    f'{UK_DATA_PATH}raw/Kelmarsh_SCADA_2019_3085/Turbine_Data_Kelmarsh_2_2019-01-01_-_2020-01-01_229.csv', f'{UK_DATA_PATH}raw/Kelmarsh_SCADA_2016_3082/Turbine_Data_Kelmarsh_2_2016-01-03_-_2017-01-01_229.csv', f'{UK_DATA_PATH}raw/Kelmarsh_SCADA_2017_3083/Turbine_Data_Kelmarsh_2_2017-01-01_-_2018-01-01_229.csv']
    # target variable to predict for uk data
    UK_TARGET_VARIABLE = 'power_(kw)'
    # optional config to select which columns to use, if empty all columns are used
    UK_COLUMNS = []

    # Brazil data related
    BRAZIL_DATA_PATH = '../data/brazil/'
    BRAZIL_RAW_FILES = f'{BRAZIL_DATA_PATH}raw/UEBB_v1.nc'
    # target variable to predict for brazil data
    BRAZIL_TARGET_VARIABLE = 'active_power_total'
    # target turbine to predict for brazil data
    BRAZIL_TURBINE_NUMBER = 2
    # optional config to select which columns to use, if empty all columns are used
    BRAZIL_COLUMNS = []

    # XGBoostRegessor model related
    XGBOOST_MODEL_PARAMS = {
        'n_estimators': 500,
        'max_depth': 5,
        'eta': 0.1,
        'subsample': 0.7,
        'colsample_bytree': 0.8,
        'early_stopping_rounds': 50,
    }
