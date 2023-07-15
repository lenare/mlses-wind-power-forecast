# mlses-wind-power-forecast
Wind power forecast project for the "Machine Learning for renewable energy systems" (MLSES) seminar in the summer term 2023 at the University of Tübingen.

## Description

The goal of the mlses-wind-power-forecast project is to implement a Machine Learning method to forecast the power output of a wind turbine of at least one turbine for each of the two considered wind farms, one in Kelmarsh, UK and one in Brazil. Three different time horizons are to be considered for the forecast: next step (10 minutes), next hour and next day.

For this, after downloading the necessary data the following steps are performed:
1. [All available data for one UK turbine and one Brazil turbine is loaded.](./src/main.py#L17)
2. [Both dataframes are preprocessed](./src/main.py#L26) according to what is defined in the respective data preprocessor classes for [UK](./src/data/data_preprocessor.py#L164) and [Brazil](./src/data/data_preprocessor.py#L197).
3. [Feature engineering is performed](./src/main.py#L37) where new features are created (e.g. rolling window statistics, lag features, etc.).
4. [Dataframes are split for training, validating and testing the models.](./src/main.py#L56)
5. [The feature and label column names for each dataframe are identified.](./src/main.py#L75)
6. [Dataframes are split into feature and label columns.](./src/main.py#L86)
7. [All 6 models (three for each UK and Brazil) are trained and validated.](./src/main.py#L102)
8. [Predictions for all horizons are made.](./src/main.py#L109)
9. [Models and predictions are evaluated.](./src/main.py#L116)
10. [Plots and models are saved.](./src/main.py#L123)

### Structure
The structure of the project is the following:
- **data**: contains raw, preprocessed data for the two datasets UK and Brazil as well as trained models
- **docs**: additional documentation
- **notebooks**: jupyter notebooks for explorative data analysis and ML prototyping and experimenting
- **src**: main source code
    - **data**: data related code, e.g. loading/saving data, general preprocessing, etc.
    - **models**: model related code, e.g. for UK XGBoost and Brazil XGBoost regression models
    - **utils**: util code, e.g. configuration parameters and logging
    - main.py: main source code

For evaluation, please mainly consider the source code from the [src folder](./src/) as the [notebooks folder](./notebooks/) is meant for exploration and prototyping.

## Setup

0. Download necessary data for [UK](https://zenodo.org/record/5841834#.ZEajKXbP2BQ) and [Brazil](https://zenodo.org/record/1475197#.ZD6iMxXP2WC) wind farm.

1. Clone this repository and change into the mlses-wind-power-forecast directory
```
git clone https://github.com/lenare/mlses-wind-power-forecast.git
cd mlses-wind-power-forecast
```

2. Create a python virtual environment and activate it
```
python3 -m venv venv
. ./venv/bin/activate
```
and install the necessary libraries via pip.
```
pip install -r requirements.txt
```

3. Then change in the src directory and run the main script
```
cd src
python main.py
```
Several plot and evaluation md files should be created where you can see the results.

NOTE: You may still need to install [graphviz](https://www.graphviz.org/) if it isn't already installed on your system!

## TODOs
- Add more documentation! (especially results and evaluation, also provide explanation for why Brazil model performs better)
- Improve feature selection!
- Maybe try another model (LSTM)

## Results
With the XGBoost model the benchmark for all horizons for the [UK wind farm](./src/XGBOOST_uk_evaluation.md) is outperformed slightly. The XGBoost model performs even better for the [Brazil wind farm](./src/XGBOOST_brazil_evaluation.md), especially also on the longer horizons. 


## Further improvements
There are several possibilities that can be evaluated to further improve the wind power forecast. These include:
- testing different models
- consider additional data (e.g. wind data)
- adapt feature selection/engineering further (e.g. try different rolling window statistics)
