# mlses-wind-power-forecast
Wind power forecast project for the "Machine Learning for renewable energy systems" (MLSES) seminar in the summer term 2023 at the University of Tübingen.

**NOTE:** For evaluation, please mainly consider the source code from the [src folder](./src/) as the [notebooks folder](./notebooks/) is meant for exploration and prototyping.

## Description

The goal of the mlses-wind-power-forecast project is to implement a Machine Learning method to forecast the power output of a wind turbine of at least one turbine for each of the two considered wind farms, one in Kelmarsh, UK and one in Brazil. Three different time horizons are to be considered for the forecast: next step (10 minutes), next hour and next day.

### Structure
The structure of the project is the following:
- **data**: contains raw, preprocessed data for the two datasets UK and Brazil as well as trained models
- **docs**: additional documentation
- **notebooks**: jupyter notebooks for explorative data analysis and ML prototyping and experimenting
- **src**: main source code
    - [**data**](./src/data): data related code, e.g. loading/saving data, general preprocessing, etc.
    - [**models**](./src/models/): model related code, e.g. for UK XGBoost and Brazil XGBoost regression models
    - [**utils**](./src/utils/): util code, e.g. configuration parameters and logging
    - [**main.ipynb**](./src/main.ipynb): main source code as notebook


## Setup

1. Download necessary data for [UK](https://zenodo.org/record/5841834#.ZEajKXbP2BQ) and [Brazil](https://zenodo.org/record/1475197#.ZD6iMxXP2WC) wind farm.

2. Clone this repository and change into the mlses-wind-power-forecast directory
```
git clone https://github.com/lenare/mlses-wind-power-forecast.git
cd mlses-wind-power-forecast
```

3. Create a python virtual environment and activate it
```
python3 -m venv venv
. ./venv/bin/activate
```
and install the necessary libraries via pip.
```
pip install -r requirements.txt
```

4. Then change in the src directory and run the main script
```
cd src
python main.py
```
Several plot and evaluation md files should be created where you can see the results.

**NOTE**: You may still need to install [graphviz](https://www.graphviz.org/) for generating some plots if it isn't already installed on your system!


## Results
Based on the evalutaion metrics MAE and RMSE the presented XGBoost model slightly outperforms the benchmark for all horizons for the UK wind farm. For the Brazil wind farm the benchmark performs a bit better. Below are some suggested improvements to enhance our implementation.


## Further improvements
There are several possibilities that can be evaluated to further improve the wind power forecast. These include:
- improve feature selection
- testing different models (e.g. LSTMs)
- consider additional data (e.g. wind data)
- adapt feature selection/engineering further (e.g. try different rolling window statistics)
