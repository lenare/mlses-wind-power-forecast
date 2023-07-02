# mlses-wind-power-forecast
Wind power forecast project for the "Machine Learning for renewable energy systems" (MLSES) seminar in the summer term 2023 at the University of Tübingen.

## Description

The goal of the mlses-wind-power-forecast project is to implement a Machine Learning method to forecast the power output of a wind turbine of at least one turbine for each of the two considered wind farms, one in Kelmarsh, UK and one in Brazil. Three different time horizons are to be considered for the forecast: next step (10 minutes), next hour and next day.

### Structure
The structure of the project is the following:
- **data**: contains raw, preprocessed data for the two datasets UK and Brazil as well as trained models
- **docs**: additional documentation
- **notebooks**: jupyter notebooks for exploratitve data analysis and ML prototyping and experimenting
- **src**: main source code
    - **data**: data related code, e.g. loading/saving data, general preprocessing, etc.
    - **models**: model related code, e.g. for UK XGBoost and Brazil XGBoost regression models
    - **utils**: util code, e.g. configuration parameters and logging
    - main.py: main source code

For further, more in-detail documentation 

## Setup

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

NOTE: You may still need to install [graphviz](https://www.graphviz.org/) if it isn't already installed on your system!

## TODOs
- Add more documentation! (especially results and evaluation)
- Improve feature selection!
- Maybe try another model
