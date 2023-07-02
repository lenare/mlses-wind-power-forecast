''' Class for im- and exporting data from/to a file path. '''
import pandas as pd
import xarray as xr


class DataLoader:
    def load_data(self, file_path: str | list, **kwargs) -> pd.DataFrame() | xr.Dataset():
        # Code for loading csv data from file_path into a single pandas dataframe
        if isinstance(file_path, list):
            if len(file_path) < 1:
                raise ValueError("List of file paths is empty!")
            if file_path[0].endswith(".csv"):
                return pd.concat([pd.read_csv(file_path, **kwargs) for file_path in file_path])
        else:
            if file_path.endswith(".csv"):
                return pd.read_csv(file_path, **kwargs)
            elif file_path.endswith(".nc"):
                return xr.open_dataset(file_path)
            else:
                raise ValueError(
                    f"File extension not supported! Please use .csv or .nc files.")

    def save_data(self, df: pd.DataFrame(), csv_file_path: str, **kwargs) -> str:
        # Code for saving csv data to file_path
        df.to_csv(csv_file_path, **kwargs)
        return csv_file_path
