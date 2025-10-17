from core.configuration import Config
from typing import Optional
import pandas as pd

class Loader:
    """
    Handles data loading operations using a Config instance.
    """

    def __init__(self, config: Config):
        self.config = config
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

    def load_dataframes(self) -> None:
        """Loads train and test CSV files into memory."""
        try:
            self.train_df = pd.read_csv(self.config.train_csv_path)
            self.test_df = pd.read_csv(self.config.test_csv_path)
            print(f"Data loaded successfully - Train: {len(self.train_df)} rows, Test: {len(self.test_df)} rows")
        except Exception as e:
            print(f"Error while loading dataframes: {e}")

    def get_train_df(self) -> pd.DataFrame:
        if self.train_df is None:
            self.load_dataframes()
        return self.train_df

    def get_test_df(self) -> pd.DataFrame:
        if self.test_df is None:
            self.load_dataframes()
        return self.test_df