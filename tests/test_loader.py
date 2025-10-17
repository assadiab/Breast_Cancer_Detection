import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from core.loader import Loader  # remplacer par le chemin réel
from core.configuration import Config


# Création d'un Config mock
class MockConfig:
    train_csv_path = "train.csv"
    test_csv_path = "test.csv"


# Fixtures pour pandas DataFrame simulés
@pytest.fixture
def mock_train_df():
    return pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})


@pytest.fixture
def mock_test_df():
    return pd.DataFrame({"colA": [5, 6], "colB": [7, 8]})


# Test Loader.load_dataframes
def test_load_dataframes(mock_train_df, mock_test_df):
    loader = Loader(config=MockConfig())

    with patch("pandas.read_csv", side_effect=[mock_train_df, mock_test_df]) as mock_read:
        loader.load_dataframes()

        # Vérifie que pd.read_csv a été appelé avec les bons chemins
        mock_read.assert_any_call(MockConfig.train_csv_path)
        mock_read.assert_any_call(MockConfig.test_csv_path)

        # Vérifie que les DataFrames sont bien assignés
        pd.testing.assert_frame_equal(loader.train_df, mock_train_df)
        pd.testing.assert_frame_equal(loader.test_df, mock_test_df)


# Test get_train_df charge si None
def test_get_train_df_calls_load(mock_train_df, mock_test_df):
    loader = Loader(config=MockConfig())

    with patch("pandas.read_csv", side_effect=[mock_train_df, mock_test_df]):
        df = loader.get_train_df()
        pd.testing.assert_frame_equal(df, mock_train_df)


# Test get_test_df charge si None
def test_get_test_df_calls_load(mock_train_df, mock_test_df):
    loader = Loader(config=MockConfig())

    with patch("pandas.read_csv", side_effect=[mock_train_df, mock_test_df]):
        df = loader.get_test_df()
        pd.testing.assert_frame_equal(df, mock_test_df)


# Test gestion d'erreur
def test_load_dataframes_exception(monkeypatch):
    loader = Loader(config=MockConfig())

    def raise_error(path):
        raise FileNotFoundError(f"{path} not found")

    monkeypatch.setattr(pd, "read_csv", raise_error)

    loader.load_dataframes()
    # Comme on print, pas de crash attendu
