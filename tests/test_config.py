import os
import pytest
from typing import Dict
from core.configuration import Config


@pytest.fixture
def config() -> Config:
    """Fixture to initialize Config for tests."""
    return Config()


# ==========================================================
# ================ TEST: DEFAULT CONFIGURATION ==============
# ==========================================================

def test_default_configuration_structure(config: Config):
    """Test that the default configuration has expected top-level sections."""
    expected_keys = {'paths', 'preprocessing', 'model'}
    assert expected_keys.issubset(config.config.keys()), "Missing one or more top-level keys in config."


def test_default_paths_exist_in_config(config: Config):
    """Test that all expected dataset paths are properly defined."""
    paths: Dict[str, str] = config.config['paths']
    expected_path_keys = {'root_dir', 'train_images', 'test_images', 'train_csv', 'test_csv'}

    assert expected_path_keys == set(paths.keys()), "Paths keys do not match expected keys."

    # Validate that paths are constructed correctly
    root_dir = paths['root_dir']
    for key, path in paths.items():
        assert path.startswith(root_dir), f"Path '{key}' does not start with root_dir."


def test_preprocessing_defaults(config: Config):
    """Test that preprocessing section contains expected defaults."""
    preprocessing = config.config['preprocessing']
    assert preprocessing['target_size'] == [512, 512], "Default target_size mismatch."
    assert preprocessing['default_density'] == 'B', "Default density code mismatch."
    assert isinstance(preprocessing['density_categories'], dict), "Density categories should be a dict."


def test_model_defaults(config: Config):
    """Test that model section contains correct structure and defaults."""
    model = config.config['model']
    assert model['input_shape'] == [512, 512, 1], "Model input shape mismatch."
    assert model['batch_size'] == 32, "Batch size mismatch."


# ==========================================================
# ================ TEST: DENSITY CATEGORIES =================
# ==========================================================

def test_density_categories_property(config: Config):
    """Test that density_categories property returns correct mapping."""
    expected_keys = {'A', 'B', 'C', 'D'}
    density_map = config.density_categories
    assert expected_keys == set(density_map.keys()), "Density categories do not match expected keys."


@pytest.mark.parametrize("density,expected", [
    ('A', 'Almost entirely fatty'),
    ('B', 'Scattered fibroglandular densities'),
    ('C', 'Heterogeneously dense'),
    ('D', 'Extremely dense'),
])
def test_get_density_description_valid(config: Config, density: str, expected: str):
    """Test that valid density codes return the correct description."""
    assert config.get_density_description(density) == expected


def test_get_density_description_invalid(config: Config):
    """Test that invalid density code returns 'Unknown density'."""
    assert config.get_density_description('Z') == 'Unknown density'
    assert config.get_density_description('') == 'Unknown density'
    assert config.get_density_description(None) == 'Unknown density'
