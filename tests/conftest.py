"""
Pytest configuration and fixtures for football prediction tests.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model import FootballPredictor, create_sample_data


@pytest.fixture
def sample_data():
    """Fixture providing sample football data."""
    return create_sample_data()


@pytest.fixture
def trained_predictor():
    """Fixture providing a trained predictor."""
    predictor = FootballPredictor()
    data = create_sample_data()
    data['outcome_numeric'] = data['outcome'].map({
        'Home Win': 0, 'Draw': 1, 'Away Win': 2
    })
    predictor.train_model(data)
    return predictor


@pytest.fixture
def sample_match_data():
    """Fixture providing sample match data for prediction."""
    return {
        'home_team': 'Real Madrid',
        'away_team': 'Barcelona',
        'home_goals_avg': 2.1,
        'away_goals_avg': 1.8,
        'home_possession': 55,
        'away_possession': 45,
        'home_shots_avg': 12,
        'away_shots_avg': 10,
        'home_form': 10,
        'away_form': 8,
        'weather_rainy': 0,
        'weather_windy': 0,
        'temperature': 20,
        'venue_capacity': 60000
    }


@pytest.fixture
def sample_batch_data(sample_data):
    """Fixture providing sample batch data for prediction."""
    return sample_data.head(5)


@pytest.fixture
def invalid_match_data():
    """Fixture providing invalid match data for error testing."""
    return {
        'home_goals_avg': 2.1,
        # Missing other required keys
    }


@pytest.fixture
def extreme_match_data():
    """Fixture providing extreme match data for edge case testing."""
    return {
        'home_goals_avg': 10.0,  # Very high
        'away_goals_avg': 0.1,   # Very low
        'home_possession': 95,   # Very high
        'away_possession': 5,    # Very low
        'home_shots_avg': 50,    # Very high
        'away_shots_avg': 1,     # Very low
        'home_form': 15,         # Maximum
        'away_form': 0,          # Minimum
        'weather_rainy': 1,
        'weather_windy': 1,
        'temperature': -10,      # Very cold
        'venue_capacity': 200000  # Very large
    }


@pytest.fixture
def empty_dataframe():
    """Fixture providing empty DataFrame for error testing."""
    return pd.DataFrame()


@pytest.fixture
def incomplete_dataframe():
    """Fixture providing incomplete DataFrame for error testing."""
    return pd.DataFrame({
        'home_team': ['Real Madrid'],
        'away_team': ['Barcelona'],
        'home_goals_avg': [2.1],
        # Missing other required columns
    })


@pytest.fixture
def invalid_types_dataframe():
    """Fixture providing DataFrame with invalid data types."""
    return pd.DataFrame({
        'home_goals_avg': ['invalid'],  # Should be numeric
        'away_goals_avg': [1.8],
        'home_possession': [55],
        'away_possession': [45],
        'home_shots_avg': [12],
        'away_shots_avg': [10],
        'home_form': [10],
        'away_form': [8],
        'weather_rainy': [0],
        'weather_windy': [0],
        'temperature': [20],
        'venue_capacity': [60000]
    })


@pytest.fixture
def weather_conditions():
    """Fixture providing different weather conditions."""
    return ["Sunny", "Rainy", "Cloudy", "Windy"]


@pytest.fixture
def temperature_range():
    """Fixture providing temperature range for testing."""
    return [-10, 0, 20, 40]


@pytest.fixture
def venue_capacities():
    """Fixture providing venue capacity range for testing."""
    return [10000, 50000, 100000, 200000]


@pytest.fixture
def form_values():
    """Fixture providing form values for testing."""
    return [0, 5, 10, 15]


@pytest.fixture
def team_names():
    """Fixture providing team names for testing."""
    return [
        "Real Madrid", "Barcelona", "Manchester United", "Liverpool",
        "Bayern Munich", "PSG", "Juventus", "AC Milan", "Chelsea", "Arsenal"
    ]


@pytest.fixture
def valid_outcomes():
    """Fixture providing valid match outcomes."""
    return ['Home Win', 'Draw', 'Away Win']


@pytest.fixture
def required_columns():
    """Fixture providing required columns for match data."""
    return [
        'home_goals_avg', 'away_goals_avg', 'home_possession', 'away_possession',
        'home_shots_avg', 'away_shots_avg', 'home_form', 'away_form',
        'weather_rainy', 'weather_windy', 'temperature', 'venue_capacity'
    ]


@pytest.fixture
def numeric_columns():
    """Fixture providing numeric columns for testing."""
    return [
        'home_goals_avg', 'away_goals_avg', 'home_possession', 'away_possession',
        'home_shots_avg', 'away_shots_avg', 'home_form', 'away_form',
        'temperature', 'venue_capacity'
    ]


@pytest.fixture
def binary_columns():
    """Fixture providing binary columns for testing."""
    return ['weather_rainy', 'weather_windy']


@pytest.fixture
def string_columns():
    """Fixture providing string columns for testing."""
    return ['home_team', 'away_team', 'outcome']


# Pytest configuration
def pytest_configure(config):
    """Configure pytest settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add unit marker to model tests
        if "test_football_model" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to app tests
        if "test_football_app" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to tests that might be slow
        if "test_create_and_save_model" in item.nodeid:
            item.add_marker(pytest.mark.slow)
