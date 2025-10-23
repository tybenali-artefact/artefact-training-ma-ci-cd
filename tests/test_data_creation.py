"""
Tests for data creation and validation functions.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model import create_sample_data


class TestDataCreation:
    """Test cases for data creation functions."""

    def test_create_sample_data_basic(self):
        """Test basic sample data creation."""
        data = create_sample_data()

        # Check basic structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert len(data.columns) > 0

        # Check it's not empty
        assert not data.empty

    def test_create_sample_data_columns(self):
        """Test that sample data has all required columns."""
        data = create_sample_data()

        required_columns = [
            "home_team",
            "away_team",
            "home_goals_avg",
            "away_goals_avg",
            "home_possession",
            "away_possession",
            "home_shots_avg",
            "away_shots_avg",
            "home_form",
            "away_form",
            "weather_rainy",
            "weather_windy",
            "temperature",
            "venue_capacity",
            "outcome",
        ]

        for col in required_columns:
            assert col in data.columns, f"Missing required column: {col}"

    def test_create_sample_data_types(self):
        """Test that sample data has correct data types."""
        data = create_sample_data()

        # Check numeric columns
        numeric_columns = [
            "home_goals_avg",
            "away_goals_avg",
            "home_possession",
            "away_possession",
            "home_shots_avg",
            "away_shots_avg",
            "home_form",
            "away_form",
            "temperature",
            "venue_capacity",
        ]

        for col in numeric_columns:
            assert pd.api.types.is_numeric_dtype(data[col]), f"Column {col} should be numeric"

        # Check binary columns
        binary_columns = ["weather_rainy", "weather_windy"]
        for col in binary_columns:
            assert data[col].isin([0, 1]).all(), f"Column {col} should contain only 0 and 1"

        # Check string columns
        string_columns = ["home_team", "away_team", "outcome"]
        for col in string_columns:
            assert pd.api.types.is_object_dtype(data[col]), f"Column {col} should be string/object"

    def test_create_sample_data_ranges(self):
        """Test that sample data values are within reasonable ranges."""
        data = create_sample_data()

        # Check goals averages
        assert data["home_goals_avg"].min() >= 0
        assert data["away_goals_avg"].min() >= 0
        assert data["home_goals_avg"].max() <= 5  # Reasonable upper bound
        assert data["away_goals_avg"].max() <= 5

        # Check possession percentages
        assert data["home_possession"].min() >= 0
        assert data["home_possession"].max() <= 100
        assert data["away_possession"].min() >= 0
        assert data["away_possession"].max() <= 100

        # Check shots
        assert data["home_shots_avg"].min() >= 0
        assert data["away_shots_avg"].min() >= 0
        assert data["home_shots_avg"].max() <= 30  # Reasonable upper bound
        assert data["away_shots_avg"].max() <= 30

        # Check form (points from last 5 games)
        assert data["home_form"].min() >= 0
        assert data["home_form"].max() <= 15  # Maximum 15 points from 5 games
        assert data["away_form"].min() >= 0
        assert data["away_form"].max() <= 15

        # Check temperature
        assert data["temperature"].min() >= -20  # Very cold
        assert data["temperature"].max() <= 50  # Very hot

        # Check venue capacity
        assert data["venue_capacity"].min() >= 1000  # Minimum reasonable capacity
        assert data["venue_capacity"].max() <= 200000  # Maximum reasonable capacity

    def test_create_sample_data_outcomes(self):
        """Test that sample data outcomes are valid."""
        data = create_sample_data()

        # Check valid outcomes
        valid_outcomes = ["Home Win", "Draw", "Away Win"]
        unique_outcomes = set(data["outcome"].unique())

        for outcome in unique_outcomes:
            assert outcome in valid_outcomes, f"Invalid outcome: {outcome}"

        # Check that we have multiple outcome types
        assert len(unique_outcomes) > 1, "Should have multiple outcome types"

    def test_create_sample_data_teams(self):
        """Test that sample data has valid team names."""
        data = create_sample_data()

        # Check that teams are not empty
        assert not data["home_team"].isna().any()
        assert not data["away_team"].isna().any()

        # Check that teams are strings
        assert data["home_team"].dtype == "object"
        assert data["away_team"].dtype == "object"

        # Check that we have multiple teams
        unique_home_teams = set(data["home_team"].unique())
        unique_away_teams = set(data["away_team"].unique())

        assert len(unique_home_teams) > 1, "Should have multiple home teams"
        assert len(unique_away_teams) > 1, "Should have multiple away teams"

    def test_create_sample_data_consistency(self):
        """Test that sample data is internally consistent."""
        data = create_sample_data()

        # Check that home and away teams are different in each match
        same_teams = data["home_team"] == data["away_team"]
        assert not same_teams.any(), "Home and away teams should be different"

        # Check that possession percentages are reasonable
        # (They don't need to sum to 100 as they're independent stats)
        assert data["home_possession"].min() >= 0
        assert data["away_possession"].min() >= 0

    def test_create_sample_data_size(self):
        """Test that sample data has reasonable size."""
        data = create_sample_data()

        # Check minimum size
        assert len(data) >= 10, "Should have at least 10 matches"

        # Check maximum size (not too large for testing)
        assert len(data) <= 100, "Should not be too large for testing"

    def test_create_sample_data_reproducibility(self):
        """Test that sample data creation is reproducible."""
        # Set seed before creating data
        np.random.seed(42)
        data1 = create_sample_data()

        np.random.seed(42)
        data2 = create_sample_data()

        # Should be identical with same seed
        pd.testing.assert_frame_equal(data1, data2)

    def test_create_sample_data_no_nan(self):
        """Test that sample data has no NaN values."""
        data = create_sample_data()

        # Check for NaN values
        assert not data.isna().any().any(), "Sample data should not contain NaN values"

    def test_create_sample_data_duplicates(self):
        """Test that sample data doesn't have exact duplicates."""
        data = create_sample_data()

        # Check for exact duplicates
        duplicates = data.duplicated()
        assert not duplicates.any(), "Sample data should not have exact duplicates"

    def test_create_sample_data_correlation(self):
        """Test that sample data has reasonable correlations."""
        data = create_sample_data()

        # Check that goals and shots are positively correlated
        goals_shots_corr = data["home_goals_avg"].corr(data["home_shots_avg"])
        assert goals_shots_corr > 0, "Goals and shots should be positively correlated"

        # Check that possession and shots are positively correlated
        possession_shots_corr = data["home_possession"].corr(data["home_shots_avg"])
        assert possession_shots_corr > 0, "Possession and shots should be positively correlated"

    def test_create_sample_data_outcome_distribution(self):
        """Test that sample data has reasonable outcome distribution."""
        data = create_sample_data()

        # Get outcome counts
        outcome_counts = data["outcome"].value_counts()

        # Should have all three outcomes
        assert "Home Win" in outcome_counts.index
        assert "Draw" in outcome_counts.index
        assert "Away Win" in outcome_counts.index

        # No single outcome should dominate (more than 80%)
        total_matches = len(data)
        for outcome in outcome_counts.index:
            proportion = outcome_counts[outcome] / total_matches
            assert proportion < 0.8, f"Outcome {outcome} should not dominate (>80%)"

    def test_create_sample_data_weather_conditions(self):
        """Test that sample data has reasonable weather conditions."""
        data = create_sample_data()

        # Check weather binary columns
        assert data["weather_rainy"].isin([0, 1]).all()
        assert data["weather_windy"].isin([0, 1]).all()

        # Check that not all matches have bad weather
        all_rainy = data["weather_rainy"].all()
        all_windy = data["weather_windy"].all()

        assert not all_rainy, "Not all matches should be rainy"
        assert not all_windy, "Not all matches should be windy"

    def test_create_sample_data_form_distribution(self):
        """Test that sample data has reasonable form distribution."""
        data = create_sample_data()

        # Check form ranges
        assert data["home_form"].min() >= 0
        assert data["home_form"].max() <= 15
        assert data["away_form"].min() >= 0
        assert data["away_form"].max() <= 15

        # Check that we have variety in form
        unique_home_form = len(data["home_form"].unique())
        unique_away_form = len(data["away_form"].unique())

        assert unique_home_form > 1, "Should have variety in home form"
        assert unique_away_form > 1, "Should have variety in away form"


class TestDataValidation:
    """Test data validation functions."""

    def test_validate_match_data_structure(self):
        """Test validation of match data structure."""
        # Valid data
        valid_data = {
            "home_team": "Real Madrid",
            "away_team": "Barcelona",
            "home_goals_avg": 2.1,
            "away_goals_avg": 1.8,
            "home_possession": 55,
            "away_possession": 45,
            "home_shots_avg": 12,
            "away_shots_avg": 10,
            "home_form": 10,
            "away_form": 8,
            "weather_rainy": 0,
            "weather_windy": 0,
            "temperature": 20,
            "venue_capacity": 60000,
        }

        # Should be valid
        required_keys = [
            "home_goals_avg",
            "away_goals_avg",
            "home_possession",
            "away_possession",
            "home_shots_avg",
            "away_shots_avg",
            "home_form",
            "away_form",
            "weather_rainy",
            "weather_windy",
            "temperature",
            "venue_capacity",
        ]

        for key in required_keys:
            assert key in valid_data, f"Missing required key: {key}"
            assert isinstance(valid_data[key], (int, float)), f"Key {key} should be numeric"

    def test_validate_dataframe_structure(self):
        """Test validation of DataFrame structure."""
        data = create_sample_data()

        # Check required columns exist
        required_columns = [
            "home_goals_avg",
            "away_goals_avg",
            "home_possession",
            "away_possession",
            "home_shots_avg",
            "away_shots_avg",
            "home_form",
            "away_form",
            "weather_rainy",
            "weather_windy",
            "temperature",
            "venue_capacity",
        ]

        for col in required_columns:
            assert col in data.columns, f"Missing required column: {col}"
            assert not data[col].isna().any(), f"Column {col} should not have NaN values"

    def test_validate_numeric_ranges(self):
        """Test validation of numeric ranges."""
        data = create_sample_data()

        # Check goals are non-negative
        assert (data["home_goals_avg"] >= 0).all()
        assert (data["away_goals_avg"] >= 0).all()

        # Check possession is between 0 and 100
        assert (data["home_possession"] >= 0).all()
        assert (data["home_possession"] <= 100).all()
        assert (data["away_possession"] >= 0).all()
        assert (data["away_possession"] <= 100).all()

        # Check shots are non-negative
        assert (data["home_shots_avg"] >= 0).all()
        assert (data["away_shots_avg"] >= 0).all()

        # Check form is between 0 and 15
        assert (data["home_form"] >= 0).all()
        assert (data["home_form"] <= 15).all()
        assert (data["away_form"] >= 0).all()
        assert (data["away_form"] <= 15).all()

        # Check binary weather columns
        assert data["weather_rainy"].isin([0, 1]).all()
        assert data["weather_windy"].isin([0, 1]).all()

        # Check venue capacity is positive
        assert (data["venue_capacity"] > 0).all()


if __name__ == "__main__":
    pytest.main([__file__])
