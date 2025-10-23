"""
Tests for the football prediction Streamlit app.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model import FootballPredictor, create_sample_data


class TestAppComponents:
    """Test cases for app components and functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.predictor = FootballPredictor()
        self.sample_data = create_sample_data()

    def test_sample_data_structure(self):
        """Test that sample data has correct structure for app."""
        data = self.sample_data

        # Check required columns for app
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
            assert col in data.columns, f"Missing column: {col}"

        # Check data types are appropriate for Streamlit widgets
        assert pd.api.types.is_numeric_dtype(data["home_goals_avg"])
        assert pd.api.types.is_numeric_dtype(data["away_goals_avg"])
        assert pd.api.types.is_numeric_dtype(data["home_possession"])
        assert pd.api.types.is_numeric_dtype(data["away_possession"])
        assert pd.api.types.is_numeric_dtype(data["home_shots_avg"])
        assert pd.api.types.is_numeric_dtype(data["away_shots_avg"])
        assert pd.api.types.is_numeric_dtype(data["home_form"])
        assert pd.api.types.is_numeric_dtype(data["away_form"])
        assert pd.api.types.is_numeric_dtype(data["temperature"])
        assert pd.api.types.is_numeric_dtype(data["venue_capacity"])

    def test_prediction_data_validation(self):
        """Test that prediction data is properly validated."""
        # Valid match data
        valid_match_data = {
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

        # Test prediction works
        self.predictor._create_simple_model()
        result = self.predictor.predict_match(valid_match_data)

        # Check result has expected structure for app
        assert "outcome" in result
        assert "home_win_prob" in result
        assert "draw_prob" in result
        assert "away_win_prob" in result
        assert "predicted_score" in result
        assert "confidence" in result

        # Check probabilities are valid for display
        assert 0 <= result["home_win_prob"] <= 1
        assert 0 <= result["draw_prob"] <= 1
        assert 0 <= result["away_win_prob"] <= 1
        assert 0 <= result["confidence"] <= 1

        # Check score is valid
        score = result["predicted_score"]
        assert isinstance(score["home_goals"], int)
        assert isinstance(score["away_goals"], int)
        assert score["home_goals"] >= 0
        assert score["away_goals"] >= 0

    def test_batch_prediction_structure(self):
        """Test batch prediction returns data suitable for app display."""
        # Create sample batch data
        sample_data = self.sample_data.head(5)

        # Get predictions
        self.predictor._create_simple_model()
        results = self.predictor.predict_batch(sample_data)

        # Check structure is suitable for app
        assert isinstance(results, dict)
        assert "outcomes" in results
        assert "home_win_probs" in results
        assert "draw_probs" in results
        assert "away_win_probs" in results

        # Check all lists have same length
        n_predictions = len(sample_data)
        assert len(results["outcomes"]) == n_predictions
        assert len(results["home_win_probs"]) == n_predictions
        assert len(results["draw_probs"]) == n_predictions
        assert len(results["away_win_probs"]) == n_predictions

        # Check probabilities are valid
        for i in range(n_predictions):
            assert 0 <= results["home_win_probs"][i] <= 1
            assert 0 <= results["draw_probs"][i] <= 1
            assert 0 <= results["away_win_probs"][i] <= 1

            # Check probabilities sum to 1
            total_prob = (
                results["home_win_probs"][i]
                + results["draw_probs"][i]
                + results["away_win_probs"][i]
            )
            assert abs(total_prob - 1.0) < 0.01

    def test_team_list_consistency(self):
        """Test that team lists are consistent across app."""
        # Teams mentioned in app
        app_teams = [
            "Real Madrid",
            "Barcelona",
            "Manchester United",
            "Liverpool",
            "Bayern Munich",
            "PSG",
            "Juventus",
            "AC Milan",
        ]

        # Check sample data contains these teams
        data = self.sample_data
        unique_teams = set(data["home_team"].unique()) | set(data["away_team"].unique())

        # At least some app teams should be in sample data
        overlap = set(app_teams) & unique_teams
        assert len(overlap) > 0, "Sample data should contain some teams from app"

    def test_weather_conditions(self):
        """Test weather condition handling."""
        # Test different weather conditions
        weather_conditions = ["Sunny", "Rainy", "Cloudy", "Windy"]

        for weather in weather_conditions:
            match_data = {
                "home_goals_avg": 2.1,
                "away_goals_avg": 1.8,
                "home_possession": 55,
                "away_possession": 45,
                "home_shots_avg": 12,
                "away_shots_avg": 10,
                "home_form": 10,
                "away_form": 8,
                "weather_rainy": 1 if weather == "Rainy" else 0,
                "weather_windy": 1 if weather == "Windy" else 0,
                "temperature": 20,
                "venue_capacity": 60000,
            }

            self.predictor._create_simple_model()
            result = self.predictor.predict_match(match_data)

            # Should work for all weather conditions
            assert isinstance(result, dict)
            assert "outcome" in result

    def test_temperature_range(self):
        """Test temperature range handling."""
        # Test extreme temperatures
        temperatures = [-10, 0, 20, 40]

        for temp in temperatures:
            match_data = {
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
                "temperature": temp,
                "venue_capacity": 60000,
            }

            self.predictor._create_simple_model()
            result = self.predictor.predict_match(match_data)

            # Should work for all temperatures
            assert isinstance(result, dict)
            assert "outcome" in result

    def test_venue_capacity_range(self):
        """Test venue capacity range handling."""
        # Test different venue sizes
        capacities = [10000, 50000, 100000, 200000]

        for capacity in capacities:
            match_data = {
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
                "venue_capacity": capacity,
            }

            self.predictor._create_simple_model()
            result = self.predictor.predict_match(match_data)

            # Should work for all capacities
            assert isinstance(result, dict)
            assert "outcome" in result

    def test_form_range(self):
        """Test form range handling."""
        # Test different form values
        form_values = [0, 5, 10, 15]

        for form in form_values:
            match_data = {
                "home_goals_avg": 2.1,
                "away_goals_avg": 1.8,
                "home_possession": 55,
                "away_possession": 45,
                "home_shots_avg": 12,
                "away_shots_avg": 10,
                "home_form": form,
                "away_form": form,
                "weather_rainy": 0,
                "weather_windy": 0,
                "temperature": 20,
                "venue_capacity": 60000,
            }

            self.predictor._create_simple_model()
            result = self.predictor.predict_match(match_data)

            # Should work for all form values
            assert isinstance(result, dict)
            assert "outcome" in result


class TestDataValidation:
    """Test data validation for app inputs."""

    def test_csv_upload_validation(self):
        """Test CSV upload data validation."""
        # Create valid CSV data
        valid_data = pd.DataFrame(
            {
                "home_team": ["Real Madrid", "Barcelona"],
                "away_team": ["Barcelona", "Real Madrid"],
                "home_goals_avg": [2.1, 1.8],
                "away_goals_avg": [1.8, 2.1],
                "home_possession": [55, 45],
                "away_possession": [45, 55],
                "home_shots_avg": [12, 10],
                "away_shots_avg": [10, 12],
                "home_form": [10, 8],
                "away_form": [8, 10],
                "weather_rainy": [0, 1],
                "weather_windy": [0, 0],
                "temperature": [20, 15],
                "venue_capacity": [60000, 55000],
            }
        )

        # Check data is valid for prediction
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
            assert col in valid_data.columns
            assert not valid_data[col].isna().any()

    def test_missing_columns_handling(self):
        """Test handling of missing columns in uploaded data."""
        # Create data with missing columns
        incomplete_data = pd.DataFrame(
            {
                "home_team": ["Real Madrid"],
                "away_team": ["Barcelona"],
                "home_goals_avg": [2.1],
                # Missing other required columns
            }
        )

        # Should raise KeyError when trying to predict
        predictor = FootballPredictor()
        predictor._create_simple_model()

        with pytest.raises(KeyError):
            predictor.predict_batch(incomplete_data)

    def test_invalid_data_types(self):
        """Test handling of invalid data types."""
        # Create data with wrong types
        invalid_data = pd.DataFrame(
            {
                "home_goals_avg": ["invalid"],  # Should be numeric
                "away_goals_avg": [1.8],
                "home_possession": [55],
                "away_possession": [45],
                "home_shots_avg": [12],
                "away_shots_avg": [10],
                "home_form": [10],
                "away_form": [8],
                "weather_rainy": [0],
                "weather_windy": [0],
                "temperature": [20],
                "venue_capacity": [60000],
            }
        )

        # Should handle gracefully or raise appropriate error
        predictor = FootballPredictor()
        predictor._create_simple_model()

        # This might raise an error depending on implementation
        # The important thing is that it doesn't crash the app
        try:
            results = predictor.predict_batch(invalid_data)
            # If it doesn't raise an error, results should be valid
            assert isinstance(results, dict)
        except (ValueError, TypeError):
            # This is expected for invalid data types
            pass


class TestAppIntegration:
    """Test app integration scenarios."""

    def test_end_to_end_prediction(self):
        """Test complete prediction workflow."""
        # Create predictor
        predictor = FootballPredictor()
        predictor._create_simple_model()

        # Create sample data
        data = create_sample_data()

        # Test single prediction
        match_data = {
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

        single_result = predictor.predict_match(match_data)
        assert isinstance(single_result, dict)

        # Test batch prediction
        batch_data = data.head(3)
        batch_results = predictor.predict_batch(batch_data)
        assert isinstance(batch_results, dict)

        # Test that results can be combined for display
        combined_data = batch_data.copy()
        combined_data["predicted_outcome"] = batch_results["outcomes"]
        combined_data["home_win_prob"] = batch_results["home_win_probs"]
        combined_data["draw_prob"] = batch_results["draw_probs"]
        combined_data["away_win_prob"] = batch_results["away_win_probs"]

        # Should be able to display this data
        assert len(combined_data) == len(batch_data)
        assert "predicted_outcome" in combined_data.columns

    def test_model_performance_metrics(self):
        """Test that model provides reasonable performance metrics."""
        predictor = FootballPredictor()
        predictor._create_simple_model()

        # Test multiple predictions
        data = create_sample_data()
        results = predictor.predict_batch(data)

        # Check that predictions are distributed reasonably
        outcomes = results["outcomes"]
        unique_outcomes = set(outcomes)

        # Should have multiple outcome types
        assert len(unique_outcomes) > 1

        # Check probability distributions
        for i in range(len(data)):
            home_prob = results["home_win_probs"][i]
            draw_prob = results["draw_probs"][i]
            away_prob = results["away_win_probs"][i]

            # Probabilities should be reasonable
            assert 0 <= home_prob <= 1
            assert 0 <= draw_prob <= 1
            assert 0 <= away_prob <= 1

            # Should sum to 1
            total = home_prob + draw_prob + away_prob
            assert abs(total - 1.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__])
