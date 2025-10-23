"""
Tests for the football prediction model.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model import FootballPredictor, create_sample_data


class TestFootballPredictor:
    """Test cases for FootballPredictor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.predictor = FootballPredictor()
        self.sample_data = create_sample_data()

    def test_predictor_initialization(self):
        """Test predictor initialization."""
        assert self.predictor.model is None
        assert self.predictor.scaler is not None
        assert len(self.predictor.feature_columns) == 12
        assert self.predictor.outcome_mapping == {0: "Home Win", 1: "Draw", 2: "Away Win"}

    def test_train_model(self):
        """Test model training."""
        # Prepare training data
        training_data = self.sample_data.copy()
        training_data["outcome_numeric"] = training_data["outcome"].map(
            {"Home Win": 0, "Draw": 1, "Away Win": 2}
        )

        # Train model
        self.predictor.train_model(training_data)

        # Check model is trained
        assert self.predictor.model is not None
        assert hasattr(self.predictor.model, "predict")
        assert hasattr(self.predictor.model, "predict_proba")

    def test_predict_match(self):
        """Test single match prediction."""
        # Create sample match data
        match_data = {
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

        # Test prediction
        result = self.predictor.predict_match(match_data)

        # Check result structure
        assert isinstance(result, dict)
        assert "outcome" in result
        assert "home_win_prob" in result
        assert "draw_prob" in result
        assert "away_win_prob" in result
        assert "predicted_score" in result
        assert "confidence" in result

        # Check outcome is valid
        assert result["outcome"] in ["Home Win", "Draw", "Away Win"]

        # Check probabilities sum to 1
        total_prob = result["home_win_prob"] + result["draw_prob"] + result["away_win_prob"]
        assert abs(total_prob - 1.0) < 0.01

        # Check confidence is between 0 and 1
        assert 0 <= result["confidence"] <= 1

        # Check predicted score structure
        score = result["predicted_score"]
        assert "home_goals" in score
        assert "away_goals" in score
        assert isinstance(score["home_goals"], int)
        assert isinstance(score["away_goals"], int)
        assert score["home_goals"] >= 0
        assert score["away_goals"] >= 0

    def test_predict_batch(self):
        """Test batch prediction."""
        # Create sample batch data
        batch_data = self.sample_data.head(5)

        # Test batch prediction
        results = self.predictor.predict_batch(batch_data)

        # Check result structure
        assert isinstance(results, dict)
        assert "outcomes" in results
        assert "home_win_probs" in results
        assert "draw_probs" in results
        assert "away_win_probs" in results

        # Check all lists have same length
        n_predictions = len(batch_data)
        assert len(results["outcomes"]) == n_predictions
        assert len(results["home_win_probs"]) == n_predictions
        assert len(results["draw_probs"]) == n_predictions
        assert len(results["away_win_probs"]) == n_predictions

        # Check outcomes are valid
        for outcome in results["outcomes"]:
            assert outcome in ["Home Win", "Draw", "Away Win"]

        # Check probabilities are valid
        for i in range(n_predictions):
            total_prob = (
                results["home_win_probs"][i]
                + results["draw_probs"][i]
                + results["away_win_probs"][i]
            )
            assert abs(total_prob - 1.0) < 0.01

    def test_create_simple_model(self):
        """Test simple model creation."""
        # Ensure no model exists
        self.predictor.model = None

        # Create simple model
        self.predictor._create_simple_model()

        # Check model is created
        assert self.predictor.model is not None
        assert hasattr(self.predictor.model, "predict")
        assert hasattr(self.predictor.model, "predict_proba")

    def test_save_and_load_model(self, tmp_path):
        """Test model saving and loading."""
        # Train a model first
        training_data = self.sample_data.copy()
        training_data["outcome_numeric"] = training_data["outcome"].map(
            {"Home Win": 0, "Draw": 1, "Away Win": 2}
        )
        self.predictor.train_model(training_data)

        # Save model
        model_path = tmp_path / "test_model.pkl"
        self.predictor.save_model(str(model_path))

        # Check file exists
        assert model_path.exists()

        # Create new predictor and load model
        new_predictor = FootballPredictor()
        new_predictor.load_model(str(model_path))

        # Check model is loaded
        assert new_predictor.model is not None
        assert new_predictor.scaler is not None
        assert new_predictor.feature_columns == self.predictor.feature_columns
        assert new_predictor.outcome_mapping == self.predictor.outcome_mapping

    def test_predict_match_with_missing_model(self):
        """Test prediction when no model is loaded."""
        # Ensure no model
        self.predictor.model = None

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

        # Should create simple model and predict
        result = self.predictor.predict_match(match_data)
        assert isinstance(result, dict)
        assert "outcome" in result

    def test_predict_batch_with_missing_model(self):
        """Test batch prediction when no model is loaded."""
        # Ensure no model
        self.predictor.model = None

        batch_data = self.sample_data.head(3)

        # Should create simple model and predict
        results = self.predictor.predict_batch(batch_data)
        assert isinstance(results, dict)
        assert "outcomes" in results


class TestSampleDataCreation:
    """Test cases for sample data creation."""

    def test_create_sample_data(self):
        """Test sample data creation."""
        data = create_sample_data()

        # Check data is DataFrame
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0

        # Check required columns exist
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
            assert col in data.columns

        # Check data types
        assert data["home_goals_avg"].dtype in ["float64", "int64"]
        assert data["away_goals_avg"].dtype in ["float64", "int64"]
        assert data["home_possession"].dtype in ["float64", "int64"]
        assert data["away_possession"].dtype in ["float64", "int64"]
        assert data["home_shots_avg"].dtype in ["float64", "int64"]
        assert data["away_shots_avg"].dtype in ["float64", "int64"]
        assert data["home_form"].dtype in ["float64", "int64"]
        assert data["away_form"].dtype in ["float64", "int64"]
        assert data["weather_rainy"].dtype in ["float64", "int64"]
        assert data["weather_windy"].dtype in ["float64", "int64"]
        assert data["temperature"].dtype in ["float64", "int64"]
        assert data["venue_capacity"].dtype in ["float64", "int64"]

        # Check outcomes are valid
        valid_outcomes = ["Home Win", "Draw", "Away Win"]
        for outcome in data["outcome"].unique():
            assert outcome in valid_outcomes

        # Check numeric ranges
        assert data["home_goals_avg"].min() >= 0
        assert data["away_goals_avg"].min() >= 0
        assert 0 <= data["home_possession"].min() <= 100
        assert 0 <= data["away_possession"].min() <= 100
        assert data["home_shots_avg"].min() >= 0
        assert data["away_shots_avg"].min() >= 0
        assert data["home_form"].min() >= 0
        assert data["away_form"].min() >= 0
        assert data["weather_rainy"].isin([0, 1]).all()
        assert data["weather_windy"].isin([0, 1]).all()
        assert data["venue_capacity"].min() > 0

    def test_sample_data_consistency(self):
        """Test sample data consistency across multiple calls."""
        data1 = create_sample_data()
        data2 = create_sample_data()

        # Should have same structure
        assert list(data1.columns) == list(data2.columns)
        assert data1.shape[1] == data2.shape[1]

        # Should have different values (random data)
        # Note: This might occasionally fail due to randomness, but should generally pass
        assert not data1.equals(data2)


class TestModelEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.predictor = FootballPredictor()

    def test_predict_with_invalid_data(self):
        """Test prediction with invalid data."""
        # Test with missing keys
        invalid_data = {"home_goals_avg": 2.1}  # Missing other required keys

        with pytest.raises(KeyError):
            self.predictor.predict_match(invalid_data)

    def test_predict_with_empty_dataframe(self):
        """Test batch prediction with empty DataFrame."""
        empty_df = pd.DataFrame()

        with pytest.raises(KeyError):
            self.predictor.predict_batch(empty_df)

    def test_save_model_without_training(self):
        """Test saving model without training."""
        with pytest.raises(AttributeError):
            self.predictor.save_model("test.pkl")

    def test_load_nonexistent_model(self):
        """Test loading non-existent model file."""
        with pytest.raises(FileNotFoundError):
            self.predictor.load_model("nonexistent_model.pkl")

    def test_predict_with_extreme_values(self):
        """Test prediction with extreme values."""
        # Create model first
        self.predictor._create_simple_model()

        extreme_data = {
            "home_goals_avg": 10.0,  # Very high
            "away_goals_avg": 0.1,  # Very low
            "home_possession": 95,  # Very high
            "away_possession": 5,  # Very low
            "home_shots_avg": 50,  # Very high
            "away_shots_avg": 1,  # Very low
            "home_form": 15,  # Maximum
            "away_form": 0,  # Minimum
            "weather_rainy": 1,
            "weather_windy": 1,
            "temperature": -10,  # Very cold
            "venue_capacity": 200000,  # Very large
        }

        # Should still work
        result = self.predictor.predict_match(extreme_data)
        assert isinstance(result, dict)
        assert "outcome" in result


if __name__ == "__main__":
    pytest.main([__file__])
