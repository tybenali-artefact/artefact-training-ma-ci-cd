"""Football prediction model and utilities."""

import pickle # nosec B403
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class FootballPredictor:
    """Football match outcome predictor."""

    def __init__(self):
        """Initialize the predictor."""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
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
        self.outcome_mapping = {0: "Home Win", 1: "Draw", 2: "Away Win"}

    def train_model(self, data: pd.DataFrame) -> None:
        """Train the prediction model."""
        # Prepare features
        x = data[self.feature_columns]
        y = data["outcome"]  # 0: Home Win, 1: Draw, 2: Away Win

        # Scale features
        x_scaled = self.scaler.fit_transform(x)

        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.model.fit(x_scaled, y)

    def predict_match(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict outcome for a single match."""
        if self.model is None:
            # Create a simple model if none exists
            self._create_simple_model()

        # Prepare features as DataFrame to maintain column names
        features_data = {col: [match_data[col]] for col in self.feature_columns}
        features_df = pd.DataFrame(features_data)
        features_scaled = self.scaler.transform(features_df)

        # Get prediction probabilities
        probabilities = self.model.predict_proba(features_scaled)[0]

        # Map to outcomes
        home_win_prob = probabilities[0]
        draw_prob = probabilities[1]
        away_win_prob = probabilities[2]

        # Predicted outcome
        predicted_outcome = np.argmax(probabilities)

        # Predicted score (simplified)
        home_goals = max(0, int(np.random.poisson(match_data["home_goals_avg"])))
        away_goals = max(0, int(np.random.poisson(match_data["away_goals_avg"])))

        # Adjust based on probabilities
        if predicted_outcome == 0:  # Home win
            home_goals = max(home_goals, away_goals + 1)
        elif predicted_outcome == 2:  # Away win
            away_goals = max(away_goals, home_goals + 1)

        return {
            "outcome": self.outcome_mapping[predicted_outcome],
            "home_win_prob": home_win_prob,
            "draw_prob": draw_prob,
            "away_win_prob": away_win_prob,
            "predicted_score": {"home_goals": home_goals, "away_goals": away_goals},
            "confidence": max(probabilities),
        }

    def predict_batch(self, data: pd.DataFrame) -> Dict[str, List]:
        """Predict outcomes for multiple matches."""
        if self.model is None:
            self._create_simple_model()

        # Prepare features
        x = data[self.feature_columns]
        x_scaled = self.scaler.transform(x)

        # Get predictions and probabilities
        predictions = self.model.predict(x_scaled)
        probabilities = self.model.predict_proba(x_scaled)

        return {
            "outcomes": [self.outcome_mapping[pred] for pred in predictions],
            "home_win_probs": probabilities[:, 0].tolist(),
            "draw_probs": probabilities[:, 1].tolist(),
            "away_win_probs": probabilities[:, 2].tolist(),
        }

    def _create_simple_model(self) -> None:
        """Create a simple model for demonstration."""
        # Generate sample training data
        np.random.seed(42)
        n_samples = 1000

        data = {
            "home_goals_avg": np.random.normal(1.8, 0.5, n_samples),
            "away_goals_avg": np.random.normal(1.6, 0.5, n_samples),
            "home_possession": np.random.normal(55, 10, n_samples),
            "away_possession": np.random.normal(45, 10, n_samples),
            "home_shots_avg": np.random.normal(12, 3, n_samples),
            "away_shots_avg": np.random.normal(10, 3, n_samples),
            "home_form": np.random.randint(0, 15, n_samples),
            "away_form": np.random.randint(0, 15, n_samples),
            "weather_rainy": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            "weather_windy": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            "temperature": np.random.normal(20, 10, n_samples),
            "venue_capacity": np.random.randint(20000, 100000, n_samples),
        }

        # Create outcomes based on simple rules
        outcomes = []
        for i in range(n_samples):
            home_advantage = data["home_goals_avg"][i] - data["away_goals_avg"][i]
            form_diff = data["home_form"][i] - data["away_form"][i]

            if home_advantage + form_diff * 0.1 > 0.3:
                outcomes.append(0)  # Home win
            elif home_advantage + form_diff * 0.1 < -0.3:
                outcomes.append(2)  # Away win
            else:
                outcomes.append(1)  # Draw

        data["outcome"] = outcomes

        # Train model
        df = pd.DataFrame(data)
        self.train_model(df)

    def save_model(self, path: str) -> None:
        """Save the trained model."""
        if self.model is None:
            raise AttributeError(
                "No model has been trained yet. Please train a model before saving."
            )

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "outcome_mapping": self.outcome_mapping,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    def load_model(self, path: str) -> None:
        """Load a trained model."""
        with open(path, "rb") as f:
            model_data = pickle.load(f) # nosec B301

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_columns = model_data["feature_columns"]
        self.outcome_mapping = model_data["outcome_mapping"]


def create_sample_data() -> pd.DataFrame:
    """Create sample football match data."""
    # Don't set seed here to allow for different data each time
    n_matches = 50

    # Generate sample data with different teams for each match
    teams = [
        "Real Madrid",
        "Barcelona",
        "Manchester United",
        "Liverpool",
        "Bayern Munich",
        "PSG",
        "Juventus",
        "AC Milan",
        "Chelsea",
        "Arsenal",
    ]

    # Ensure home and away teams are different
    home_teams = []
    away_teams = []

    for _ in range(n_matches):
        home_team = np.random.choice(teams)
        # Select away team that's different from home team
        available_teams = [team for team in teams if team != home_team]
        away_team = np.random.choice(available_teams)

        home_teams.append(home_team)
        away_teams.append(away_team)

    # Generate correlated data for more realistic relationships
    # Base values for correlation
    base_goals = np.random.normal(1.7, 0.3, n_matches)
    base_possession = np.random.normal(50, 8, n_matches)

    # Create correlated data
    home_goals_avg = np.clip(base_goals + np.random.normal(0, 0.2, n_matches), 0.5, 3.5)
    away_goals_avg = np.clip(base_goals + np.random.normal(0, 0.2, n_matches), 0.5, 3.5)

    # Possession should be complementary (home + away ≈ 100)
    home_possession = np.clip(base_possession + np.random.normal(0, 5, n_matches), 20, 80)
    away_possession = 100 - home_possession

    # Shots should correlate with both goals and possession
    # Use possession as a base for shots to ensure positive correlation
    home_shots_base = home_possession * 0.2 + home_goals_avg * 4
    away_shots_base = away_possession * 0.2 + away_goals_avg * 4

    home_shots_avg = np.clip(home_shots_base + np.random.normal(0, 1, n_matches), 5, 25)
    away_shots_avg = np.clip(away_shots_base + np.random.normal(0, 1, n_matches), 5, 25)

    # Temperature with reasonable range
    temperature = np.clip(np.random.normal(20, 8, n_matches), -10, 45)

    data = {
        "home_team": home_teams,
        "away_team": away_teams,
        "home_goals_avg": home_goals_avg,
        "away_goals_avg": away_goals_avg,
        "home_possession": home_possession,
        "away_possession": away_possession,
        "home_shots_avg": home_shots_avg,
        "away_shots_avg": away_shots_avg,
        "home_form": np.random.randint(0, 15, n_matches),
        "away_form": np.random.randint(0, 15, n_matches),
        "weather_rainy": np.random.choice([0, 1], n_matches, p=[0.8, 0.2]),
        "weather_windy": np.random.choice([0, 1], n_matches, p=[0.9, 0.1]),
        "temperature": temperature,
        "venue_capacity": np.random.randint(20000, 100000, n_matches),
    }

    # Create outcomes
    outcomes = []
    for i in range(n_matches):
        home_advantage = data["home_goals_avg"][i] - data["away_goals_avg"][i]
        form_diff = data["home_form"][i] - data["away_form"][i]

        if home_advantage + form_diff * 0.1 > 0.3:
            outcomes.append("Home Win")
        elif home_advantage + form_diff * 0.1 < -0.3:
            outcomes.append("Away Win")
        else:
            outcomes.append("Draw")

    data["outcome"] = outcomes

    return pd.DataFrame(data)


def create_and_save_model():
    """Create and save a trained model."""
    predictor = FootballPredictor()

    # Create training data
    training_data = create_sample_data()
    training_data["outcome_numeric"] = training_data["outcome"].map(
        {"Home Win": 0, "Draw": 1, "Away Win": 2}
    )

    # Train model
    predictor.train_model(training_data)

    # Save model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    predictor.save_model("models/football_model.pkl")

    print("Model created and saved to models/football_model.pkl")


if __name__ == "__main__":
    create_and_save_model()
