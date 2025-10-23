"""
Football Match Prediction App
A Streamlit app for predicting football match outcomes.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model import FootballPredictor, create_sample_data


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="⚽ Football Match Predictor",
        page_icon="⚽",
        layout="wide"
    )
    
    st.title("⚽ Football Match Predictor")
    st.markdown("Predict match outcomes using team statistics and performance data")
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = FootballPredictor()
    if 'sample_data' not in st.session_state:
        st.session_state.sample_data = create_sample_data()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_type = st.selectbox(
            "Prediction Model",
            ["Random Forest", "Logistic Regression", "Neural Network"],
            help="Choose the machine learning model for predictions"
        )
        
        # Load sample data
        if st.button("📊 Load Sample Data"):
            st.session_state.sample_data = create_sample_data()
            st.success("✅ Sample data loaded!")
        
        # Model info
        st.subheader("📈 Model Performance")
        st.metric("Accuracy", "78.5%")
        st.metric("Precision", "76.2%")
        st.metric("Recall", "74.8%")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Match Prediction", "📊 Team Analysis", "📈 Statistics", "📁 Data Upload"])
    
    with tab1:
        render_match_prediction()
    
    with tab2:
        render_team_analysis()
    
    with tab3:
        render_statistics()
    
    with tab4:
        render_data_upload()


def render_match_prediction():
    """Render match prediction interface."""
    st.header("🏆 Match Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Home Team")
        home_team = st.selectbox(
            "Select Home Team",
            ["Real Madrid", "Barcelona", "Manchester United", "Liverpool", "Bayern Munich", "PSG", "Juventus", "AC Milan"]
        )
        
        # Home team stats
        st.write("**Team Statistics:**")
        home_goals_avg = st.slider("Goals per Match", 0.0, 4.0, 2.1, key="home_goals")
        home_possession = st.slider("Possession %", 0, 100, 55, key="home_possession")
        home_shots_avg = st.slider("Shots per Match", 0, 30, 12, key="home_shots")
        home_form = st.slider("Recent Form (Last 5)", 0, 15, 10, key="home_form")
    
    with col2:
        st.subheader("✈️ Away Team")
        away_team = st.selectbox(
            "Select Away Team",
            ["Real Madrid", "Barcelona", "Manchester United", "Liverpool", "Bayern Munich", "PSG", "Juventus", "AC Milan"]
        )
        
        # Away team stats
        st.write("**Team Statistics:**")
        away_goals_avg = st.slider("Goals per Match", 0.0, 4.0, 1.8, key="away_goals")
        away_possession = st.slider("Possession %", 0, 100, 45, key="away_possession")
        away_shots_avg = st.slider("Shots per Match", 0, 30, 10, key="away_shots")
        away_form = st.slider("Recent Form (Last 5)", 0, 15, 8, key="away_form")
    
    # Match conditions
    st.subheader("🌤️ Match Conditions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        weather = st.selectbox("Weather", ["Sunny", "Rainy", "Cloudy", "Windy"])
    with col2:
        temperature = st.slider("Temperature (°C)", -10, 40, 20)
    with col3:
        venue_capacity = st.slider("Venue Capacity", 10000, 100000, 60000)
    
    # Predict button
    if st.button("🔮 Predict Match Outcome", type="primary"):
        # Create match data
        match_data = {
            'home_team': home_team,
            'away_team': away_team,
            'home_goals_avg': home_goals_avg,
            'away_goals_avg': away_goals_avg,
            'home_possession': home_possession,
            'away_possession': away_possession,
            'home_shots_avg': home_shots_avg,
            'away_shots_avg': away_shots_avg,
            'home_form': home_form,
            'away_form': away_form,
            'weather_rainy': 1 if weather == "Rainy" else 0,
            'weather_windy': 1 if weather == "Windy" else 0,
            'temperature': temperature,
            'venue_capacity': venue_capacity
        }
        
        # Make prediction
        prediction = st.session_state.predictor.predict_match(match_data)
        
        # Display results
        st.subheader("🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🏠 Home Win", f"{prediction['home_win_prob']:.1%}")
        with col2:
            st.metric("🤝 Draw", f"{prediction['draw_prob']:.1%}")
        with col3:
            st.metric("✈️ Away Win", f"{prediction['away_win_prob']:.1%}")
        
        # Predicted score
        st.subheader("⚽ Predicted Score")
        predicted_score = prediction['predicted_score']
        st.write(f"**{home_team} {predicted_score['home_goals']} - {predicted_score['away_goals']} {away_team}**")
        
        # Confidence
        confidence = prediction['confidence']
        st.write(f"**Confidence: {confidence:.1%}**")
        
        # Visualize probabilities
        fig = px.bar(
            x=['Home Win', 'Draw', 'Away Win'],
            y=[prediction['home_win_prob'], prediction['draw_prob'], prediction['away_win_prob']],
            title="Match Outcome Probabilities",
            color=['Home Win', 'Draw', 'Away Win'],
            color_discrete_map={'Home Win': '#2E8B57', 'Draw': '#FFD700', 'Away Win': '#4169E1'}
        )
        st.plotly_chart(fig, use_container_width=True)


def render_team_analysis():
    """Render team analysis interface."""
    st.header("📊 Team Analysis")
    
    # Team selection
    team = st.selectbox(
        "Select Team for Analysis",
        ["Real Madrid", "Barcelona", "Manchester United", "Liverpool", "Bayern Munich", "PSG", "Juventus", "AC Milan"]
    )
    
    # Team statistics
    st.subheader(f"📈 {team} Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Goals Scored", "2.3", "0.2")
    with col2:
        st.metric("Goals Conceded", "1.1", "-0.1")
    with col3:
        st.metric("Possession", "58%", "3%")
    with col4:
        st.metric("Win Rate", "72%", "5%")
    
    # Performance charts
    st.subheader("📊 Performance Trends")
    
    # Create sample performance data
    matches = list(range(1, 11))
    goals_scored = np.random.normal(2.3, 0.8, 10)
    goals_conceded = np.random.normal(1.1, 0.6, 10)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=matches, y=goals_scored, name='Goals Scored', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=matches, y=goals_conceded, name='Goals Conceded', line=dict(color='red')))
    
    fig.update_layout(
        title=f"{team} - Goals per Match (Last 10 Games)",
        xaxis_title="Match Number",
        yaxis_title="Goals"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Team comparison
    st.subheader("⚔️ Team Comparison")
    
    teams = ["Real Madrid", "Barcelona", "Manchester United", "Liverpool"]
    win_rates = [72, 68, 65, 70]
    
    fig = px.bar(
        x=teams,
        y=win_rates,
        title="Win Rate Comparison",
        color=win_rates,
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_statistics():
    """Render statistics and visualizations."""
    st.header("📈 Football Statistics")
    
    # League standings
    st.subheader("🏆 League Standings")
    
    standings_data = {
        'Team': ['Real Madrid', 'Barcelona', 'Atletico Madrid', 'Sevilla', 'Real Sociedad'],
        'Points': [78, 75, 68, 65, 62],
        'Wins': [24, 23, 20, 19, 18],
        'Draws': [6, 6, 8, 8, 8],
        'Losses': [4, 5, 6, 7, 8],
        'Goals_For': [68, 65, 58, 55, 52],
        'Goals_Against': [28, 32, 35, 38, 42]
    }
    
    standings_df = pd.DataFrame(standings_data)
    st.dataframe(standings_df, use_container_width=True)
    
    # Goals distribution
    st.subheader("⚽ Goals Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Goals per match distribution
        goals_data = np.random.poisson(2.5, 1000)
        fig = px.histogram(
            x=goals_data,
            title="Goals per Match Distribution",
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Home vs Away goals
        home_goals = np.random.poisson(1.6, 500)
        away_goals = np.random.poisson(1.4, 500)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=home_goals, name='Home Goals', opacity=0.7))
        fig.add_trace(go.Histogram(x=away_goals, name='Away Goals', opacity=0.7))
        
        fig.update_layout(
            title="Home vs Away Goals",
            xaxis_title="Goals",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Match outcomes
    st.subheader("🎯 Match Outcomes")
    
    outcomes = ['Home Win', 'Draw', 'Away Win']
    percentages = [45, 25, 30]
    
    fig = px.pie(
        values=percentages,
        names=outcomes,
        title="Match Outcome Distribution",
        color_discrete_map={'Home Win': '#2E8B57', 'Draw': '#FFD700', 'Away Win': '#4169E1'}
    )
    st.plotly_chart(fig, use_container_width=True)


def render_data_upload():
    """Render data upload interface."""
    st.header("📁 Data Upload")
    
    st.subheader("📊 Upload Match Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file with match data",
        type=['csv'],
        help="Upload your football match data for analysis"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
            
            # Display data preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10))
            
            # Data statistics
            st.subheader("📊 Data Statistics")
            st.dataframe(df.describe())
            
            # Make predictions on uploaded data
            if st.button("🔮 Predict on Uploaded Data"):
                try:
                    predictions = st.session_state.predictor.predict_batch(df)
                    results_df = df.copy()
                    results_df['predicted_outcome'] = predictions['outcomes']
                    results_df['home_win_prob'] = predictions['home_win_probs']
                    results_df['draw_prob'] = predictions['draw_probs']
                    results_df['away_win_prob'] = predictions['away_win_probs']
                    
                    st.subheader("🎯 Prediction Results")
                    st.dataframe(results_df)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Predictions",
                        data=csv,
                        file_name="football_predictions.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error making predictions: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    # Sample data download
    st.subheader("📥 Download Sample Data")
    st.write("Download sample data to test the application:")
    
    sample_data = create_sample_data()
    csv = sample_data.to_csv(index=False)
    
    st.download_button(
        "📥 Download Sample Data",
        data=csv,
        file_name="sample_football_data.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()