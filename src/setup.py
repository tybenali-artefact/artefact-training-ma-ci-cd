"""Setup script for the Football Prediction App.

Run this script to initialize the model and create sample data.
"""

import subprocess
import sys


def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")


def create_model():
    """Create and save the football prediction model."""
    print("Creating football prediction model...")
    try:
        from model import create_and_save_model

        create_and_save_model()
        print("✅ Model created successfully!")
    except Exception as e:
        print(f"❌ Error creating model: {e}")


def main():
    """Main setup function."""
    print("🏈 Setting up Football Prediction App...")
    print("=" * 50)

    # Install requirements
    install_requirements()

    # Create model
    create_model()

    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("=" * 50)
    print("To run the app:")
    print("1. cd src")
    print("2. streamlit run app.py")
    print("3. Open http://localhost:8501 in your browser")
    print("\nFeatures available:")
    print("• 🏆 Match Prediction - Predict individual match outcomes")
    print("• 📊 Team Analysis - Analyze team performance")
    print("• 📈 Statistics - View league statistics and trends")
    print("• 📁 Data Upload - Upload your own match data")


if __name__ == "__main__":
    main()
