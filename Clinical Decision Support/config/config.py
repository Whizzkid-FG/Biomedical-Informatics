import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration settings for the Clinical Decision Support System."""

    # API Keys and security
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    JWT_SECRET = os.getenv('JWT_SECRET')

    if not OPENAI_API_KEY or not JWT_SECRET:
        raise ValueError("Missing critical environment variables: OPENAI_API_KEY or JWT_SECRET")

    # Model parameters
    DISEASE_MODEL_PARAMS = {
        'hidden_layers': [128, 64, 32],
        'dropout_rate': 0.2,
        'learning_rate': 0.001
    }

    # Feature definitions
    CATEGORICAL_FEATURES = ['gender', 'blood_type']
    NUMERICAL_FEATURES = ['age', 'temperature', 'blood_pressure', 'heart_rate']
    LAB_RESULTS = ['wbc_count', 'rbc_count', 'hemoglobin']
