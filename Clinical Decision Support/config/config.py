# File: config/config.py
import os
from dotenv import load_dotenv
from typing import Dict, List, Any


load_dotenv()

class Config:
    """Configuration settings for the Clinical Decision Support System."""
    
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Model parameters
    DISEASE_MODEL_PARAMS = {
        'hidden_layers': [128, 64, 32],
        'dropout_rate': 0.2,
        'learning_rate': 0.001
    }
    
    # Data processing parameters
    CATEGORICAL_FEATURES = ['gender', 'blood_type']
    NUMERICAL_FEATURES = ['age', 'temperature', 'blood_pressure', 'heart_rate']
    LAB_RESULTS = ['wbc_count', 'rbc_count', 'hemoglobin']