import pandas as pd
import numpy as np
from typing import Dict, Any
from config.config import Config

class DataProcessor:
    """Utility class for processing and validating medical data."""
    
    @staticmethod
    def validate_patient_data(data: Dict[str, Any]) -> bool:
        """Validate required patient data fields."""
        required_fields = {
            'age': (int, lambda x: 0 <= x <= 120),
            'gender': (str, lambda x: x.lower() in ['male', 'female', 'other']),
            'symptoms': (list, lambda x: len(x) > 0)
        }
        
        for field, (expected_type, validator) in required_fields.items():
            if field not in data:
                return False
            if not isinstance(data[field], expected_type):
                return False
            if not validator(data[field]):
                return False
        
        return True
    
    @staticmethod
    def process_lab_results(lab_results: Dict[str, float]) -> Dict[str, float]:
        """Process and normalize lab results."""
        processed_results = {}
        
        for test, value in lab_results.items():
            if test in Config.LAB_RESULTS:
                # Add your normalization logic here
                processed_results[test] = float(value)
        
        return processed_results