import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from config.config import Config


class DataProcessor:
    """Preprocessor for patient data."""

    def __init__(self):
        self.categorical_features = Config.CATEGORICAL_FEATURES
        self.numerical_features = Config.NUMERICAL_FEATURES

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), self.categorical_features)
            ]
        )

    def preprocess_data(self, data: pd.DataFrame):
        missing_cols = set(self.numerical_features + self.categorical_features) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        return self.preprocessor.fit_transform(data)
