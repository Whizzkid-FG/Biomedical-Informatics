from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
import pandas as pd
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
import uvicorn

from models.disease_predictor import DiseasePredictor
from models.medical_chatbot import MedicalChatbot
from utils.data_processor import DataProcessor
from utils.security import SecurityManager


# Load the dataset
data = pd.read_csv("dataset.csv")

# Preprocess the dataset
processor = DataProcessor()
processed_data = processor.preprocess_data(data)

# Split data into features and target
X = processed_data
y = data['disease_present']  # Target column

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize components
data_processor = DataProcessor()
medical_chatbot = MedicalChatbot()

# Initialize disease predictor with saved model
disease_predictor = DiseasePredictor(
    input_size=64,  # Set based on preprocessed feature size
    hidden_layers=[128, 64, 32],
    num_classes=1
)
disease_predictor.load_state_dict(torch.load('best_model.pt'))

class PatientData(BaseModel):
    """Pydantic model for patient data validation."""
    age: int
    gender: str
    symptoms: List[str]
    lab_results: Dict[str, float]

async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """Get current user from token."""
    try:
        user_data = SecurityManager.verify_token(token)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return user_data
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

@app.post("/predict_disease")
async def predict_disease(
    patient_data: PatientData,
    current_user: Dict = Depends(get_current_user)
):
    """Endpoint for disease prediction."""
    try:
        # Check access rights
        if not SecurityManager.has_access_rights(current_user['role'], 'doctor'):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        # Preprocess patient data
        processed_data = data_processor.transform_single_patient(patient_data.dict())
        
        # Get prediction
        with torch.no_grad():
            predictions, probabilities = disease_predictor.predict(
                torch.FloatTensor(processed_data)
            )
        
        return {
            "prediction": bool(predictions[0][0]),
            "probability": float(probabilities[0][0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/ask_clinician")
async def ask_clinician(
    query: str,
    patient_data: PatientData,
    current_user: Dict = Depends(get_current_user)
):
    """Endpoint for clinician queries about patient cases."""
    try:
        # Check access rights
        if not SecurityManager.has_acess_rights(current_user['role'], 'nurse'):
            raise HTTPException(status_code=403, detail="Insufficient permission")
        
        # Prepare context from patient data
        context = medical_chatbot.prepare_context(
            patient_data.dict(),
            medical_literature=[]  # Add relevant literature retrieval logic
        )
        
        # Get response from chatbot
        respomse = await medical_chatbot.get_response(query, context)

        return respomse
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/token")
async def login(username: str, password: str):
    """Login endpoint to get access taken."""
    # In a real application, verify credentials against a database
    # This is a sinplified example
    if username == "doctor" and password == "password":
        access_token = SecurityManager.create_access_token(
            {"sub": username, "role": "doctor"}
        )
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0", port=8000)