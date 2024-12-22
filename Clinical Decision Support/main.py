import torch
import uvicorn
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from typing import List, Dict, Any
from utils.data_processor import DataProcessor
from models.disease_predictor import DiseasePredictor
from models.medical_chatbot import MedicalChatbot
from utils.security import SecurityManager

# Load and preprocess the dataset
try:
    data = pd.read_csv("C:\My Work Station\Biomedical Informatics\Clinical Decision Support\diabetes dataset\diabetes.csv")
    processor = DataProcessor()
    processed_data = processor.preprocess_data(data)
    X = processed_data
    y = data['disease_present']  # Target column
except Exception as e:
    raise RuntimeError(f"Error loading or processing dataset: {str(e)}")

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize components
data_processor = DataProcessor()
medical_chatbot = MedicalChatbot()

# Initialize disease predictor with saved model
disease_predictor = DiseasePredictor(
    input_size=processed_data.shape[1],  # Automatically set input size
    hidden_layers=[128, 64, 32],
    num_classes=1
)
try:
    disease_predictor.load_state_dict(torch.load('best_model.pt'))
    disease_predictor.eval()  # Ensure the model is in evaluation mode
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")


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
    except Exception as e:
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
        if not SecurityManager.has_access_rights(current_user['role'], 'nurse'):
            raise HTTPException(status_code=403, detail="Insufficient permissions")

        # Prepare context from patient data
        context = medical_chatbot.prepare_context(
            patient_data.dict(),
            medical_literature=[]  # Add relevant literature retrieval logic
        )

        # Get response from chatbot
        response = await medical_chatbot.get_response(query, context)

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/token")
async def login(username: str, password: str):
    """Login endpoint to get access token."""
    # In a real application, verify credentials against a database
    if username == "doctor" and password == "password":
        access_token = SecurityManager.create_access_token(
            {"sub": username, "role": "doctor"}
        )
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
