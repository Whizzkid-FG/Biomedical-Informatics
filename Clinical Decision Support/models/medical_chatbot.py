from typing import Any, Dict, List, Optional, Tuple
import openai
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from config.config import Config
from utils.logger import setup_logger

logger = setup_logger('medical_chatbot')

class MedicalChatbot:
    """Chatbot interface for answering clinical queries."""
    
    def __init__(self):
        """Initialize the medical chatbot."""
        openai.api_key = Config.OPENAI_API_KEY
        self.conversation_history = []
        self.max_history = 5
    
    def prepare_context(
        self,
        patient_data: Dict[str, Any],
        medical_literature: List[str]
    ) -> str:
        """Prepare context from patient data and relevant medical literature."""
        context = f"""
        Patient Information:
        - Age: {patient_data.get('age')}
        - Gender: {patient_data.get('gender')}
        - Symptoms: {', '.join(patient_data.get('symptoms', []))}
        - Lab Results: {', '.join(f'{k}: {v}' for k, v in patient_data.get('lab_results', {}).items())}
        
        Relevant Medical Literature:
        {' '.join(medical_literature)}
        """
        return context
    
    async def get_response(
        self,
        query: str,
        context: str,
        response_type: str = "general"
    ) -> Dict[str, Any]:
        """Generate response to clinical query."""
        try:
            response_structure = {
                "general": "Provide a clear and concise response.",
                "diagnosis": """
                Provide response in the following structure:
                1. Differential diagnoses
                2. Recommended diagnostic tests
                3. Clinical reasoning
                """,
                "treatment": """
                Provide response in the following structure:
                1. Recommended treatments
                2. Dosing considerations
                3. Monitoring parameters
                """
            }
            
            messages = [
                {"role": "system", "content": f"""
                You are a clinical decision support system. Provide evidence-based responses
                using the provided context. {response_structure.get(response_type)}
                """},
                {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
            ]
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )
            
            generated_response = response.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.append({
                "role": "user",
                "content": query
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": generated_response
            })
            
            # Trim conversation history if too long
            if len(self.conversation_history) > self.max_history * 2:
                self.conversation_history = self.conversation_history[-self.max_history * 2:]
            
            return {
                "response": generated_response,
                "conversation_id": len(self.conversation_history) // 2
            }
            
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")
