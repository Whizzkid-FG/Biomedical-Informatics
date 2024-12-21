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
    """
    Enhanced chatbot interface for medical queries with improved context management
    and response generation.
    
    Features:
    - Semantic search for relevant medical literature
    - Context management with memory
    - Response verification using medical knowledge base
    - Structured output formatting
    """
    
    def __init__(self):
        """Initialize the medical chatbot with necessary components."""
        openai.api_key = Config.OPENAI_API_KEY
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.encoder = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.conversation_history = []
        self.max_history = 5
        
    def _encode_text(self, text: str) -> np.ndarray:
        """
        Encode text using sentence transformer for semantic search.
        
        Args:
            text: Input text to encode
            
        Returns:
            numpy.ndarray: Encoded text embedding
        """
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
            
        return normalized_embeddings.numpy()
    
    def find_relevant_literature(
        self,
        query: str,
        medical_literature: List[Dict[str, str]],
        top_k: int = 3
    ) -> List[Dict[str, str]]:
        """
        Find most relevant medical literature using semantic search.
        
        Args:
            query: Clinical query
            medical_literature: List of medical literature documents
            top_k: Number of relevant documents to return
            
        Returns:
            List of most relevant literature documents
        """
        query_embedding = self._encode_text(query)
        
        # Encode all documents
        doc_embeddings = np.vstack([
            self._encode_text(doc['content'])
            for doc in medical_literature
        ])
        
        # Calculate similarities
        similarities = np.dot(query_embedding, doc_embeddings.T)[0]
        
        # Get top-k documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [medical_literature[i] for i in top_indices]
    
    def prepare_context(
        self,
        patient_data: Dict,
        medical_literature: List[Dict[str, str]],
        query: str
    ) -> str:
        """
        Prepare context by combining patient data, relevant literature,
        and conversation history.
        
        Args:
            patient_data: Patient information
            medical_literature: Available medical literature
            query: Current clinical query
            
        Returns:
            str: Prepared context for the model
        """
        # Find relevant literature
        relevant_docs = self.find_relevant_literature(query, medical_literature)
        
        # Format patient information
        patient_context = f"""
        Patient Information:
        - Demographics: {patient_data.get('age')} years old, {patient_data.get('gender')}
        - Vital Signs: BP {patient_data.get('systolic_bp')}/{patient_data.get('diastolic_bp')}, 
          HR {patient_data.get('heart_rate')}, Temp {patient_data.get('temperature')}
        - Symptoms: {', '.join(patient_data.get('symptoms', []))}
        - Lab Results: {', '.join(f'{k}: {v}' for k, v in patient_data.get('lab_results', {}).items())}
        - Medical History: {patient_data.get('medical_history', 'None reported')}
        """
        
        # Format relevant literature
        literature_context = "\n".join([
            f"Reference {i+1}: {doc.get('title')}\n{doc.get('content')}\n"
            for i, doc in enumerate(relevant_docs)
        ])
        
        # Add recent conversation history
        conversation_context = "\n".join([
            f"Previous {exchange['role']}: {exchange['content']}"
            for exchange in self.conversation_history[-self.max_history:]
        ])
        
        return f"""
        {patient_context}
        
        Relevant Medical Literature:
        {literature_context}
        
        Recent Conversation History:
        {conversation_context}
        """
    
    async def verify_response(self, response: str, context: str) -> Tuple[bool, str]:
        """
        Verify generated response against medical knowledge base.
        
        Args:
            response: Generated response to verify
            context: Context used to generate the response
            
        Returns:
            Tuple containing verification result and explanation
        """
        try:
            verify_messages = [
                {"role": "system", "content": """
                You are a medical response verification system. Verify the following response
                against the provided context and medical knowledge. Check for:
                1. Clinical accuracy
                2. Consistency with provided patient data
                3. Evidence-based recommendations
                4. Appropriate medical terminology
                """},
                {"role": "user", "content": f"""
                Context: {context}
                
                Response to verify: {response}
                
                Verify this response and explain your reasoning.
                """}
            ]
            
            verification = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=verify_messages,
                temperature=0.3,
                max_tokens=300
            )
            
            # Parse verification result
            verification_text = verification.choices[0].message.content
            is_valid = "accurate" in verification_text.lower() and "consistent" in verification_text.lower()
            
            return is_valid, verification_text
            
        except Exception as e:
            logger.error(f"Response verification failed: {str(e)}")
            return True, "Verification service unavailable, proceeding with generated response"
    
    async def get_response(
        self,
        query: str,
        context: str,
        response_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Generate structured response to clinical query using GPT model.
        
        Args:
            query: Clinical question from healthcare provider
            context: Relevant patient data and medical literature
            response_type: Type of response required (general, diagnosis, treatment, etc.)
            
        Returns:
            Dict containing structured response
        """
        try:
            # Define response structure based on type
            response_structure = {
                "general": "Provide a clear and concise response.",
                "diagnosis": """
                Provide response in the following structure:
                1. Differential diagnoses (ranked by likelihood)
                2. Recommended diagnostic tests
                3. Clinical reasoning
                4. Additional information needed
                """,
                "treatment": """
                Provide response in the following structure:
                1. Recommended treatments (first-line and alternatives)
                2. Dosing considerations
                3. Monitoring parameters
                4. Potential complications
                5. Follow-up recommendations
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
                max_tokens=800
            )
            
            generated_response = response.choices[0].message.content
            
            # Verify response
            is_valid, verification_notes = await self.verify_response(
                generated_response,
                context
            )
            
            if not is_valid:
                logger.warning(f"Response verification failed: {verification_notes}")
                # Generate alternative response or add warning
                generated_response += "\n\nNote: This response may require additional verification."
            
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
                "verification_status": is_valid,
                "verification_notes": verification_notes,
                "references": [doc.get('title') for doc in self.find_relevant_literature(query, [], top_k=3)]
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
