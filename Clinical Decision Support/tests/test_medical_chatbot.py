import unittest
from unittest.mock import patch, MagicMock
from models.medical_chatbot import MedicalChatbot

class TestMedicalChatbot(unittest.TestCase):
    """Test cases for MedicalChatbot class."""
    
    def setUp(self):
        self.chatbot = MedicalChatbot()
        self.sample_patient_data = {
            "age": 45,
            "gender": "female",
            "symptoms": ["fever", "cough"],
            "lab_results": {"wbc_count": 11000}
        }
    
    @patch('openai.ChatCompletion.acreate')
    async def test_get_response(self, mock_create):
        """Test chatbot response generation."""
        # Mock OpenAI response
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Test response"))]
        )
        
        response = await self.chatbot.get_response(
            "What is the likely diagnosis?",
            "Patient context",
            "diagnosis"
        )
        
        self.assertIn("response", response)
        self.assertIn("verification_status", response)
    
    def test_prepare_context(self):
        """Test context preparation."""
        context = self.chatbot.prepare_context(
            self.sample_patient_data,
            [],
            "What is the diagnosis?"
        )
        
        self.assertIn("Patient Information", context)
        self.assertIn("45 years old", context)
        self.assertIn("fever", context)

if __name__ == "__main__":
    unittest.main()