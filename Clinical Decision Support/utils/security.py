# File: utils/security.py
from typing import Dict, Any
import jwt
from cryptography.fernet import Fernet
from datetime import datetime, timedelta
from config.config import Config

class SecurityManager:
    """Handles security operations including encryption, authentication, and authorization."""
    
    def __init__(self):
        self.fernet = Fernet(Config.ENCRYPTION_KEY.encode())
    
    def encrypt_patient_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive patient data fields."""
        sensitive_fields = ['ssn', 'patient_id', 'contact_info']
        encrypted_data = data.copy()
        
        for field in sensitive_fields:
            if field in encrypted_data:
                encrypted_data[field] = self.fernet.encrypt(
                    str(encrypted_data[field]).encode()
                ).decode()
        
        return encrypted_data
    
    def decrypt_patient_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive patient data fields."""
        encrypted_fields = ['ssn', 'patient_id', 'contact_info']
        decrypted_data = data.copy()
        
        for field in encrypted_fields:
            if field in decrypted_data:
                decrypted_data[field] = self.fernet.decrypt(
                    decrypted_data[field].encode()
                ).decode()
        
        return decrypted_data
    
    def generate_access_token(self, user_id: str, role: str) -> str:
        """Generate JWT access token for API authentication."""
        payload = {
            'user_id': user_id,
            'role': role,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }
        return jwt.encode(payload, Config.JWT_SECRET, algorithm='HS256')
    
    def verify_access_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT access token."""
        try:
            payload = jwt.decode(token, Config.JWT_SECRET, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")