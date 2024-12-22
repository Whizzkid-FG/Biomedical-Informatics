from typing import Dict, Any
from jose import jwt
from datetime import datetime, timedelta
from config.config import Config

class SecurityManager:
    """Handles security operations including authentication and authorization."""
    
    @staticmethod
    def create_access_token(
        user_data: Dict[str, Any],
        expires_delta: timedelta = timedelta(hours=24)
    ) -> str:
        """Create JWT access token."""
        to_encode = user_data.copy()
        expire = datetime.utcnow() + expires_delta
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, Config.JWT_SECRET, algorithm="HS256")
    
    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, Config.JWT_SECRET, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.JWTError:
            raise ValueError("Invalid token")
    
    @staticmethod
    def has_access_rights(user_role: str, required_role: str) -> bool:
        """Check if user has required access rights."""
        role_hierarchy = {
            'admin': 3,
            'doctor': 2,
            'nurse': 1
        }
        return role_hierarchy.get(user_role, 0) >= role_hierarchy.get(required_role, 0)