"""
JWT authentication middleware for Python backend.
Validates JWT tokens issued by Java backend.
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from app.config import settings
from typing import Optional
from uuid import UUID


security = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Verify JWT token and extract payload.

    Args:
        credentials: HTTP Bearer token from Authorization header

    Returns:
        dict: Token payload containing userId and username

    Raises:
        HTTPException: If token is invalid or expired
    """
    token = credentials.credentials

    try:
        # Decode JWT token using same secret as Java backend
        # Support multiple algorithms (HS256, HS384, HS512) as Java auto-selects based on key length
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=settings.JWT_ALGORITHMS
        )

        # Extract user information from payload
        user_id_str: Optional[str] = payload.get("userId")  # Match Java's claim name
        username: Optional[str] = payload.get("username")

        if user_id_str is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user ID"
            )

        # Convert userId string to UUID
        try:
            user_id = UUID(user_id_str)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: invalid user ID format"
            )

        return {
            "user_id": user_id,
            "username": username
        }

    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}"
        )


async def get_current_user(token_data: dict = Depends(verify_token)) -> dict:
    """
    FastAPI dependency to get current authenticated user.

    Usage:
        @app.get("/protected")
        async def protected_route(user: dict = Depends(get_current_user)):
            user_id = user["user_id"]
            username = user["username"]
    """
    return token_data
