"""JWT token helper for Music Assistant authentication."""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta
from typing import Any

import jwt
from music_assistant_models.auth import User, UserRole

from music_assistant.helpers.datetime import utc


class JWTHelper:
    """Helper class for JWT token operations."""

    def __init__(self, secret_key: str) -> None:
        """Initialize JWT helper.

        :param secret_key: Secret key for signing JWTs.
        """
        self.secret_key = secret_key
        self.algorithm = "HS256"

    def encode_token(
        self,
        user: User,
        token_id: str,
        token_name: str,
        expires_at: datetime,
        is_long_lived: bool = False,
    ) -> str:
        """Encode a JWT token for a user.

        :param user: User object to create token for.
        :param token_id: Unique token identifier.
        :param token_name: Human-readable token name.
        :param expires_at: Token expiration datetime.
        :param is_long_lived: Whether this is a long-lived token.
        :return: Encoded JWT token string.
        """
        now = utc()
        payload = {
            # Standard JWT claims
            "sub": user.user_id,  # Subject (user ID)
            "jti": token_id,  # JWT ID (token ID)
            "iat": int(now.timestamp()),  # Issued at
            "exp": int(expires_at.timestamp()),  # Expiration
            # Custom claims
            "username": user.username,
            "role": user.role.value,
            "player_filter": user.player_filter,
            "provider_filter": user.provider_filter,
            "token_name": token_name,
            "is_long_lived": is_long_lived,
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def decode_token(self, token: str, verify_exp: bool = True) -> dict[str, Any]:
        """Decode and verify a JWT token.

        :param token: JWT token string to decode.
        :param verify_exp: Whether to verify token expiration.
        :return: Decoded token payload.
        :raises jwt.InvalidTokenError: If token is invalid or expired.
        """
        options = {"verify_exp": verify_exp}
        payload: dict[str, Any] = jwt.decode(
            token,
            self.secret_key,
            algorithms=[self.algorithm],
            options=options,
        )
        return payload

    def refresh_short_lived_token(
        self,
        token: str,
        token_name: str,
        expiration_days: int = 30,
    ) -> str:
        """Refresh a short-lived token with new expiration.

        :param token: Original JWT token.
        :param token_name: Token name.
        :param expiration_days: Days until new expiration.
        :return: New JWT token with updated expiration.
        :raises jwt.InvalidTokenError: If token is invalid.
        """
        # Decode without verifying expiration (allow expired tokens to be refreshed)
        payload = self.decode_token(token, verify_exp=False)

        # Update timestamps
        now = utc()
        new_expires_at = now + timedelta(days=expiration_days)

        payload["iat"] = int(now.timestamp())
        payload["exp"] = int(new_expires_at.timestamp())
        payload["token_name"] = token_name

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    @staticmethod
    def generate_secret_key() -> str:
        """Generate a secure random secret key for JWT signing.

        :return: Base64-encoded 256-bit random key.
        """
        return secrets.token_urlsafe(32)  # 32 bytes = 256 bits

    def get_token_id(self, token: str) -> str | None:
        """Extract token ID (jti) from JWT without full validation.

        :param token: JWT token string.
        :return: Token ID or None if invalid.
        """
        try:
            # Decode without verification to get jti for lookup
            payload: dict[str, Any] = jwt.decode(
                token,
                options={"verify_signature": False, "verify_exp": False},
            )
            jti = payload.get("jti")
            return str(jti) if jti else None
        except Exception:
            return None

    def get_user_from_token(self, token: str) -> tuple[str, UserRole] | None:
        """Extract user ID and role from token without full validation.

        Useful for quick user lookup without database access.

        :param token: JWT token string.
        :return: Tuple of (user_id, role) or None if invalid.
        """
        try:
            payload = jwt.decode(
                token,
                options={"verify_signature": False, "verify_exp": False},
            )
            user_id = payload.get("sub")
            role_str = payload.get("role")
            if user_id and role_str:
                return user_id, UserRole(role_str)
            return None
        except Exception:
            return None
