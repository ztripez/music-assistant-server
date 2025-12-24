"""Authentication for MCP Server using Music Assistant tokens."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.server.auth.provider import AccessToken, TokenVerifier

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant


class MusicAssistantTokenVerifier(TokenVerifier):
    """Verify Music Assistant access tokens for MCP requests."""

    def __init__(self, mass: MusicAssistant) -> None:
        """Initialize the token verifier.

        :param mass: MusicAssistant instance for token validation.
        """
        self._mass = mass

    async def verify_token(self, token: str) -> AccessToken | None:
        """Verify a Music Assistant access token.

        :param token: The bearer token from the Authorization header.
        :return: AccessToken if valid, None if invalid.
        """
        try:
            # Use MA's built-in token authentication
            user = await self._mass.webserver.auth.authenticate_with_token(token)
            if user is None:
                return None

            # Map MA user role to scopes
            scopes = ["user"]
            if user.role and user.role.value == "admin":
                scopes.append("admin")

            return AccessToken(
                token=token,
                client_id="music-assistant",
                scopes=scopes,
                expires_at=None,  # MA handles expiration internally
            )
        except Exception:
            return None
