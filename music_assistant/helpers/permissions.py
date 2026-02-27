"""Permission and authorization helpers for Music Assistant.

This module provides a claims-based permission system that works with both:
- Internal authentication (permissions generated from user roles)
- External OIDC providers (permissions provided in JWT claims)

Future providers can also contribute custom claims to user tokens.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from music_assistant_models.auth import UserRole

if TYPE_CHECKING:
    from music_assistant_models.auth import User


class Permission(str, Enum):
    """Permission scopes for Music Assistant operations.

    Permissions follow the format: resource:action
    Example: library:read, players:control
    """

    # Wildcard (admin)
    ALL = "*"

    # Music library permissions
    LIBRARY_READ = "library:read"
    LIBRARY_WRITE = "library:write"
    LIBRARY_DELETE = "library:delete"

    # Player permissions
    PLAYERS_READ = "players:read"
    PLAYERS_CONTROL = "players:control"
    PLAYERS_CONFIGURE = "players:configure"

    # User management permissions
    USERS_READ = "users:read"
    USERS_MANAGE = "users:manage"

    # Configuration permissions
    CONFIG_READ = "config:read"
    CONFIG_WRITE = "config:write"

    # Provider management permissions
    PROVIDERS_READ = "providers:read"
    PROVIDERS_MANAGE = "providers:manage"

    # Metadata permissions
    METADATA_READ = "metadata:read"
    METADATA_REFRESH = "metadata:refresh"

    # Stream permissions
    STREAMS_READ = "streams:read"
    STREAMS_CONTROL = "streams:control"


def get_permissions_for_role(role: UserRole) -> list[str]:
    """Get permission claims for a user role.

    :param role: User role to get permissions for.
    :return: List of permission scope strings.
    """
    if role == UserRole.ADMIN:
        return [Permission.ALL]

    return [
        Permission.LIBRARY_READ,
        Permission.PLAYERS_READ,
        Permission.PLAYERS_CONTROL,
        Permission.CONFIG_READ,
        Permission.PROVIDERS_READ,
        Permission.METADATA_READ,
        Permission.STREAMS_READ,
        Permission.STREAMS_CONTROL,
    ]


def has_permission(user: User, *required: Permission) -> bool:
    """Check if user has all required permissions.

    Checks the user's permissions list first. If empty, falls back to
    generating permissions from the user's role. This allows external
    OIDC tokens to supply their own permission claims while maintaining
    backward compatibility with role-based auth.

    :param user: User to check permissions for.
    :param required: One or more required permissions.
    :return: True if user has all required permissions.
    """
    user_permissions: list[str] = getattr(user, "permissions", None) or []

    if not user_permissions:
        user_permissions = get_permissions_for_role(user.role)

    if Permission.ALL.value in user_permissions:
        return True

    return all(perm.value in user_permissions for perm in required)


def get_user_claim(user: User, claim: str, default: Any = None) -> Any:
    """Get a custom claim from user's JWT token.

    Useful for provider-contributed claims like:
    - spotify:premium
    - sonos:features
    - tidal:subscription_tier

    :param user: User to get claim from.
    :param claim: Claim name (e.g., "spotify:premium").
    :param default: Default value if claim not found.
    :return: Claim value or default.
    """
    claims: dict[str, Any] = getattr(user, "claims", {}) or {}
    return claims.get(claim, default)
