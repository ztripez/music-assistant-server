"""Tests for the claims-based permission system."""

from __future__ import annotations

from typing import Any

from music_assistant_models.auth import User, UserRole

from music_assistant.helpers.permissions import (
    Permission,
    get_permissions_for_role,
    get_user_claim,
    has_permission,
)


def _make_user(
    role: UserRole = UserRole.USER,
    permissions: list[str] | None = None,
    claims: dict[str, Any] | None = None,
) -> User:
    """Create a mock User for testing.

    :param role: User role.
    :param permissions: Optional explicit permissions list.
    :param claims: Optional JWT claims dict.
    """
    user = User(user_id="test-user", username="testuser", role=role)
    if permissions is not None:
        user.permissions = permissions  # type: ignore[attr-defined]
    if claims is not None:
        user.claims = claims  # type: ignore[attr-defined]
    return user


class TestPermissionEnum:
    """Tests for Permission enum values."""

    def test_wildcard_value(self) -> None:
        """Wildcard permission should be '*'."""
        assert Permission.ALL.value == "*"

    def test_resource_action_format(self) -> None:
        """All non-wildcard permissions should follow resource:action format."""
        for perm in Permission:
            if perm == Permission.ALL:
                continue
            assert ":" in perm.value, f"{perm.name} does not follow resource:action format"

    def test_string_enum(self) -> None:
        """Permission should be a string enum for JSON serialization."""
        assert isinstance(Permission.LIBRARY_READ, str)
        assert Permission.LIBRARY_READ.value == "library:read"


class TestGetPermissionsForRole:
    """Tests for role-to-permissions mapping."""

    def test_admin_gets_wildcard(self) -> None:
        """Admin role should receive wildcard permission only."""
        perms = get_permissions_for_role(UserRole.ADMIN)
        assert perms == [Permission.ALL]

    def test_user_gets_read_and_control(self) -> None:
        """Regular user should get read and control permissions."""
        perms = get_permissions_for_role(UserRole.USER)
        assert Permission.LIBRARY_READ in perms
        assert Permission.PLAYERS_READ in perms
        assert Permission.PLAYERS_CONTROL in perms
        assert Permission.STREAMS_CONTROL in perms

    def test_user_lacks_admin_permissions(self) -> None:
        """Regular user should not have configuration or management permissions."""
        perms = get_permissions_for_role(UserRole.USER)
        assert Permission.LIBRARY_WRITE not in perms
        assert Permission.LIBRARY_DELETE not in perms
        assert Permission.PLAYERS_CONFIGURE not in perms
        assert Permission.USERS_MANAGE not in perms
        assert Permission.CONFIG_WRITE not in perms
        assert Permission.PROVIDERS_MANAGE not in perms

    def test_guest_same_as_user(self) -> None:
        """Guest role should fall through to the regular user permissions."""
        perms = get_permissions_for_role(UserRole.GUEST)
        user_perms = get_permissions_for_role(UserRole.USER)
        assert perms == user_perms


class TestHasPermission:
    """Tests for permission checking."""

    def test_admin_has_all_permissions(self) -> None:
        """Admin wildcard should grant any permission."""
        user = _make_user(UserRole.ADMIN, permissions=["*"])
        assert has_permission(user, Permission.LIBRARY_DELETE)
        assert has_permission(user, Permission.USERS_MANAGE)
        assert has_permission(user, Permission.CONFIG_WRITE)

    def test_user_has_granted_permission(self) -> None:
        """User should pass check for explicitly granted permissions."""
        user = _make_user(permissions=["library:read", "players:control"])
        assert has_permission(user, Permission.LIBRARY_READ)
        assert has_permission(user, Permission.PLAYERS_CONTROL)

    def test_user_lacks_missing_permission(self) -> None:
        """User should fail check for permissions not in their list."""
        user = _make_user(permissions=["library:read"])
        assert not has_permission(user, Permission.LIBRARY_WRITE)
        assert not has_permission(user, Permission.CONFIG_WRITE)

    def test_multiple_required_all_must_match(self) -> None:
        """All required permissions must be present for the check to pass."""
        user = _make_user(permissions=["library:read", "library:write"])
        assert has_permission(user, Permission.LIBRARY_READ, Permission.LIBRARY_WRITE)
        assert not has_permission(user, Permission.LIBRARY_READ, Permission.LIBRARY_DELETE)

    def test_fallback_to_role_when_no_permissions(self) -> None:
        """When no permissions are set, should fall back to role-based generation."""
        user = _make_user(UserRole.ADMIN)
        # No explicit permissions set — should use role to generate (admin = wildcard)
        assert has_permission(user, Permission.CONFIG_WRITE)

    def test_fallback_for_regular_user(self) -> None:
        """Regular user with no explicit permissions should get role defaults."""
        user = _make_user(UserRole.USER)
        assert has_permission(user, Permission.LIBRARY_READ)
        assert not has_permission(user, Permission.LIBRARY_WRITE)

    def test_empty_permissions_list_triggers_fallback(self) -> None:
        """An empty permissions list should trigger role-based fallback."""
        user = _make_user(UserRole.ADMIN, permissions=[])
        assert has_permission(user, Permission.CONFIG_WRITE)


class TestGetUserClaim:
    """Tests for JWT claim extraction."""

    def test_existing_claim(self) -> None:
        """Should return claim value when present."""
        user = _make_user(claims={"spotify:premium": True, "tidal:tier": "hifi"})
        assert get_user_claim(user, "spotify:premium") is True
        assert get_user_claim(user, "tidal:tier") == "hifi"

    def test_missing_claim_returns_default(self) -> None:
        """Should return default when claim is not present."""
        user = _make_user(claims={})
        assert get_user_claim(user, "nonexistent") is None
        assert get_user_claim(user, "nonexistent", "fallback") == "fallback"

    def test_no_claims_attribute(self) -> None:
        """Should handle user with no claims attribute gracefully."""
        user = _make_user()
        assert get_user_claim(user, "anything") is None
