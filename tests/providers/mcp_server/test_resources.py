"""Tests for MCP Server resources."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import Mock

from mcp.server.fastmcp import FastMCP

from music_assistant.providers.mcp_server.resources import (
    register_library_resources,
    register_player_resources,
)


def _get_resource(mcp: FastMCP, uri_contains: str) -> Any:
    """Get a static resource by URI pattern."""
    for resource in mcp._resource_manager._resources.values():
        if uri_contains in str(resource.uri):
            return resource
    return None


def _get_template(mcp: FastMCP, uri_contains: str) -> Any:
    """Get a template resource by URI pattern."""
    for template in mcp._resource_manager._templates.values():
        if uri_contains in template.uri_template:
            return template
    return None


# =============================================================================
# PLAYER RESOURCES
# =============================================================================


async def test_list_players_resource(mock_mass: Mock) -> None:
    """Test players:// resource."""
    mcp = FastMCP("test")
    register_player_resources(mcp, mock_mass)

    list_players = _get_resource(mcp, "players://")
    assert list_players is not None
    result = await list_players.fn()
    data = json.loads(result)
    assert "players" in data
    assert len(data["players"]) == 2
    assert data["players"][0]["name"] == "Living Room"
    assert data["players"][1]["name"] == "Kitchen"


async def test_get_player_resource(mock_mass: Mock) -> None:
    """Test player://{player_id} resource."""
    mcp = FastMCP("test")
    register_player_resources(mcp, mock_mass)

    get_player = _get_template(mcp, "player://")
    assert get_player is not None
    result = await get_player.fn(player_id="player_1")
    data = json.loads(result)
    assert "player" in data
    assert data["player"]["name"] == "Living Room"
    assert data["player"]["volume"] == 50


async def test_now_playing_resource(mock_mass: Mock) -> None:
    """Test nowplaying://{player_id} resource."""
    mcp = FastMCP("test")
    register_player_resources(mcp, mock_mass)

    now_playing = _get_template(mcp, "nowplaying://")
    assert now_playing is not None
    result = await now_playing.fn(player_id="player_1")
    data = json.loads(result)
    assert "now_playing" in data
    assert data["now_playing"]["name"] == "Test Track"


async def test_queue_resource(mock_mass: Mock) -> None:
    """Test queue://{player_id} resource."""
    mcp = FastMCP("test")
    register_player_resources(mcp, mock_mass)

    queue_resource = _get_template(mcp, "queue://")
    assert queue_resource is not None
    result = await queue_resource.fn(player_id="player_1")
    data = json.loads(result)
    assert "queue" in data
    assert "items" in data["queue"]


# =============================================================================
# LIBRARY RESOURCES
# =============================================================================


async def test_library_stats_resource(mock_mass: Mock) -> None:
    """Test library://stats resource."""
    mcp = FastMCP("test")
    register_library_resources(mcp, mock_mass)

    stats_resource = _get_resource(mcp, "library://stats")
    assert stats_resource is not None
    result = await stats_resource.fn()
    data = json.loads(result)
    assert "library_stats" in data
    assert data["library_stats"]["artists"] == 100
    assert data["library_stats"]["albums"] == 50
    assert data["library_stats"]["tracks"] == 500


async def test_favorites_resource(mock_mass: Mock) -> None:
    """Test library://favorites resource."""
    mcp = FastMCP("test")
    register_library_resources(mcp, mock_mass)

    fav_resource = _get_resource(mcp, "library://favorites")
    assert fav_resource is not None
    result = await fav_resource.fn()
    data = json.loads(result)
    assert "favorites" in data
    assert "artists" in data["favorites"]
    assert "albums" in data["favorites"]
    assert "tracks" in data["favorites"]


async def test_recently_played_resource(mock_mass: Mock) -> None:
    """Test library://recently_played resource."""
    mcp = FastMCP("test")
    register_library_resources(mcp, mock_mass)

    recent_resource = _get_resource(mcp, "library://recently_played")
    assert recent_resource is not None
    result = await recent_resource.fn()
    data = json.loads(result)
    assert "recently_played" in data
    mock_mass.music.recently_played.assert_called()


async def test_providers_resource(mock_mass: Mock) -> None:
    """Test providers:// resource."""
    mcp = FastMCP("test")
    register_library_resources(mcp, mock_mass)

    providers_resource = _get_resource(mcp, "providers://")
    assert providers_resource is not None
    result = await providers_resource.fn()
    data = json.loads(result)
    assert "providers" in data
    assert len(data["providers"]) == 1
    assert data["providers"][0]["name"] == "Spotify"
