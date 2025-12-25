"""Tests for MCP Server prompts."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

from mcp.server.fastmcp import FastMCP

from music_assistant.providers.mcp_server.prompts import register_prompts


def _get_prompt(mcp: FastMCP, name: str) -> Any:
    """Get a prompt by name from the MCP server."""
    for prompt in mcp._prompt_manager._prompts.values():
        if prompt.name == name:
            return prompt
    return None


# =============================================================================
# PROMPT TESTS
# =============================================================================


async def test_play_music_prompt(mock_mass: Mock) -> None:
    """Test play_music prompt."""
    mcp = FastMCP("test")
    register_prompts(mcp, mock_mass)

    play_prompt = _get_prompt(mcp, "play_music")
    assert play_prompt is not None
    result = await play_prompt.fn(query="Beatles", player="Living Room")
    assert "Beatles" in result
    assert "Living Room" in result


async def test_play_music_prompt_with_players_list(mock_mass: Mock) -> None:
    """Test play_music prompt includes available players."""
    mcp = FastMCP("test")
    register_prompts(mcp, mock_mass)

    play_prompt = _get_prompt(mcp, "play_music")
    assert play_prompt is not None
    result = await play_prompt.fn()
    assert "Available players" in result
    assert "Living Room" in result


async def test_whats_playing_prompt(mock_mass: Mock) -> None:
    """Test whats_playing prompt."""
    mcp = FastMCP("test")
    register_prompts(mcp, mock_mass)

    whats_prompt = _get_prompt(mcp, "whats_playing")
    assert whats_prompt is not None
    result = await whats_prompt.fn(player="Living Room")
    assert "Living Room" in result


async def test_whats_playing_prompt_shows_current_track(mock_mass: Mock) -> None:
    """Test whats_playing prompt shows current track when player specified."""
    mcp = FastMCP("test")
    register_prompts(mcp, mock_mass)

    whats_prompt = _get_prompt(mcp, "whats_playing")
    assert whats_prompt is not None
    result = await whats_prompt.fn(player="Living")
    assert "Test Track" in result


async def test_control_playback_prompt(mock_mass: Mock) -> None:
    """Test control_playback prompt."""
    mcp = FastMCP("test")
    register_prompts(mcp, mock_mass)

    control_prompt = _get_prompt(mcp, "control_playback")
    assert control_prompt is not None
    result = await control_prompt.fn(player="Kitchen", action="pause")
    assert "pause" in result
    assert "Kitchen" in result


async def test_discover_music_prompt(mock_mass: Mock) -> None:
    """Test discover_music prompt."""
    mcp = FastMCP("test")
    register_prompts(mcp, mock_mass)

    discover_prompt = _get_prompt(mcp, "discover_music")
    assert discover_prompt is not None
    result = await discover_prompt.fn(mood="relaxing", genre="jazz")
    assert "relaxing" in result
    assert "jazz" in result


async def test_manage_queue_prompt(mock_mass: Mock) -> None:
    """Test manage_queue prompt."""
    mcp = FastMCP("test")
    register_prompts(mcp, mock_mass)

    queue_prompt = _get_prompt(mcp, "manage_queue")
    assert queue_prompt is not None
    result = await queue_prompt.fn(player="Living Room")
    assert "queue" in result.lower()
    assert "Living Room" in result


async def test_setup_multiroom_prompt(mock_mass: Mock) -> None:
    """Test setup_multiroom prompt."""
    mcp = FastMCP("test")
    register_prompts(mcp, mock_mass)

    multiroom_prompt = _get_prompt(mcp, "setup_multiroom")
    assert multiroom_prompt is not None
    result = await multiroom_prompt.fn(rooms="Kitchen, Living Room")
    assert "Kitchen" in result
    assert "Living Room" in result


async def test_setup_multiroom_prompt_shows_speakers(mock_mass: Mock) -> None:
    """Test setup_multiroom prompt includes available speakers."""
    mcp = FastMCP("test")
    register_prompts(mcp, mock_mass)

    multiroom_prompt = _get_prompt(mcp, "setup_multiroom")
    assert multiroom_prompt is not None
    result = await multiroom_prompt.fn()
    assert "Available speakers" in result


async def test_transfer_playback_prompt(mock_mass: Mock) -> None:
    """Test transfer_playback prompt."""
    mcp = FastMCP("test")
    register_prompts(mcp, mock_mass)

    transfer_prompt = _get_prompt(mcp, "transfer_playback")
    assert transfer_prompt is not None
    result = await transfer_prompt.fn(from_player="Kitchen", to_player="Bedroom")
    assert "Kitchen" in result
    assert "Bedroom" in result


async def test_transfer_playback_to_only(mock_mass: Mock) -> None:
    """Test transfer_playback prompt with only to_player."""
    mcp = FastMCP("test")
    register_prompts(mcp, mock_mass)

    transfer_prompt = _get_prompt(mcp, "transfer_playback")
    assert transfer_prompt is not None
    result = await transfer_prompt.fn(to_player="Living Room")
    assert "continue listening" in result
    assert "Living Room" in result
