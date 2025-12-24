"""Tests for MCP Server tools."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import Mock

import pytest
from mcp.server.fastmcp import FastMCP

from music_assistant.providers.mcp_server import server
from music_assistant.providers.mcp_server.server import (
    _register_library_tools,
    _register_playback_tools,
    _register_player_tools,
    _register_playlist_tools,
    _register_queue_tools,
    _register_volume_tools,
)


def _get_tool(mcp: FastMCP, name: str) -> Any:
    """Get a tool by name from the MCP server."""
    for tool in mcp._tool_manager._tools.values():
        if tool.name == name:
            return tool
    return None


# =============================================================================
# PLAYBACK TOOLS
# =============================================================================


@pytest.mark.usefixtures("setup_mcp_state")
async def test_play(mock_mass: Mock) -> None:
    """Test play tool."""
    mcp = FastMCP("test")
    _register_playback_tools(mcp)

    play_tool = _get_tool(mcp, "play")
    assert play_tool is not None
    result = await play_tool.fn(player_id="player_1")
    assert "Playback started" in result
    mock_mass.player_queues.play.assert_called_once_with("player_1")


@pytest.mark.usefixtures("setup_mcp_state")
async def test_pause(mock_mass: Mock) -> None:
    """Test pause tool."""
    mcp = FastMCP("test")
    _register_playback_tools(mcp)

    pause_tool = _get_tool(mcp, "pause")
    assert pause_tool is not None
    result = await pause_tool.fn(player_id="player_1")
    assert "paused" in result
    mock_mass.player_queues.pause.assert_called_once_with("player_1")


@pytest.mark.usefixtures("setup_mcp_state")
async def test_stop(mock_mass: Mock) -> None:
    """Test stop tool."""
    mcp = FastMCP("test")
    _register_playback_tools(mcp)

    stop_tool = _get_tool(mcp, "stop")
    assert stop_tool is not None
    result = await stop_tool.fn(player_id="player_1")
    assert "stopped" in result
    mock_mass.player_queues.stop.assert_called_once_with("player_1")


@pytest.mark.usefixtures("setup_mcp_state")
async def test_next_track(mock_mass: Mock) -> None:
    """Test next_track tool."""
    mcp = FastMCP("test")
    _register_playback_tools(mcp)

    next_tool = _get_tool(mcp, "next_track")
    assert next_tool is not None
    result = await next_tool.fn(player_id="player_1")
    assert "next" in result.lower()
    mock_mass.player_queues.next.assert_called_once_with("player_1")


@pytest.mark.usefixtures("setup_mcp_state")
async def test_previous_track(mock_mass: Mock) -> None:
    """Test previous_track tool."""
    mcp = FastMCP("test")
    _register_playback_tools(mcp)

    prev_tool = _get_tool(mcp, "previous_track")
    assert prev_tool is not None
    result = await prev_tool.fn(player_id="player_1")
    assert "previous" in result.lower()
    mock_mass.player_queues.previous.assert_called_once_with("player_1")


@pytest.mark.usefixtures("setup_mcp_state")
async def test_seek(mock_mass: Mock) -> None:
    """Test seek tool."""
    mcp = FastMCP("test")
    _register_playback_tools(mcp)

    seek_tool = _get_tool(mcp, "seek")
    assert seek_tool is not None
    result = await seek_tool.fn(player_id="player_1", position=60)
    assert "60" in result
    mock_mass.player_queues.seek.assert_called_once_with("player_1", 60)


@pytest.mark.usefixtures("setup_mcp_state")
async def test_skip_forward(mock_mass: Mock) -> None:
    """Test skip_forward tool."""
    mcp = FastMCP("test")
    _register_playback_tools(mcp)

    skip_tool = _get_tool(mcp, "skip_forward")
    assert skip_tool is not None
    result = await skip_tool.fn(player_id="player_1", seconds=30)
    assert "30" in result
    mock_mass.player_queues.skip.assert_called_once_with("player_1", 30)


@pytest.mark.usefixtures("setup_mcp_state")
async def test_skip_backward(mock_mass: Mock) -> None:
    """Test skip_backward tool."""
    mcp = FastMCP("test")
    _register_playback_tools(mcp)

    skip_tool = _get_tool(mcp, "skip_backward")
    assert skip_tool is not None
    result = await skip_tool.fn(player_id="player_1", seconds=15)
    assert "15" in result
    mock_mass.player_queues.skip.assert_called_once_with("player_1", -15)


@pytest.mark.usefixtures("setup_mcp_state")
async def test_play_media(mock_mass: Mock) -> None:
    """Test play_media tool."""
    mcp = FastMCP("test")
    _register_playback_tools(mcp)

    play_media_tool = _get_tool(mcp, "play_media")
    assert play_media_tool is not None
    result = await play_media_tool.fn(
        player_id="player_1", uri="spotify://track/abc", enqueue_mode="play"
    )
    assert "Playing" in result
    mock_mass.player_queues.play_media.assert_called_once()


@pytest.mark.usefixtures("setup_mcp_state")
async def test_search_music(mock_mass: Mock) -> None:
    """Test search_music tool."""
    mcp = FastMCP("test")
    _register_playback_tools(mcp)

    search_tool = _get_tool(mcp, "search_music")
    assert search_tool is not None
    result = await search_tool.fn(query="test song")
    data = json.loads(result)
    assert "query" in data
    assert data["query"] == "test song"
    mock_mass.music.search.assert_called_once()


# =============================================================================
# QUEUE TOOLS
# =============================================================================


@pytest.mark.usefixtures("setup_mcp_state")
async def test_get_queue() -> None:
    """Test get_queue tool."""
    mcp = FastMCP("test")
    _register_queue_tools(mcp)

    queue_tool = _get_tool(mcp, "get_queue")
    assert queue_tool is not None
    result = await queue_tool.fn(player_id="player_1")
    data = json.loads(result)
    assert "queue_id" in data
    assert "items" in data


@pytest.mark.usefixtures("setup_mcp_state")
async def test_clear_queue(mock_mass: Mock) -> None:
    """Test clear_queue tool."""
    mcp = FastMCP("test")
    _register_queue_tools(mcp)

    clear_tool = _get_tool(mcp, "clear_queue")
    assert clear_tool is not None
    result = await clear_tool.fn(player_id="player_1")
    assert "cleared" in result.lower()
    mock_mass.player_queues.clear.assert_called_once_with("player_1")


@pytest.mark.usefixtures("setup_mcp_state")
async def test_shuffle_queue(mock_mass: Mock) -> None:
    """Test shuffle_queue tool."""
    mcp = FastMCP("test")
    _register_queue_tools(mcp)

    shuffle_tool = _get_tool(mcp, "shuffle_queue")
    assert shuffle_tool is not None
    result = await shuffle_tool.fn(player_id="player_1", enabled=True)
    assert "shuffle" in result.lower()
    mock_mass.player_queues.set_shuffle.assert_called_once_with("player_1", True)


@pytest.mark.usefixtures("setup_mcp_state")
async def test_repeat_queue(mock_mass: Mock) -> None:
    """Test repeat_queue tool."""
    mcp = FastMCP("test")
    _register_queue_tools(mcp)

    repeat_tool = _get_tool(mcp, "repeat_queue")
    assert repeat_tool is not None
    result = await repeat_tool.fn(player_id="player_1", mode="all")
    assert "repeat" in result.lower()
    mock_mass.player_queues.set_repeat.assert_called_once()


@pytest.mark.usefixtures("setup_mcp_state")
async def test_transfer_queue(mock_mass: Mock) -> None:
    """Test transfer_queue tool."""
    mcp = FastMCP("test")
    _register_queue_tools(mcp)

    transfer_tool = _get_tool(mcp, "transfer_queue")
    assert transfer_tool is not None
    result = await transfer_tool.fn(source_player_id="player_1", target_player_id="player_2")
    assert "transferred" in result.lower()
    mock_mass.player_queues.transfer_queue.assert_called_once_with("player_1", "player_2")


# =============================================================================
# VOLUME TOOLS
# =============================================================================


@pytest.mark.usefixtures("setup_mcp_state")
async def test_set_volume(mock_mass: Mock) -> None:
    """Test set_volume tool."""
    mcp = FastMCP("test")
    _register_volume_tools(mcp)

    vol_tool = _get_tool(mcp, "set_volume")
    assert vol_tool is not None
    result = await vol_tool.fn(player_id="player_1", volume=75)
    assert "75" in result
    mock_mass.players.cmd_volume_set.assert_called_once_with("player_1", 75)


@pytest.mark.usefixtures("setup_mcp_state")
async def test_set_volume_clamped(mock_mass: Mock) -> None:
    """Test set_volume clamps values to 0-100."""
    mcp = FastMCP("test")
    _register_volume_tools(mcp)

    vol_tool = _get_tool(mcp, "set_volume")
    assert vol_tool is not None
    await vol_tool.fn(player_id="player_1", volume=150)
    mock_mass.players.cmd_volume_set.assert_called_with("player_1", 100)


@pytest.mark.usefixtures("setup_mcp_state")
async def test_volume_up(mock_mass: Mock) -> None:
    """Test volume_up tool."""
    mcp = FastMCP("test")
    _register_volume_tools(mcp)

    vol_tool = _get_tool(mcp, "volume_up")
    assert vol_tool is not None
    result = await vol_tool.fn(player_id="player_1")
    assert "increased" in result.lower()
    mock_mass.players.cmd_volume_up.assert_called_once_with("player_1")


@pytest.mark.usefixtures("setup_mcp_state")
async def test_volume_down(mock_mass: Mock) -> None:
    """Test volume_down tool."""
    mcp = FastMCP("test")
    _register_volume_tools(mcp)

    vol_tool = _get_tool(mcp, "volume_down")
    assert vol_tool is not None
    result = await vol_tool.fn(player_id="player_1")
    assert "decreased" in result.lower()
    mock_mass.players.cmd_volume_down.assert_called_once_with("player_1")


@pytest.mark.usefixtures("setup_mcp_state")
async def test_mute(mock_mass: Mock) -> None:
    """Test mute tool."""
    mcp = FastMCP("test")
    _register_volume_tools(mcp)

    mute_tool = _get_tool(mcp, "mute")
    assert mute_tool is not None
    result = await mute_tool.fn(player_id="player_1", muted=True)
    assert "muted" in result.lower()
    mock_mass.players.cmd_volume_mute.assert_called_once_with("player_1", True)


# =============================================================================
# PLAYER TOOLS
# =============================================================================


@pytest.mark.usefixtures("setup_mcp_state")
async def test_power_player(mock_mass: Mock) -> None:
    """Test power_player tool."""
    mcp = FastMCP("test")
    _register_player_tools(mcp)

    power_tool = _get_tool(mcp, "power_player")
    assert power_tool is not None
    result = await power_tool.fn(player_id="player_1", powered=True)
    assert "powered" in result.lower()
    mock_mass.players.cmd_power.assert_called_once_with("player_1", True)


@pytest.mark.usefixtures("setup_mcp_state")
async def test_group_players(mock_mass: Mock) -> None:
    """Test group_players tool."""
    mcp = FastMCP("test")
    _register_player_tools(mcp)

    group_tool = _get_tool(mcp, "group_players")
    assert group_tool is not None
    result = await group_tool.fn(target_player_id="player_1", child_player_ids="player_2,player_3")
    assert "grouped" in result.lower()
    mock_mass.players.cmd_group_many.assert_called_once_with("player_1", ["player_2", "player_3"])


@pytest.mark.usefixtures("setup_mcp_state")
async def test_ungroup_player(mock_mass: Mock) -> None:
    """Test ungroup_player tool."""
    mcp = FastMCP("test")
    _register_player_tools(mcp)

    ungroup_tool = _get_tool(mcp, "ungroup_player")
    assert ungroup_tool is not None
    result = await ungroup_tool.fn(player_id="player_1")
    assert "ungrouped" in result.lower()
    mock_mass.players.cmd_ungroup.assert_called_once_with("player_1")


@pytest.mark.usefixtures("setup_mcp_state")
async def test_get_player_by_name() -> None:
    """Test get_player_by_name tool."""
    mcp = FastMCP("test")
    _register_player_tools(mcp)

    find_tool = _get_tool(mcp, "get_player_by_name")
    assert find_tool is not None
    result = await find_tool.fn(name="Living")
    data = json.loads(result)
    assert "matches" in data
    assert len(data["matches"]) == 1
    assert data["matches"][0]["name"] == "Living Room"


# =============================================================================
# LIBRARY TOOLS
# =============================================================================


@pytest.mark.usefixtures("setup_mcp_state")
async def test_get_recently_played(mock_mass: Mock) -> None:
    """Test get_recently_played tool."""
    mcp = FastMCP("test")
    _register_library_tools(mcp)

    recent_tool = _get_tool(mcp, "get_recently_played")
    assert recent_tool is not None
    result = await recent_tool.fn()
    data = json.loads(result)
    assert "recently_played" in data
    mock_mass.music.recently_played.assert_called_once()


@pytest.mark.usefixtures("setup_mcp_state")
async def test_browse_library(mock_mass: Mock) -> None:
    """Test browse_library tool."""
    mcp = FastMCP("test")
    _register_library_tools(mcp)

    browse_tool = _get_tool(mcp, "browse_library")
    assert browse_tool is not None
    result = await browse_tool.fn(path="library://artists")
    data = json.loads(result)
    assert "items" in data
    mock_mass.music.browse.assert_called_once()


@pytest.mark.usefixtures("setup_mcp_state")
async def test_add_to_favorites(mock_mass: Mock) -> None:
    """Test add_to_favorites tool."""
    mcp = FastMCP("test")
    _register_library_tools(mcp)

    fav_tool = _get_tool(mcp, "add_to_favorites")
    assert fav_tool is not None
    result = await fav_tool.fn(uri="library://track/123")
    assert "favorites" in result.lower()
    mock_mass.music.add_item_to_favorites.assert_called_once()


# =============================================================================
# PLAYLIST TOOLS
# =============================================================================


@pytest.mark.usefixtures("setup_mcp_state")
async def test_get_playlists(mock_mass: Mock) -> None:
    """Test get_playlists tool."""
    mcp = FastMCP("test")
    _register_playlist_tools(mcp)

    pl_tool = _get_tool(mcp, "get_playlists")
    assert pl_tool is not None
    result = await pl_tool.fn()
    data = json.loads(result)
    assert "playlists" in data
    mock_mass.music.playlists.library_items.assert_called_once()


@pytest.mark.usefixtures("setup_mcp_state")
async def test_create_playlist(mock_mass: Mock) -> None:
    """Test create_playlist tool."""
    mcp = FastMCP("test")
    _register_playlist_tools(mcp)

    create_tool = _get_tool(mcp, "create_playlist")
    assert create_tool is not None
    result = await create_tool.fn(name="My New Playlist")
    data = json.loads(result)
    assert data["created"] is True
    mock_mass.music.playlists.create_playlist.assert_called_once_with("My New Playlist")


# =============================================================================
# ERROR HANDLING
# =============================================================================


async def test_tool_without_mass_initialized() -> None:
    """Test tools return error when MusicAssistant not initialized."""
    server._state["mass"] = None

    mcp = FastMCP("test")
    _register_playback_tools(mcp)

    play_tool = _get_tool(mcp, "play")
    assert play_tool is not None
    result = await play_tool.fn(player_id="player_1")
    assert "Error" in result
    assert "not initialized" in result
