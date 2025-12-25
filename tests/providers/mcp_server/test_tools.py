"""Tests for MCP Server tools."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, Mock

from mcp.server.fastmcp import FastMCP

from music_assistant.providers.mcp_server.tools import (
    register_library_delete_tools,
    register_library_edit_tools,
    register_library_query_tools,
    register_playback_control_tools,
    register_playback_query_tools,
    register_player_control_tools,
    register_player_query_tools,
    register_playlist_delete_tools,
    register_playlist_edit_tools,
    register_playlist_query_tools,
    register_queue_control_tools,
    register_queue_delete_tools,
    register_queue_edit_tools,
    register_queue_query_tools,
    register_volume_control_tools,
)
from music_assistant.providers.mcp_server.tools.media import (
    register_audiobook_tools,
    register_podcast_tools,
    register_radio_tools,
)
from music_assistant.providers.mcp_server.tools.metadata import register_metadata_tools


def _get_tool(mcp: FastMCP, name: str) -> Any:
    """Get a tool by name from the MCP server."""
    for tool in mcp._tool_manager._tools.values():
        if tool.name == name:
            return tool
    return None


# =============================================================================
# PLAYBACK TOOLS
# =============================================================================


async def test_play(mock_mass: Mock) -> None:
    """Test play tool."""
    mcp = FastMCP("test")
    register_playback_control_tools(mcp, mock_mass)

    play_tool = _get_tool(mcp, "play")
    assert play_tool is not None
    result = await play_tool.fn(player_id="player_1")
    assert "Playback started" in result
    mock_mass.player_queues.play.assert_called_once_with("player_1")


async def test_pause(mock_mass: Mock) -> None:
    """Test pause tool."""
    mcp = FastMCP("test")
    register_playback_control_tools(mcp, mock_mass)

    pause_tool = _get_tool(mcp, "pause")
    assert pause_tool is not None
    result = await pause_tool.fn(player_id="player_1")
    assert "paused" in result
    mock_mass.player_queues.pause.assert_called_once_with("player_1")


async def test_stop(mock_mass: Mock) -> None:
    """Test stop tool."""
    mcp = FastMCP("test")
    register_playback_control_tools(mcp, mock_mass)

    stop_tool = _get_tool(mcp, "stop")
    assert stop_tool is not None
    result = await stop_tool.fn(player_id="player_1")
    assert "stopped" in result
    mock_mass.player_queues.stop.assert_called_once_with("player_1")


async def test_next_track(mock_mass: Mock) -> None:
    """Test next_track tool."""
    mcp = FastMCP("test")
    register_playback_control_tools(mcp, mock_mass)

    next_tool = _get_tool(mcp, "next_track")
    assert next_tool is not None
    result = await next_tool.fn(player_id="player_1")
    assert "next" in result.lower()
    mock_mass.player_queues.next.assert_called_once_with("player_1")


async def test_previous_track(mock_mass: Mock) -> None:
    """Test previous_track tool."""
    mcp = FastMCP("test")
    register_playback_control_tools(mcp, mock_mass)

    prev_tool = _get_tool(mcp, "previous_track")
    assert prev_tool is not None
    result = await prev_tool.fn(player_id="player_1")
    assert "previous" in result.lower()
    mock_mass.player_queues.previous.assert_called_once_with("player_1")


async def test_seek(mock_mass: Mock) -> None:
    """Test seek tool."""
    mcp = FastMCP("test")
    register_playback_control_tools(mcp, mock_mass)

    seek_tool = _get_tool(mcp, "seek")
    assert seek_tool is not None
    result = await seek_tool.fn(player_id="player_1", position=60)
    assert "60" in result
    mock_mass.player_queues.seek.assert_called_once_with("player_1", 60)


async def test_skip_forward(mock_mass: Mock) -> None:
    """Test skip_forward tool."""
    mcp = FastMCP("test")
    register_playback_control_tools(mcp, mock_mass)

    skip_tool = _get_tool(mcp, "skip_forward")
    assert skip_tool is not None
    result = await skip_tool.fn(player_id="player_1", seconds=30)
    assert "30" in result
    mock_mass.player_queues.skip.assert_called_once_with("player_1", 30)


async def test_skip_backward(mock_mass: Mock) -> None:
    """Test skip_backward tool."""
    mcp = FastMCP("test")
    register_playback_control_tools(mcp, mock_mass)

    skip_tool = _get_tool(mcp, "skip_backward")
    assert skip_tool is not None
    result = await skip_tool.fn(player_id="player_1", seconds=15)
    assert "15" in result
    mock_mass.player_queues.skip.assert_called_once_with("player_1", -15)


async def test_play_media(mock_mass: Mock) -> None:
    """Test play_media tool."""
    mcp = FastMCP("test")
    register_playback_control_tools(mcp, mock_mass)

    play_media_tool = _get_tool(mcp, "play_media")
    assert play_media_tool is not None
    result = await play_media_tool.fn(
        player_id="player_1", uri="spotify://track/abc", enqueue_mode="play"
    )
    assert "Playing" in result
    mock_mass.player_queues.play_media.assert_called_once()


async def test_search_music(mock_mass: Mock) -> None:
    """Test search_music tool."""
    mcp = FastMCP("test")
    register_playback_query_tools(mcp, mock_mass)

    search_tool = _get_tool(mcp, "search_music")
    assert search_tool is not None
    result = await search_tool.fn(query="test song")
    data = json.loads(result)
    assert data["query"] == "test song"
    assert "results" in data
    # Verify search results contain tracks from mock
    assert "tracks" in data["results"]
    assert len(data["results"]["tracks"]) > 0
    assert data["results"]["tracks"][0]["name"] == "Test Track"
    mock_mass.music.search.assert_called_once()


# =============================================================================
# QUEUE TOOLS
# =============================================================================


async def test_get_queue(mock_mass: Mock) -> None:
    """Test get_queue tool."""
    mcp = FastMCP("test")
    register_queue_query_tools(mcp, mock_mass)

    queue_tool = _get_tool(mcp, "get_queue")
    assert queue_tool is not None
    result = await queue_tool.fn(player_id="player_1")
    data = json.loads(result)
    assert data["queue_id"] == "player_1"
    assert "items" in data
    assert "current_index" in data
    # Verify items structure from mock
    assert len(data["items"]) > 0
    assert data["items"][0]["name"] == "Test Track"


async def test_clear_queue(mock_mass: Mock) -> None:
    """Test clear_queue tool."""
    mcp = FastMCP("test")
    register_queue_delete_tools(mcp, mock_mass)

    clear_tool = _get_tool(mcp, "clear_queue")
    assert clear_tool is not None
    result = await clear_tool.fn(player_id="player_1")
    assert "cleared" in result.lower()
    mock_mass.player_queues.clear.assert_called_once_with("player_1")


async def test_shuffle_queue(mock_mass: Mock) -> None:
    """Test shuffle_queue tool."""
    mcp = FastMCP("test")
    register_queue_control_tools(mcp, mock_mass)

    shuffle_tool = _get_tool(mcp, "shuffle_queue")
    assert shuffle_tool is not None
    result = await shuffle_tool.fn(player_id="player_1", enabled=True)
    assert "shuffle" in result.lower()
    mock_mass.player_queues.set_shuffle.assert_called_once_with("player_1", True)


async def test_repeat_queue(mock_mass: Mock) -> None:
    """Test repeat_queue tool."""
    mcp = FastMCP("test")
    register_queue_control_tools(mcp, mock_mass)

    repeat_tool = _get_tool(mcp, "repeat_queue")
    assert repeat_tool is not None
    result = await repeat_tool.fn(player_id="player_1", mode="all")
    assert "repeat" in result.lower()
    mock_mass.player_queues.set_repeat.assert_called_once()


async def test_transfer_queue(mock_mass: Mock) -> None:
    """Test transfer_queue tool."""
    mcp = FastMCP("test")
    register_queue_control_tools(mcp, mock_mass)

    transfer_tool = _get_tool(mcp, "transfer_queue")
    assert transfer_tool is not None
    result = await transfer_tool.fn(source_player_id="player_1", target_player_id="player_2")
    assert "transferred" in result.lower()
    mock_mass.player_queues.transfer_queue.assert_called_once_with("player_1", "player_2")


async def test_move_queue_item(mock_mass: Mock) -> None:
    """Test move_queue_item tool."""
    mcp = FastMCP("test")
    register_queue_edit_tools(mcp, mock_mass)

    move_tool = _get_tool(mcp, "move_queue_item")
    assert move_tool is not None
    result = await move_tool.fn(player_id="player_1", queue_item_id="qi_1", position_shift=-2)
    assert "up" in result.lower()
    mock_mass.player_queues.move_item.assert_called_once_with("player_1", "qi_1", -2)


async def test_move_queue_item_down(mock_mass: Mock) -> None:
    """Test move_queue_item tool moving down."""
    mcp = FastMCP("test")
    register_queue_edit_tools(mcp, mock_mass)

    move_tool = _get_tool(mcp, "move_queue_item")
    assert move_tool is not None
    result = await move_tool.fn(player_id="player_1", queue_item_id="qi_1", position_shift=3)
    assert "down" in result.lower()
    mock_mass.player_queues.move_item.assert_called_once_with("player_1", "qi_1", 3)


async def test_remove_queue_item(mock_mass: Mock) -> None:
    """Test remove_queue_item tool."""
    mcp = FastMCP("test")
    register_queue_delete_tools(mcp, mock_mass)

    remove_tool = _get_tool(mcp, "remove_queue_item")
    assert remove_tool is not None
    result = await remove_tool.fn(player_id="player_1", queue_item_id="qi_1")
    assert "removed" in result.lower()
    mock_mass.player_queues.delete_item.assert_called_once_with("player_1", "qi_1")


async def test_play_queue_index(mock_mass: Mock) -> None:
    """Test play_queue_index tool."""
    mcp = FastMCP("test")
    register_queue_control_tools(mcp, mock_mass)

    play_idx_tool = _get_tool(mcp, "play_queue_index")
    assert play_idx_tool is not None
    result = await play_idx_tool.fn(player_id="player_1", index=5)
    assert "5" in result
    mock_mass.player_queues.play_index.assert_called_once_with("player_1", 5)


# =============================================================================
# VOLUME TOOLS
# =============================================================================


async def test_set_volume(mock_mass: Mock) -> None:
    """Test set_volume tool."""
    mcp = FastMCP("test")
    register_volume_control_tools(mcp, mock_mass)

    vol_tool = _get_tool(mcp, "set_volume")
    assert vol_tool is not None
    result = await vol_tool.fn(player_id="player_1", volume=75)
    assert "75" in result
    mock_mass.players.cmd_volume_set.assert_called_once_with("player_1", 75)


async def test_set_volume_clamped(mock_mass: Mock) -> None:
    """Test set_volume clamps values to 0-100."""
    mcp = FastMCP("test")
    register_volume_control_tools(mcp, mock_mass)

    vol_tool = _get_tool(mcp, "set_volume")
    assert vol_tool is not None
    await vol_tool.fn(player_id="player_1", volume=150)
    mock_mass.players.cmd_volume_set.assert_called_with("player_1", 100)


async def test_volume_up(mock_mass: Mock) -> None:
    """Test volume_up tool."""
    mcp = FastMCP("test")
    register_volume_control_tools(mcp, mock_mass)

    vol_tool = _get_tool(mcp, "volume_up")
    assert vol_tool is not None
    result = await vol_tool.fn(player_id="player_1")
    assert "increased" in result.lower()
    mock_mass.players.cmd_volume_up.assert_called_once_with("player_1")


async def test_volume_down(mock_mass: Mock) -> None:
    """Test volume_down tool."""
    mcp = FastMCP("test")
    register_volume_control_tools(mcp, mock_mass)

    vol_tool = _get_tool(mcp, "volume_down")
    assert vol_tool is not None
    result = await vol_tool.fn(player_id="player_1")
    assert "decreased" in result.lower()
    mock_mass.players.cmd_volume_down.assert_called_once_with("player_1")


async def test_mute(mock_mass: Mock) -> None:
    """Test mute tool."""
    mcp = FastMCP("test")
    register_volume_control_tools(mcp, mock_mass)

    mute_tool = _get_tool(mcp, "mute")
    assert mute_tool is not None
    result = await mute_tool.fn(player_id="player_1", muted=True)
    assert "muted" in result.lower()
    mock_mass.players.cmd_volume_mute.assert_called_once_with("player_1", True)


async def test_set_group_volume(mock_mass: Mock) -> None:
    """Test set_group_volume tool."""
    mcp = FastMCP("test")
    register_volume_control_tools(mcp, mock_mass)

    vol_tool = _get_tool(mcp, "set_group_volume")
    assert vol_tool is not None
    result = await vol_tool.fn(player_id="player_1", volume=60)
    assert "60" in result
    assert "group" in result.lower()
    mock_mass.players.cmd_group_volume.assert_called_once_with("player_1", 60)


async def test_set_group_volume_clamped(mock_mass: Mock) -> None:
    """Test set_group_volume clamps values to 0-100."""
    mcp = FastMCP("test")
    register_volume_control_tools(mcp, mock_mass)

    vol_tool = _get_tool(mcp, "set_group_volume")
    assert vol_tool is not None
    await vol_tool.fn(player_id="player_1", volume=-10)
    mock_mass.players.cmd_group_volume.assert_called_with("player_1", 0)


# =============================================================================
# PLAYER TOOLS
# =============================================================================


async def test_power_player(mock_mass: Mock) -> None:
    """Test power_player tool."""
    mcp = FastMCP("test")
    register_player_control_tools(mcp, mock_mass)

    power_tool = _get_tool(mcp, "power_player")
    assert power_tool is not None
    result = await power_tool.fn(player_id="player_1", powered=True)
    assert "powered" in result.lower()
    mock_mass.players.cmd_power.assert_called_once_with("player_1", True)


async def test_group_players(mock_mass: Mock) -> None:
    """Test group_players tool."""
    mcp = FastMCP("test")
    register_player_control_tools(mcp, mock_mass)

    group_tool = _get_tool(mcp, "group_players")
    assert group_tool is not None
    result = await group_tool.fn(target_player_id="player_1", child_player_ids="player_2,player_3")
    assert "grouped" in result.lower()
    mock_mass.players.cmd_group_many.assert_called_once_with("player_1", ["player_2", "player_3"])


async def test_ungroup_player(mock_mass: Mock) -> None:
    """Test ungroup_player tool."""
    mcp = FastMCP("test")
    register_player_control_tools(mcp, mock_mass)

    ungroup_tool = _get_tool(mcp, "ungroup_player")
    assert ungroup_tool is not None
    result = await ungroup_tool.fn(player_id="player_1")
    assert "ungrouped" in result.lower()
    mock_mass.players.cmd_ungroup.assert_called_once_with("player_1")


async def test_get_player_by_name(mock_mass: Mock) -> None:
    """Test get_player_by_name tool."""
    mcp = FastMCP("test")
    register_player_query_tools(mcp, mock_mass)

    find_tool = _get_tool(mcp, "get_player_by_name")
    assert find_tool is not None
    result = await find_tool.fn(name="Living")
    data = json.loads(result)
    assert "matches" in data
    assert len(data["matches"]) == 1
    assert data["matches"][0]["name"] == "Living Room"


async def test_play_announcement(mock_mass: Mock) -> None:
    """Test play_announcement tool."""
    mcp = FastMCP("test")
    register_player_control_tools(mcp, mock_mass)

    announce_tool = _get_tool(mcp, "play_announcement")
    assert announce_tool is not None
    result = await announce_tool.fn(
        player_id="player_1", url="http://example.com/audio.mp3", volume=80
    )
    assert "announcement" in result.lower()
    mock_mass.players.play_announcement.assert_called_once_with(
        "player_1", "http://example.com/audio.mp3", volume_level=80
    )


# =============================================================================
# LIBRARY TOOLS
# =============================================================================


async def test_get_recently_played(mock_mass: Mock) -> None:
    """Test get_recently_played tool."""
    mcp = FastMCP("test")
    register_library_query_tools(mcp, mock_mass)

    recent_tool = _get_tool(mcp, "get_recently_played")
    assert recent_tool is not None
    result = await recent_tool.fn()
    data = json.loads(result)
    assert "recently_played" in data
    mock_mass.music.recently_played.assert_called_once()


async def test_browse_library(mock_mass: Mock) -> None:
    """Test browse_library tool."""
    mcp = FastMCP("test")
    register_library_query_tools(mcp, mock_mass)

    browse_tool = _get_tool(mcp, "browse_library")
    assert browse_tool is not None
    result = await browse_tool.fn(path="library://artists")
    data = json.loads(result)
    assert "items" in data
    mock_mass.music.browse.assert_called_once()


async def test_get_in_progress_items(mock_mass: Mock) -> None:
    """Test get_in_progress_items tool."""
    mcp = FastMCP("test")
    register_library_query_tools(mcp, mock_mass)

    in_progress_tool = _get_tool(mcp, "get_in_progress_items")
    assert in_progress_tool is not None
    result = await in_progress_tool.fn(limit=10)
    data = json.loads(result)
    assert "in_progress" in data
    mock_mass.music.in_progress_items.assert_called_once_with(limit=10)


async def test_add_to_favorites(mock_mass: Mock) -> None:
    """Test add_to_favorites tool."""
    mcp = FastMCP("test")
    register_library_edit_tools(mcp, mock_mass)

    fav_tool = _get_tool(mcp, "add_to_favorites")
    assert fav_tool is not None
    result = await fav_tool.fn(uri="library://track/123")
    assert "favorites" in result.lower()
    mock_mass.music.add_item_to_favorites.assert_called_once()


async def test_remove_from_favorites(mock_mass: Mock) -> None:
    """Test remove_from_favorites tool."""
    mcp = FastMCP("test")
    register_library_delete_tools(mcp, mock_mass)

    fav_tool = _get_tool(mcp, "remove_from_favorites")
    assert fav_tool is not None
    result = await fav_tool.fn(uri="library://track/123")
    assert "removed" in result.lower()
    mock_mass.music.remove_item_from_favorites.assert_called_once()


async def test_get_recommendations(mock_mass: Mock) -> None:
    """Test get_recommendations tool."""
    mcp = FastMCP("test")
    register_library_query_tools(mcp, mock_mass)

    rec_tool = _get_tool(mcp, "get_recommendations")
    assert rec_tool is not None
    result = await rec_tool.fn()
    data = json.loads(result)
    assert "recommendations" in data
    mock_mass.music.recommendations.assert_called_once()


async def test_get_recently_added(mock_mass: Mock) -> None:
    """Test get_recently_added tool."""
    mcp = FastMCP("test")
    register_library_query_tools(mcp, mock_mass)

    recent_tool = _get_tool(mcp, "get_recently_added")
    assert recent_tool is not None
    result = await recent_tool.fn(limit=10)
    data = json.loads(result)
    assert "recently_added" in data
    mock_mass.music.recently_added_tracks.assert_called_once_with(limit=10)


async def test_get_similar_tracks(mock_mass: Mock) -> None:
    """Test get_similar_tracks tool."""
    mcp = FastMCP("test")
    register_library_query_tools(mcp, mock_mass)

    similar_tool = _get_tool(mcp, "get_similar_tracks")
    assert similar_tool is not None
    result = await similar_tool.fn(track_uri="library://track/123", limit=15)
    data = json.loads(result)
    assert "similar_tracks" in data
    mock_mass.music.tracks.similar_tracks.assert_called_once()


async def test_get_artist_tracks(mock_mass: Mock) -> None:
    """Test get_artist_tracks tool."""
    mcp = FastMCP("test")
    register_library_query_tools(mcp, mock_mass)

    artist_tool = _get_tool(mcp, "get_artist_tracks")
    assert artist_tool is not None
    result = await artist_tool.fn(artist_uri="library://artist/456")
    data = json.loads(result)
    assert "tracks" in data
    mock_mass.music.artists.tracks.assert_called_once()


async def test_get_artist_albums(mock_mass: Mock) -> None:
    """Test get_artist_albums tool."""
    mcp = FastMCP("test")
    register_library_query_tools(mcp, mock_mass)

    artist_tool = _get_tool(mcp, "get_artist_albums")
    assert artist_tool is not None
    result = await artist_tool.fn(artist_uri="library://artist/456")
    data = json.loads(result)
    assert "albums" in data
    mock_mass.music.artists.albums.assert_called_once()


async def test_get_album_tracks(mock_mass: Mock) -> None:
    """Test get_album_tracks tool."""
    mcp = FastMCP("test")
    register_library_query_tools(mcp, mock_mass)

    album_tool = _get_tool(mcp, "get_album_tracks")
    assert album_tool is not None
    result = await album_tool.fn(album_uri="library://album/789")
    data = json.loads(result)
    assert "tracks" in data
    mock_mass.music.albums.tracks.assert_called_once()


async def test_add_to_library(mock_mass: Mock) -> None:
    """Test add_to_library tool."""
    mcp = FastMCP("test")
    register_library_edit_tools(mcp, mock_mass)

    lib_tool = _get_tool(mcp, "add_to_library")
    assert lib_tool is not None
    result = await lib_tool.fn(uri="spotify://track/abc")
    assert "library" in result.lower()
    mock_mass.music.add_item_to_library.assert_called_once_with("spotify://track/abc")


async def test_remove_from_library(mock_mass: Mock) -> None:
    """Test remove_from_library tool."""
    mcp = FastMCP("test")
    register_library_delete_tools(mcp, mock_mass)

    lib_tool = _get_tool(mcp, "remove_from_library")
    assert lib_tool is not None
    result = await lib_tool.fn(uri="library://track/123")
    assert "removed" in result.lower()
    mock_mass.music.remove_item_from_library.assert_called_once()


async def test_get_library_artists(mock_mass: Mock) -> None:
    """Test get_library_artists tool."""
    mcp = FastMCP("test")
    register_library_query_tools(mcp, mock_mass)

    lib_tool = _get_tool(mcp, "get_library_artists")
    assert lib_tool is not None
    result = await lib_tool.fn(limit=25)
    data = json.loads(result)
    assert "artists" in data
    mock_mass.music.artists.library_items.assert_called()


async def test_get_library_albums(mock_mass: Mock) -> None:
    """Test get_library_albums tool."""
    mcp = FastMCP("test")
    register_library_query_tools(mcp, mock_mass)

    lib_tool = _get_tool(mcp, "get_library_albums")
    assert lib_tool is not None
    result = await lib_tool.fn(limit=25)
    data = json.loads(result)
    assert "albums" in data
    mock_mass.music.albums.library_items.assert_called()


async def test_get_library_tracks(mock_mass: Mock) -> None:
    """Test get_library_tracks tool."""
    mcp = FastMCP("test")
    register_library_query_tools(mcp, mock_mass)

    lib_tool = _get_tool(mcp, "get_library_tracks")
    assert lib_tool is not None
    result = await lib_tool.fn(limit=25)
    data = json.loads(result)
    assert "tracks" in data
    mock_mass.music.tracks.library_items.assert_called()


# =============================================================================
# ADVANCED SEARCH/FILTER TOOLS
# =============================================================================


async def test_get_library_tracks_with_order_by(mock_mass: Mock) -> None:
    """Test get_library_tracks with order_by parameter."""
    mcp = FastMCP("test")
    register_library_query_tools(mcp, mock_mass)

    tool = _get_tool(mcp, "get_library_tracks")
    assert tool is not None
    result = await tool.fn(order_by="play_count_desc")
    data = json.loads(result)
    assert "tracks" in data
    mock_mass.music.tracks.library_items.assert_called_with(
        search=None,
        limit=50,
        favorite=None,
        order_by="play_count_desc",
        provider=None,
    )


async def test_get_library_tracks_with_provider(mock_mass: Mock) -> None:
    """Test get_library_tracks with provider filter."""
    mcp = FastMCP("test")
    register_library_query_tools(mcp, mock_mass)

    tool = _get_tool(mcp, "get_library_tracks")
    assert tool is not None
    result = await tool.fn(provider="spotify_1")
    data = json.loads(result)
    assert "tracks" in data
    mock_mass.music.tracks.library_items.assert_called_with(
        search=None,
        limit=50,
        favorite=None,
        order_by="sort_name",
        provider="spotify_1",
    )


async def test_get_library_tracks_invalid_order_by(mock_mass: Mock) -> None:
    """Test get_library_tracks with invalid order_by returns error."""
    mcp = FastMCP("test")
    register_library_query_tools(mcp, mock_mass)

    tool = _get_tool(mcp, "get_library_tracks")
    assert tool is not None
    result = await tool.fn(order_by="invalid_sort")
    assert "error" in result.lower()
    assert "invalid order_by" in result.lower()


async def test_get_library_albums_with_sorting(mock_mass: Mock) -> None:
    """Test get_library_albums with extended sort options."""
    mcp = FastMCP("test")
    register_library_query_tools(mcp, mock_mass)

    tool = _get_tool(mcp, "get_library_albums")
    assert tool is not None
    result = await tool.fn(order_by="year_desc")
    data = json.loads(result)
    assert "albums" in data
    mock_mass.music.albums.library_items.assert_called_with(
        search=None,
        limit=50,
        favorite=None,
        order_by="year_desc",
        provider=None,
    )


async def test_get_library_artists_with_provider(mock_mass: Mock) -> None:
    """Test get_library_artists with provider filter."""
    mcp = FastMCP("test")
    register_library_query_tools(mcp, mock_mass)

    tool = _get_tool(mcp, "get_library_artists")
    assert tool is not None
    result = await tool.fn(provider="spotify_1", order_by="timestamp_added_desc")
    data = json.loads(result)
    assert "artists" in data
    mock_mass.music.artists.library_items.assert_called_with(
        search=None,
        limit=50,
        favorite=None,
        order_by="timestamp_added_desc",
        provider="spotify_1",
    )


# =============================================================================
# PLAYLIST TOOLS
# =============================================================================


async def test_get_playlists(mock_mass: Mock) -> None:
    """Test get_playlists tool."""
    mcp = FastMCP("test")
    register_playlist_query_tools(mcp, mock_mass)

    pl_tool = _get_tool(mcp, "get_playlists")
    assert pl_tool is not None
    result = await pl_tool.fn()
    data = json.loads(result)
    assert "playlists" in data
    mock_mass.music.playlists.library_items.assert_called_once()


async def test_create_playlist(mock_mass: Mock) -> None:
    """Test create_playlist tool."""
    mcp = FastMCP("test")
    register_playlist_edit_tools(mcp, mock_mass)

    create_tool = _get_tool(mcp, "create_playlist")
    assert create_tool is not None
    result = await create_tool.fn(name="My New Playlist")
    data = json.loads(result)
    assert data["created"] is True
    mock_mass.music.playlists.create_playlist.assert_called_once_with("My New Playlist")


async def test_get_playlist_tracks(mock_mass: Mock) -> None:
    """Test get_playlist_tracks tool."""
    mcp = FastMCP("test")
    register_playlist_query_tools(mcp, mock_mass)

    pl_tool = _get_tool(mcp, "get_playlist_tracks")
    assert pl_tool is not None
    result = await pl_tool.fn(playlist_uri="library://playlist/101")
    data = json.loads(result)
    assert "tracks" in data
    assert "playlist" in data
    mock_mass.music.get_item_by_uri.assert_called()


async def test_add_to_playlist(mock_mass: Mock) -> None:
    """Test add_to_playlist tool."""
    mcp = FastMCP("test")
    register_playlist_edit_tools(mcp, mock_mass)

    add_tool = _get_tool(mcp, "add_to_playlist")
    assert add_tool is not None
    result = await add_tool.fn(
        playlist_uri="library://playlist/101", track_uri="library://track/123"
    )
    assert "added" in result.lower()
    mock_mass.music.playlists.add_playlist_track.assert_called_once()


async def test_remove_from_playlist(mock_mass: Mock) -> None:
    """Test remove_from_playlist tool."""
    mcp = FastMCP("test")
    register_playlist_delete_tools(mcp, mock_mass)

    remove_tool = _get_tool(mcp, "remove_from_playlist")
    assert remove_tool is not None
    result = await remove_tool.fn(playlist_uri="library://playlist/101", position=2)
    assert "removed" in result.lower()
    mock_mass.music.playlists.remove_playlist_tracks.assert_called_once()


async def test_delete_playlist(mock_mass: Mock) -> None:
    """Test delete_playlist tool."""
    mcp = FastMCP("test")
    register_playlist_delete_tools(mcp, mock_mass)

    delete_tool = _get_tool(mcp, "delete_playlist")
    assert delete_tool is not None
    result = await delete_tool.fn(playlist_uri="library://playlist/101")
    data = json.loads(result)
    assert data["deleted"] is True
    mock_mass.music.playlists.remove_item_from_library.assert_called_once()


async def test_clear_playlist(mock_mass: Mock, mock_track: Mock) -> None:
    """Test clear_playlist tool."""
    mcp = FastMCP("test")
    register_playlist_delete_tools(mcp, mock_mass)

    # Setup mock to yield tracks
    async def mock_playlist_tracks(*_args: Any, **_kwargs: Any) -> Any:
        yield mock_track
        yield mock_track

    mock_mass.music.playlists.tracks = mock_playlist_tracks

    clear_tool = _get_tool(mcp, "clear_playlist")
    assert clear_tool is not None
    result = await clear_tool.fn(playlist_uri="library://playlist/101")
    assert "cleared" in result.lower()
    mock_mass.music.playlists.remove_playlist_tracks.assert_called()


# =============================================================================
# PODCAST TOOLS
# =============================================================================


async def test_get_library_podcasts(mock_mass: Mock) -> None:
    """Test get_library_podcasts tool."""
    mcp = FastMCP("test")
    register_podcast_tools(mcp, mock_mass)

    podcast_tool = _get_tool(mcp, "get_library_podcasts")
    assert podcast_tool is not None
    result = await podcast_tool.fn()
    data = json.loads(result)
    assert "podcasts" in data
    assert len(data["podcasts"]) == 1
    assert data["podcasts"][0]["name"] == "Test Podcast"
    mock_mass.music.podcasts.library_items.assert_called_once()


async def test_get_podcast_episodes(mock_mass: Mock) -> None:
    """Test get_podcast_episodes tool."""
    mcp = FastMCP("test")
    register_podcast_tools(mcp, mock_mass)

    episodes_tool = _get_tool(mcp, "get_podcast_episodes")
    assert episodes_tool is not None
    result = await episodes_tool.fn(podcast_uri="library://podcast/201")
    data = json.loads(result)
    assert "episodes" in data
    assert "podcast" in data
    assert len(data["episodes"]) == 1
    assert data["episodes"][0]["name"] == "Episode 1: Introduction"


async def test_play_podcast_episode(mock_mass: Mock) -> None:
    """Test play_podcast_episode tool."""
    mcp = FastMCP("test")
    register_podcast_tools(mcp, mock_mass)

    play_tool = _get_tool(mcp, "play_podcast_episode")
    assert play_tool is not None
    result = await play_tool.fn(
        player_id="player_1",
        episode_uri="library://podcast_episode/301",
    )
    assert "playing" in result.lower()
    mock_mass.player_queues.play_media.assert_called_once()


# =============================================================================
# AUDIOBOOK TOOLS
# =============================================================================


async def test_get_library_audiobooks(mock_mass: Mock) -> None:
    """Test get_library_audiobooks tool."""
    mcp = FastMCP("test")
    register_audiobook_tools(mcp, mock_mass)

    audiobook_tool = _get_tool(mcp, "get_library_audiobooks")
    assert audiobook_tool is not None
    result = await audiobook_tool.fn()
    data = json.loads(result)
    assert "audiobooks" in data
    assert len(data["audiobooks"]) == 1
    assert data["audiobooks"][0]["name"] == "Test Audiobook"
    assert data["audiobooks"][0]["authors"] == ["Test Author"]
    mock_mass.music.audiobooks.library_items.assert_called_once()


async def test_get_audiobook_chapters(mock_mass: Mock, mock_audiobook: Mock) -> None:
    """Test get_audiobook_chapters tool."""
    # Mock get_item_by_uri to return the audiobook
    mock_mass.music.get_item_by_uri = AsyncMock(return_value=mock_audiobook)

    mcp = FastMCP("test")
    register_audiobook_tools(mcp, mock_mass)

    chapters_tool = _get_tool(mcp, "get_audiobook_chapters")
    assert chapters_tool is not None
    result = await chapters_tool.fn(audiobook_uri="library://audiobook/401")
    data = json.loads(result)
    assert "chapters" in data
    assert "audiobook" in data
    assert len(data["chapters"]) == 2
    assert data["chapters"][0]["name"] == "Chapter 1: Beginning"


async def test_play_audiobook(mock_mass: Mock) -> None:
    """Test play_audiobook tool."""
    mcp = FastMCP("test")
    register_audiobook_tools(mcp, mock_mass)

    play_tool = _get_tool(mcp, "play_audiobook")
    assert play_tool is not None
    result = await play_tool.fn(
        player_id="player_1",
        audiobook_uri="library://audiobook/401",
    )
    assert "playing" in result.lower()
    mock_mass.player_queues.play_media.assert_called()


async def test_play_audiobook_with_chapter(mock_mass: Mock) -> None:
    """Test play_audiobook tool with chapter selection."""
    mcp = FastMCP("test")
    register_audiobook_tools(mcp, mock_mass)

    play_tool = _get_tool(mcp, "play_audiobook")
    assert play_tool is not None
    result = await play_tool.fn(
        player_id="player_1",
        audiobook_uri="library://audiobook/401",
        chapter=2,
    )
    assert "chapter 2" in result.lower()
    mock_mass.player_queues.play_media.assert_called()


# =============================================================================
# RADIO TOOLS
# =============================================================================


async def test_get_library_radios(mock_mass: Mock) -> None:
    """Test get_library_radios tool."""
    mcp = FastMCP("test")
    register_radio_tools(mcp, mock_mass)

    tool = _get_tool(mcp, "get_library_radios")
    assert tool is not None
    result = await tool.fn()
    assert "radios" in result
    assert "Test Radio" in result


async def test_play_radio_station(mock_mass: Mock) -> None:
    """Test play_radio_station tool."""
    mcp = FastMCP("test")
    register_radio_tools(mcp, mock_mass)

    play_tool = _get_tool(mcp, "play_radio_station")
    assert play_tool is not None
    result = await play_tool.fn(
        player_id="player_1",
        radio_uri="library://radio/501",
    )
    assert "playing" in result.lower()
    mock_mass.player_queues.play_media.assert_called()


# =============================================================================
# METADATA TOOLS
# =============================================================================


async def test_get_track_lyrics(mock_mass: Mock) -> None:
    """Test get_track_lyrics tool."""
    mcp = FastMCP("test")
    register_metadata_tools(mcp, mock_mass)

    tool = _get_tool(mcp, "get_track_lyrics")
    assert tool is not None
    result = await tool.fn(track_uri="library://track/123")
    assert "lyrics" in result.lower()
    assert "Test lyrics" in result


async def test_get_item_artwork(mock_mass: Mock) -> None:
    """Test get_item_artwork tool."""
    mcp = FastMCP("test")
    register_metadata_tools(mcp, mock_mass)

    tool = _get_tool(mcp, "get_item_artwork")
    assert tool is not None
    result = await tool.fn(uri="library://track/123")
    assert "thumbnail" in result
    assert "example.com" in result
