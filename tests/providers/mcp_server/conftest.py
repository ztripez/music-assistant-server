"""Fixtures for MCP Server tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock

import pytest

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


@pytest.fixture
def mock_player() -> Mock:
    """Create a mock player."""
    from music_assistant_models.enums import PlayerFeature  # noqa: PLC0415

    player = Mock()
    player.player_id = "player_1"
    player.display_name = "Living Room"
    player.available = True
    player.powered = True
    player.volume_level = 50
    player.volume_muted = False
    player.playback_state = Mock(value="playing")
    player.type = Mock(value="speaker")
    player.group_members = []
    player.supported_features = {
        PlayerFeature.POWER,
        PlayerFeature.VOLUME_SET,
        PlayerFeature.PAUSE,
        PlayerFeature.SEEK,
        PlayerFeature.SET_MEMBERS,
    }
    return player


@pytest.fixture
def mock_player_2() -> Mock:
    """Create a second mock player."""
    from music_assistant_models.enums import PlayerFeature  # noqa: PLC0415

    player = Mock()
    player.player_id = "player_2"
    player.display_name = "Kitchen"
    player.available = True
    player.powered = True
    player.volume_level = 30
    player.volume_muted = False
    player.playback_state = Mock(value="idle")
    player.type = Mock(value="speaker")
    player.group_members = []
    player.supported_features = {
        PlayerFeature.POWER,
        PlayerFeature.VOLUME_SET,
        PlayerFeature.PAUSE,
    }
    return player


@pytest.fixture
def mock_queue_item() -> Mock:
    """Create a mock queue item."""
    item = Mock()
    item.name = "Test Track"
    item.uri = "library://track/123"
    item.duration = 180
    item.queue_item_id = "qi_1"
    item.artist_str = "Test Artist"
    return item


@pytest.fixture
def mock_queue(mock_queue_item: Mock) -> Mock:
    """Create a mock queue."""
    queue = Mock()
    queue.queue_id = "player_1"
    queue.state = Mock(value="playing")
    queue.shuffle_enabled = False
    queue.repeat_mode = Mock(value="off")
    queue.current_index = 0
    queue.current_item = mock_queue_item
    queue.elapsed_time = 45
    return queue


@pytest.fixture
def mock_track() -> Mock:
    """Create a mock track."""
    from music_assistant_models.enums import MediaType  # noqa: PLC0415

    track = Mock(spec=["name", "uri", "item_id", "provider", "media_type", "track_number"])
    track.name = "Test Track"
    track.uri = "library://track/123"
    track.item_id = "123"
    track.provider = "library"
    track.media_type = MediaType.TRACK
    track.track_number = 1
    return track


@pytest.fixture
def mock_artist() -> Mock:
    """Create a mock artist."""
    artist = Mock(spec=["name", "uri", "item_id", "provider"])
    artist.name = "Test Artist"
    artist.uri = "library://artist/456"
    artist.item_id = "456"
    artist.provider = "library"
    return artist


@pytest.fixture
def mock_album() -> Mock:
    """Create a mock album."""
    album = Mock(spec=["name", "uri", "item_id", "provider"])
    album.name = "Test Album"
    album.uri = "library://album/789"
    album.item_id = "789"
    album.provider = "library"
    return album


@pytest.fixture
def mock_playlist() -> Mock:
    """Create a mock playlist."""
    playlist = Mock(spec=["name", "uri", "item_id", "provider"])
    playlist.name = "Test Playlist"
    playlist.uri = "library://playlist/101"
    playlist.item_id = "101"
    playlist.provider = "library"
    return playlist


@pytest.fixture
def mock_radio() -> Mock:
    """Create a mock radio station."""
    radio = Mock(spec=["name", "uri", "item_id", "provider", "favorite"])
    radio.name = "Test Radio"
    radio.uri = "library://radio/501"
    radio.item_id = "501"
    radio.provider = "library"
    radio.favorite = True
    return radio


@pytest.fixture
def mock_podcast() -> Mock:
    """Create a mock podcast."""
    podcast = Mock(spec=["name", "uri", "item_id", "provider", "publisher", "total_episodes"])
    podcast.name = "Test Podcast"
    podcast.uri = "library://podcast/201"
    podcast.item_id = "201"
    podcast.provider = "library"
    podcast.publisher = "Test Publisher"
    podcast.total_episodes = 50
    return podcast


@pytest.fixture
def mock_podcast_episode() -> Mock:
    """Create a mock podcast episode."""
    episode = Mock(
        spec=[
            "name",
            "uri",
            "item_id",
            "provider",
            "duration",
            "position",
            "resume_position_ms",
            "fully_played",
        ]
    )
    episode.name = "Episode 1: Introduction"
    episode.uri = "library://podcast_episode/301"
    episode.item_id = "301"
    episode.provider = "library"
    episode.duration = 3600
    episode.position = 1
    episode.resume_position_ms = 120000
    episode.fully_played = False
    return episode


@pytest.fixture
def mock_audiobook() -> Mock:
    """Create a mock audiobook."""
    audiobook = Mock(
        spec=[
            "name",
            "uri",
            "item_id",
            "provider",
            "authors",
            "narrators",
            "duration",
            "resume_position_ms",
            "fully_played",
            "metadata",
        ]
    )
    audiobook.name = "Test Audiobook"
    audiobook.uri = "library://audiobook/401"
    audiobook.item_id = "401"
    audiobook.provider = "library"
    audiobook.authors = ["Test Author"]
    audiobook.narrators = ["Test Narrator"]
    audiobook.duration = 36000
    audiobook.resume_position_ms = 300000
    audiobook.fully_played = False

    # Mock chapters
    chapter1 = Mock()
    chapter1.position = 1
    chapter1.name = "Chapter 1: Beginning"
    chapter1.start = 0
    chapter2 = Mock()
    chapter2.position = 2
    chapter2.name = "Chapter 2: Middle"
    chapter2.start = 12000
    audiobook.metadata = Mock()
    audiobook.metadata.chapters = [chapter1, chapter2]
    return audiobook


@pytest.fixture
def mock_search_results(mock_track: Mock, mock_artist: Mock, mock_album: Mock) -> Mock:
    """Create mock search results."""
    results = Mock()
    results.tracks = [mock_track]
    results.artists = [mock_artist]
    results.albums = [mock_album]
    results.playlists = []
    results.radio = []
    return results


@pytest.fixture
def mock_provider() -> Mock:
    """Create a mock music provider."""
    provider = Mock()
    provider.instance_id = "spotify_1"
    provider.name = "Spotify"
    provider.domain = "spotify"
    provider.available = True
    return provider


@pytest.fixture
def mock_mass(  # noqa: PLR0913, PLR0915
    mock_player: Mock,
    mock_player_2: Mock,
    mock_queue: Mock,
    mock_queue_item: Mock,
    mock_track: Mock,
    mock_artist: Mock,
    mock_album: Mock,
    mock_playlist: Mock,
    mock_radio: Mock,
    mock_podcast: Mock,
    mock_podcast_episode: Mock,
    mock_audiobook: Mock,
    mock_search_results: Mock,
    mock_provider: Mock,
) -> Mock:
    """Create a mock MusicAssistant instance."""
    mass = Mock()

    # Players controller
    mass.players.all.return_value = [mock_player, mock_player_2]
    mass.players.get.return_value = mock_player
    mass.players.cmd_power = AsyncMock()
    mass.players.cmd_volume_set = AsyncMock()
    mass.players.cmd_volume_up = AsyncMock()
    mass.players.cmd_volume_down = AsyncMock()
    mass.players.cmd_volume_mute = AsyncMock()
    mass.players.cmd_group_volume = AsyncMock()
    mass.players.cmd_group_many = AsyncMock()
    mass.players.cmd_ungroup = AsyncMock()
    mass.players.play_announcement = AsyncMock()

    # Player queues controller
    mass.player_queues.get.return_value = mock_queue
    mass.player_queues.items.return_value = [mock_queue_item]
    mass.player_queues.play = AsyncMock()
    mass.player_queues.pause = AsyncMock()
    mass.player_queues.stop = AsyncMock()
    mass.player_queues.next = AsyncMock()
    mass.player_queues.previous = AsyncMock()
    mass.player_queues.seek = AsyncMock()
    mass.player_queues.skip = AsyncMock()
    mass.player_queues.play_media = AsyncMock()
    mass.player_queues.play_index = AsyncMock()
    mass.player_queues.clear = Mock()
    mass.player_queues.set_shuffle = AsyncMock()
    mass.player_queues.set_repeat = Mock()
    mass.player_queues.move_item = Mock()
    mass.player_queues.delete_item = Mock()
    mass.player_queues.transfer_queue = AsyncMock()

    # Music controller
    mass.music.search = AsyncMock(return_value=mock_search_results)
    mass.music.get_item_by_uri = AsyncMock(return_value=mock_track)
    mass.music.browse = AsyncMock(return_value=[mock_track, mock_album])
    mass.music.recommendations = AsyncMock(return_value=[])
    mass.music.recently_played = AsyncMock(return_value=[mock_track])
    mass.music.recently_added_tracks = AsyncMock(return_value=[mock_track])
    mass.music.in_progress_items = AsyncMock(return_value=[])
    mass.music.add_item_to_library = AsyncMock()
    mass.music.remove_item_from_library = AsyncMock()
    mass.music.add_item_to_favorites = AsyncMock()
    mass.music.remove_item_from_favorites = AsyncMock()
    mass.music.providers = [mock_provider]

    # Music sub-controllers
    mass.music.artists.library_items = AsyncMock(return_value=[mock_artist])
    mass.music.artists.library_count = AsyncMock(return_value=100)
    mass.music.artists.tracks = AsyncMock(return_value=[mock_track])
    mass.music.artists.albums = AsyncMock(return_value=[mock_album])

    mass.music.albums.library_items = AsyncMock(return_value=[mock_album])
    mass.music.albums.library_count = AsyncMock(return_value=50)
    mass.music.albums.tracks = AsyncMock(return_value=[mock_track])

    mass.music.tracks.library_items = AsyncMock(return_value=[mock_track])
    mass.music.tracks.library_count = AsyncMock(return_value=500)
    mass.music.tracks.similar_tracks = AsyncMock(return_value=[mock_track])

    mass.music.playlists.library_items = AsyncMock(return_value=[mock_playlist])
    mass.music.playlists.library_count = AsyncMock(return_value=10)
    mass.music.playlists.create_playlist = AsyncMock(return_value=mock_playlist)
    mass.music.playlists.add_playlist_track = AsyncMock()
    mass.music.playlists.remove_playlist_tracks = AsyncMock()
    mass.music.playlists.remove_item_from_library = AsyncMock()

    async def mock_playlist_tracks(*_args: Any, **_kwargs: Any) -> Any:
        yield mock_track

    mass.music.playlists.tracks = mock_playlist_tracks

    # Podcasts controller
    mass.music.podcasts.library_items = AsyncMock(return_value=[mock_podcast])

    async def mock_podcast_episodes(*_args: Any, **_kwargs: Any) -> Any:
        yield mock_podcast_episode

    mass.music.podcasts.episodes = mock_podcast_episodes

    # Audiobooks controller
    mass.music.audiobooks.library_items = AsyncMock(return_value=[mock_audiobook])
    mass.music.audiobooks.library_count = AsyncMock(return_value=5)

    # Radio controller
    mass.music.radio.library_items = AsyncMock(return_value=[mock_radio])
    mass.music.radio.library_count = AsyncMock(return_value=10)

    # Add library_count for podcasts
    mass.music.podcasts.library_count = AsyncMock(return_value=3)

    # Metadata controller
    mass.metadata.get_track_lyrics = AsyncMock(
        return_value=("Test lyrics line 1\nTest lyrics line 2", None)
    )
    mass.metadata.get_image_url_for_item = AsyncMock(return_value="https://example.com/image.jpg")

    return mass


@pytest.fixture
def mcp_server(mock_mass: Mock) -> FastMCP:
    """Create MCP server with mocked MusicAssistant."""
    from music_assistant.providers.mcp_server.server import create_mcp_server  # noqa: PLC0415

    return create_mcp_server(mock_mass, require_auth=False)
