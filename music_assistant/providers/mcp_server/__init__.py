"""
MCP Server Plugin for Music Assistant.

Exposes Music Assistant functionality via the Model Context Protocol (MCP),
enabling LLMs and AI assistants to control playback, query music library,
and interact with speakers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from music_assistant_models.config_entries import ConfigEntry, ConfigValueType
from music_assistant_models.enums import ConfigEntryType

from music_assistant.models.plugin import PluginProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

# Configuration keys
CONF_PORT = "port"
CONF_REQUIRE_AUTH = "require_auth"

# Query permissions (read-only)
CONF_LIBRARY_QUERY = "library_query"
CONF_PLAYER_QUERY = "player_query"
CONF_QUEUE_QUERY = "queue_query"
CONF_PLAYLIST_QUERY = "playlist_query"

# Control/Act permissions
CONF_PLAYBACK_CONTROL = "playback_control"
CONF_VOLUME_CONTROL = "volume_control"
CONF_PLAYER_CONTROL = "player_control"
CONF_QUEUE_CONTROL = "queue_control"

# Edit permissions
CONF_LIBRARY_EDIT = "library_edit"
CONF_PLAYLIST_EDIT = "playlist_edit"
CONF_QUEUE_EDIT = "queue_edit"

# Delete permissions
CONF_LIBRARY_DELETE = "library_delete"
CONF_PLAYLIST_DELETE = "playlist_delete"
CONF_QUEUE_DELETE = "queue_delete"

# Resources and prompts
CONF_PLAYER_RESOURCES = "player_resources"
CONF_LIBRARY_RESOURCES = "library_resources"
CONF_PROMPTS = "prompts"

# Default port for MCP server
DEFAULT_PORT = 8096

SUPPORTED_FEATURES: set[object] = set()


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return MCPServerProvider(mass, manifest, config)


async def get_config_entries(
    mass: MusicAssistant,
    instance_id: str | None = None,  # noqa: ARG001
    action: str | None = None,  # noqa: ARG001
    values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    """Return Config entries to setup this provider."""
    # Get the configured port (use default if not set yet)
    port = values.get(CONF_PORT, DEFAULT_PORT) if values else DEFAULT_PORT
    if not isinstance(port, int):
        port = DEFAULT_PORT

    # Build the MCP server URL using the webserver's publish IP
    try:
        host = mass.streams.publish_ip
    except Exception:
        host = "localhost"
    mcp_url = f"http://{host}:{port}/"

    # Build connection info text
    connection_info = (
        f"**MCP Server Endpoint:** `{mcp_url}`\n\n"
        "**Authentication:** When authentication is enabled, include a bearer token "
        "in your requests:\n"
        "```\n"
        "Authorization: Bearer <your-token>\n"
        "```\n\n"
        "**To create a token:**\n"
        f"1. Open the Music Assistant UI at `{mass.webserver.base_url}`\n"
        "2. Go to Settings → Security\n"
        "3. Create a new API token\n"
        "4. Copy the token and use it in your MCP client configuration"
    )

    return (
        ConfigEntry(
            key="connection_info",
            type=ConfigEntryType.LABEL,
            label=connection_info,
        ),
        ConfigEntry(
            key=CONF_PORT,
            type=ConfigEntryType.INTEGER,
            label="Port",
            description="Port number for the MCP server.",
            default_value=DEFAULT_PORT,
            range=(1024, 65535),
        ),
        ConfigEntry(
            key=CONF_REQUIRE_AUTH,
            type=ConfigEntryType.BOOLEAN,
            label="Require Authentication",
            description=(
                "Require Music Assistant authentication token for MCP requests. "
                "When enabled, clients must provide a valid MA token."
            ),
            default_value=True,
        ),
        # Query permissions (read-only)
        ConfigEntry(
            key=CONF_LIBRARY_QUERY,
            type=ConfigEntryType.BOOLEAN,
            label="Library Query",
            description=(
                "Search music, browse library, get recommendations, "
                "recently played, similar tracks, artist/album details."
            ),
            default_value=True,
            category="query",
        ),
        ConfigEntry(
            key=CONF_PLAYER_QUERY,
            type=ConfigEntryType.BOOLEAN,
            label="Player Query",
            description="Find players by name, get player capabilities.",
            default_value=True,
            category="query",
        ),
        ConfigEntry(
            key=CONF_QUEUE_QUERY,
            type=ConfigEntryType.BOOLEAN,
            label="Queue Query",
            description="Get current queue items and state.",
            default_value=True,
            category="query",
        ),
        ConfigEntry(
            key=CONF_PLAYLIST_QUERY,
            type=ConfigEntryType.BOOLEAN,
            label="Playlist Query",
            description="List playlists and get playlist tracks.",
            default_value=True,
            category="query",
        ),
        # Control/Act permissions
        ConfigEntry(
            key=CONF_PLAYBACK_CONTROL,
            type=ConfigEntryType.BOOLEAN,
            label="Playback Control",
            description="Play, pause, stop, seek, skip, next/previous track, play media.",
            default_value=True,
            category="control",
        ),
        ConfigEntry(
            key=CONF_VOLUME_CONTROL,
            type=ConfigEntryType.BOOLEAN,
            label="Volume Control",
            description="Set volume, volume up/down, mute, group volume.",
            default_value=True,
            category="control",
        ),
        ConfigEntry(
            key=CONF_PLAYER_CONTROL,
            type=ConfigEntryType.BOOLEAN,
            label="Player Control",
            description="Power on/off, group/ungroup players, play announcements.",
            default_value=True,
            category="control",
        ),
        ConfigEntry(
            key=CONF_QUEUE_CONTROL,
            type=ConfigEntryType.BOOLEAN,
            label="Queue Control",
            description="Shuffle, repeat mode, transfer queue, play specific index.",
            default_value=True,
            category="control",
        ),
        # Edit permissions
        ConfigEntry(
            key=CONF_LIBRARY_EDIT,
            type=ConfigEntryType.BOOLEAN,
            label="Library Edit",
            description="Add items to library, mark as favorite.",
            default_value=True,
            category="edit",
        ),
        ConfigEntry(
            key=CONF_PLAYLIST_EDIT,
            type=ConfigEntryType.BOOLEAN,
            label="Playlist Edit",
            description="Create playlists, add tracks to playlists.",
            default_value=True,
            category="edit",
        ),
        ConfigEntry(
            key=CONF_QUEUE_EDIT,
            type=ConfigEntryType.BOOLEAN,
            label="Queue Edit",
            description="Move items in the queue.",
            default_value=True,
            category="edit",
        ),
        # Delete permissions
        ConfigEntry(
            key=CONF_LIBRARY_DELETE,
            type=ConfigEntryType.BOOLEAN,
            label="Library Delete",
            description="Remove items from library, remove from favorites.",
            default_value=False,
            category="delete",
        ),
        ConfigEntry(
            key=CONF_PLAYLIST_DELETE,
            type=ConfigEntryType.BOOLEAN,
            label="Playlist Delete",
            description="Delete playlists, remove tracks, clear playlists.",
            default_value=False,
            category="delete",
        ),
        ConfigEntry(
            key=CONF_QUEUE_DELETE,
            type=ConfigEntryType.BOOLEAN,
            label="Queue Delete",
            description="Remove queue items, clear queue.",
            default_value=True,
            category="delete",
        ),
        # Resources
        ConfigEntry(
            key=CONF_PLAYER_RESOURCES,
            type=ConfigEntryType.BOOLEAN,
            label="Player Resources",
            description="Expose player resources (list, details, now playing, queue).",
            default_value=True,
            category="resources",
        ),
        ConfigEntry(
            key=CONF_LIBRARY_RESOURCES,
            type=ConfigEntryType.BOOLEAN,
            label="Library Resources",
            description="Expose library resources (stats, favorites, recently played).",
            default_value=True,
            category="resources",
        ),
        ConfigEntry(
            key=CONF_PROMPTS,
            type=ConfigEntryType.BOOLEAN,
            label="Prompts",
            description="Expose MCP prompts for AI assistant context.",
            default_value=True,
            category="resources",
        ),
    )


class MCPServerProvider(PluginProvider):
    """MCP Server provider for Music Assistant."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize the provider."""
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._mcp_server: object | None = None

    @property
    def port(self) -> int:
        """Return the configured port."""
        value = self.config.get_value(CONF_PORT)
        if isinstance(value, int):
            return value
        return DEFAULT_PORT

    @property
    def require_auth(self) -> bool:
        """Return whether authentication is required."""
        return bool(self.config.get_value(CONF_REQUIRE_AUTH))

    @property
    def enabled_features(self) -> dict[str, bool]:
        """Return a dictionary of enabled feature flags."""
        return {
            # Query
            "library_query": bool(self.config.get_value(CONF_LIBRARY_QUERY)),
            "player_query": bool(self.config.get_value(CONF_PLAYER_QUERY)),
            "queue_query": bool(self.config.get_value(CONF_QUEUE_QUERY)),
            "playlist_query": bool(self.config.get_value(CONF_PLAYLIST_QUERY)),
            # Control
            "playback_control": bool(self.config.get_value(CONF_PLAYBACK_CONTROL)),
            "volume_control": bool(self.config.get_value(CONF_VOLUME_CONTROL)),
            "player_control": bool(self.config.get_value(CONF_PLAYER_CONTROL)),
            "queue_control": bool(self.config.get_value(CONF_QUEUE_CONTROL)),
            # Edit
            "library_edit": bool(self.config.get_value(CONF_LIBRARY_EDIT)),
            "playlist_edit": bool(self.config.get_value(CONF_PLAYLIST_EDIT)),
            "queue_edit": bool(self.config.get_value(CONF_QUEUE_EDIT)),
            # Delete
            "library_delete": bool(self.config.get_value(CONF_LIBRARY_DELETE)),
            "playlist_delete": bool(self.config.get_value(CONF_PLAYLIST_DELETE)),
            "queue_delete": bool(self.config.get_value(CONF_QUEUE_DELETE)),
            # Resources
            "player_resources": bool(self.config.get_value(CONF_PLAYER_RESOURCES)),
            "library_resources": bool(self.config.get_value(CONF_LIBRARY_RESOURCES)),
            "prompts": bool(self.config.get_value(CONF_PROMPTS)),
        }

    async def loaded_in_mass(self) -> None:
        """Call after the provider has been loaded."""
        await super().loaded_in_mass()
        from .server import start_mcp_server  # noqa: PLC0415

        self._mcp_server = await start_mcp_server(
            mass=self.mass,
            port=self.port,
            require_auth=self.require_auth,
            enabled_features=self.enabled_features,
            logger=self.logger,
        )

        # Get the publish IP from the streams controller for consistent URL display
        publish_ip = self.mass.streams.publish_ip
        self.logger.info(
            "MCP Server started on http://%s:%d/",
            publish_ip,
            self.port,
        )

    async def unload(self, is_removed: bool = False) -> None:
        """Handle unload/close of the provider."""
        if self._mcp_server is not None:
            from .server import MCPServer  # noqa: PLC0415

            if isinstance(self._mcp_server, MCPServer):
                await self._mcp_server.stop()
            self._mcp_server = None
