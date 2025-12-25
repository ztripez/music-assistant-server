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

# Feature enable/disable keys
CONF_ENABLE_PLAYBACK_TOOLS = "enable_playback_tools"
CONF_ENABLE_QUEUE_TOOLS = "enable_queue_tools"
CONF_ENABLE_VOLUME_TOOLS = "enable_volume_tools"
CONF_ENABLE_LIBRARY_TOOLS = "enable_library_tools"
CONF_ENABLE_PLAYLIST_TOOLS = "enable_playlist_tools"
CONF_ENABLE_PLAYER_TOOLS = "enable_player_tools"
CONF_ENABLE_PLAYER_RESOURCES = "enable_player_resources"
CONF_ENABLE_LIBRARY_RESOURCES = "enable_library_resources"
CONF_ENABLE_PROMPTS = "enable_prompts"

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
        # Feature toggles
        ConfigEntry(
            key=CONF_ENABLE_PLAYBACK_TOOLS,
            type=ConfigEntryType.BOOLEAN,
            label="Enable Playback Tools",
            description="Expose play, pause, stop, seek, skip, and media playback tools.",
            default_value=True,
            category="features",
        ),
        ConfigEntry(
            key=CONF_ENABLE_QUEUE_TOOLS,
            type=ConfigEntryType.BOOLEAN,
            label="Enable Queue Tools",
            description="Expose queue management tools (get, clear, shuffle, repeat, move items).",
            default_value=True,
            category="features",
        ),
        ConfigEntry(
            key=CONF_ENABLE_VOLUME_TOOLS,
            type=ConfigEntryType.BOOLEAN,
            label="Enable Volume Tools",
            description="Expose volume control tools (set, up, down, mute, group volume).",
            default_value=True,
            category="features",
        ),
        ConfigEntry(
            key=CONF_ENABLE_LIBRARY_TOOLS,
            type=ConfigEntryType.BOOLEAN,
            label="Enable Library Tools",
            description=(
                "Expose library tools (recommendations, recently played, browse, "
                "artist/album tracks, favorites)."
            ),
            default_value=True,
            category="features",
        ),
        ConfigEntry(
            key=CONF_ENABLE_PLAYLIST_TOOLS,
            type=ConfigEntryType.BOOLEAN,
            label="Enable Playlist Tools",
            description="Expose playlist tools (get, create, add/remove tracks).",
            default_value=True,
            category="features",
        ),
        ConfigEntry(
            key=CONF_ENABLE_PLAYER_TOOLS,
            type=ConfigEntryType.BOOLEAN,
            label="Enable Player Tools",
            description="Expose player tools (power, grouping, announcements, find by name).",
            default_value=True,
            category="features",
        ),
        ConfigEntry(
            key=CONF_ENABLE_PLAYER_RESOURCES,
            type=ConfigEntryType.BOOLEAN,
            label="Enable Player Resources",
            description=(
                "Expose player resources (players list, player details, now playing, queue)."
            ),
            default_value=True,
            category="features",
        ),
        ConfigEntry(
            key=CONF_ENABLE_LIBRARY_RESOURCES,
            type=ConfigEntryType.BOOLEAN,
            label="Enable Library Resources",
            description=(
                "Expose library resources (stats, favorites, recently played, providers)."
            ),
            default_value=True,
            category="features",
        ),
        ConfigEntry(
            key=CONF_ENABLE_PROMPTS,
            type=ConfigEntryType.BOOLEAN,
            label="Enable Prompts",
            description="Expose MCP prompts for AI assistant context.",
            default_value=True,
            category="features",
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
            "playback_tools": bool(self.config.get_value(CONF_ENABLE_PLAYBACK_TOOLS)),
            "queue_tools": bool(self.config.get_value(CONF_ENABLE_QUEUE_TOOLS)),
            "volume_tools": bool(self.config.get_value(CONF_ENABLE_VOLUME_TOOLS)),
            "library_tools": bool(self.config.get_value(CONF_ENABLE_LIBRARY_TOOLS)),
            "playlist_tools": bool(self.config.get_value(CONF_ENABLE_PLAYLIST_TOOLS)),
            "player_tools": bool(self.config.get_value(CONF_ENABLE_PLAYER_TOOLS)),
            "player_resources": bool(self.config.get_value(CONF_ENABLE_PLAYER_RESOURCES)),
            "library_resources": bool(self.config.get_value(CONF_ENABLE_LIBRARY_RESOURCES)),
            "prompts": bool(self.config.get_value(CONF_ENABLE_PROMPTS)),
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
