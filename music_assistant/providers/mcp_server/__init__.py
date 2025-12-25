"""
MCP Server Plugin for Music Assistant.

Exposes Music Assistant functionality via the Model Context Protocol (MCP),
enabling LLMs and AI assistants to control playback, query music library,
and interact with speakers.
"""

from __future__ import annotations

import asyncio
import contextlib
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
    mass: MusicAssistant,  # noqa: ARG001
    instance_id: str | None = None,  # noqa: ARG001
    action: str | None = None,  # noqa: ARG001
    values: dict[str, ConfigValueType] | None = None,  # noqa: ARG001
) -> tuple[ConfigEntry, ...]:
    """Return Config entries to setup this provider."""
    return (
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

    _server_task: asyncio.Task[None] | None = None

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
        from .server import run_mcp_server  # noqa: PLC0415

        self._server_task = await run_mcp_server(
            mass=self.mass,
            host="0.0.0.0",
            port=self.port,
            require_auth=self.require_auth,
            enabled_features=self.enabled_features,
        )

        self.logger.info("MCP server available at http://0.0.0.0:%d/mcp", self.port)

    async def unload(self, is_removed: bool = False) -> None:
        """Handle unload/close of the provider."""
        if self._server_task and not self._server_task.done():
            self._server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._server_task
            self._server_task = None
        self.logger.info("MCP server stopped")
