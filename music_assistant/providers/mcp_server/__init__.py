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
CONF_INTRO_PROMPT = "intro_prompt"
CONF_PLAYER_CONTEXT_PROMPT = "player_context_prompt"

# Default values
DEFAULT_MCP_PORT = 8096

# Default prompt templates
# {player_list} is replaced with the list of available players
DEFAULT_INTRO_PROMPT = """You are connected to Music Assistant, \
a music library manager and multi-room audio system.

Available capabilities:
- Control playback: play, pause, stop, next/previous track
- Adjust volume on any player
- Search and play music from the library
- Control multiple speakers/rooms

Available players: {player_list}

To control music, use the available tools. \
Always check which players are available before issuing commands.
When the user asks to play music, search for it first, \
then use play_media with the URI from the search results."""

# {player_name}, {state}, {volume}, {current_track} are replaced with actual values
DEFAULT_PLAYER_CONTEXT_PROMPT = """You are controlling: {player_name}
Current state: {state}
Volume: {volume}%
Now playing: {current_track}

Available actions: play, pause, stop, next_track, previous_track, set_volume"""

SUPPORTED_FEATURES: set[object] = set()


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return MCPServerProvider(mass, manifest, config)


async def get_config_entries(
    _mass: MusicAssistant,
    _instance_id: str | None = None,
    _action: str | None = None,
    _values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    """Return Config entries to setup this provider."""
    return (
        ConfigEntry(
            key=CONF_PORT,
            type=ConfigEntryType.INTEGER,
            label="MCP Server Port",
            description=(
                "The TCP port for the MCP server. Clients connect via http://host:port/mcp"
            ),
            default_value=DEFAULT_MCP_PORT,
            required=True,
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
        ConfigEntry(
            key=CONF_INTRO_PROMPT,
            type=ConfigEntryType.STRING,
            label="Introduction Prompt",
            description=(
                "Prompt to introduce AI assistants to Music Assistant capabilities. "
                "Use {player_list} as a placeholder for available players."
            ),
            default_value=DEFAULT_INTRO_PROMPT,
            category="advanced",
            multi_value=True,
        ),
        ConfigEntry(
            key=CONF_PLAYER_CONTEXT_PROMPT,
            type=ConfigEntryType.STRING,
            label="Player Context Prompt",
            description=(
                "Prompt for player-specific context. Placeholders: "
                "{player_name}, {state}, {volume}, {current_track}"
            ),
            default_value=DEFAULT_PLAYER_CONTEXT_PROMPT,
            category="advanced",
            multi_value=True,
        ),
    )


class MCPServerProvider(PluginProvider):
    """MCP Server provider for Music Assistant."""

    _server_task: asyncio.Task[None] | None = None
    _shutdown_event: asyncio.Event | None = None

    @property
    def port(self) -> int:
        """Return the configured MCP server port."""
        port_value = self.config.get_value(CONF_PORT)
        if isinstance(port_value, int):
            return port_value
        return DEFAULT_MCP_PORT

    @property
    def require_auth(self) -> bool:
        """Return whether authentication is required."""
        return bool(self.config.get_value(CONF_REQUIRE_AUTH))

    @property
    def intro_prompt(self) -> str:
        """Return the introduction prompt template."""
        value = self.config.get_value(CONF_INTRO_PROMPT)
        if isinstance(value, str):
            return value
        return DEFAULT_INTRO_PROMPT

    @property
    def player_context_prompt(self) -> str:
        """Return the player context prompt template."""
        value = self.config.get_value(CONF_PLAYER_CONTEXT_PROMPT)
        if isinstance(value, str):
            return value
        return DEFAULT_PLAYER_CONTEXT_PROMPT

    async def loaded_in_mass(self) -> None:
        """Call after the provider has been loaded."""
        from .server import start_mcp_server  # noqa: PLC0415

        self.logger.info("Starting MCP server on port %s", self.port)
        self._server_task, self._shutdown_event = await start_mcp_server(
            mass=self.mass,
            port=self.port,
            require_auth=self.require_auth,
            intro_prompt=self.intro_prompt,
            player_context_prompt=self.player_context_prompt,
            logger=self.logger,
        )
        self.logger.info(
            "MCP server started. Connect via http://%s:%s/mcp",
            self.mass.webserver.base_url.split("://")[1].split(":")[0],
            self.port,
        )

    async def unload(self, is_removed: bool = False) -> None:
        """Handle unload/close of the provider."""
        if self._shutdown_event is not None:
            self._shutdown_event.set()
            if self._server_task is not None:
                # Give the server time to shutdown gracefully
                with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(
                        asyncio.shield(self._server_task),
                        timeout=5.0,
                    )
            self.logger.info("MCP server stopped")
