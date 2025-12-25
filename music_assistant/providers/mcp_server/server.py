"""MCP Server implementation for Music Assistant."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from .prompts import register_prompts
from .resources import register_library_resources, register_player_resources
from .tools import (
    register_audiobook_tools,
    register_library_tools,
    register_metadata_tools,
    register_playback_tools,
    register_player_tools,
    register_playlist_tools,
    register_podcast_tools,
    register_queue_tools,
    register_radio_tools,
    register_volume_tools,
)

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant

LOGGER = logging.getLogger(__name__)


def create_mcp_server(
    mass: MusicAssistant,
    require_auth: bool = True,
    enabled_features: dict[str, bool] | None = None,
) -> FastMCP:
    """Create and configure the MCP server instance.

    :param mass: MusicAssistant instance.
    :param require_auth: Whether to require authentication.
    :param enabled_features: Dictionary of feature flags to enable/disable tool categories.
    :return: Configured FastMCP server instance.
    """
    from mcp.server.transport_security import TransportSecuritySettings  # noqa: PLC0415

    server_kwargs: dict[str, Any] = {
        "name": "Music Assistant",
        "instructions": (
            "Music Assistant MCP server for controlling music playback "
            "and managing your music library."
        ),
        "stateless_http": True,
        "json_response": True,
        "transport_security": TransportSecuritySettings(
            enable_dns_rebinding_protection=False,
        ),
    }

    if require_auth:
        from mcp.server.auth.settings import AuthSettings  # noqa: PLC0415
        from pydantic import AnyHttpUrl  # noqa: PLC0415

        from .auth import MusicAssistantTokenVerifier  # noqa: PLC0415

        base_url = mass.webserver.base_url
        server_kwargs["auth"] = AuthSettings(
            issuer_url=AnyHttpUrl(base_url),
            resource_server_url=None,
        )
        server_kwargs["token_verifier"] = MusicAssistantTokenVerifier(mass)

    mcp = FastMCP(**server_kwargs)
    features = enabled_features or {}

    # Register tools
    if features.get("playback_tools", True):
        register_playback_tools(mcp, mass)
    if features.get("queue_tools", True):
        register_queue_tools(mcp, mass)
    if features.get("volume_tools", True):
        register_volume_tools(mcp, mass)
    if features.get("library_tools", True):
        register_library_tools(mcp, mass)
        register_podcast_tools(mcp, mass)
        register_radio_tools(mcp, mass)
        register_audiobook_tools(mcp, mass)
        register_metadata_tools(mcp, mass)
    if features.get("playlist_tools", True):
        register_playlist_tools(mcp, mass)
    if features.get("player_tools", True):
        register_player_tools(mcp, mass)

    # Register resources
    if features.get("player_resources", True):
        register_player_resources(mcp, mass)
    if features.get("library_resources", True):
        register_library_resources(mcp, mass)

    # Register prompts
    if features.get("prompts", True):
        register_prompts(mcp, mass)

    return mcp


class MCPServer:
    """Wrapper for the MCP uvicorn server."""

    def __init__(
        self,
        mass: MusicAssistant,
        port: int,
        require_auth: bool,
        enabled_features: dict[str, bool] | None,
        logger: logging.Logger,
    ) -> None:
        """Initialize the MCP server wrapper."""
        import uvicorn  # noqa: PLC0415

        mcp = create_mcp_server(mass, require_auth, enabled_features)
        mcp.settings.streamable_http_path = "/"

        # Map MA log level to uvicorn log level
        ma_log_level = logger.getEffectiveLevel()
        uvicorn_log_level = logging.getLevelName(ma_log_level).lower()

        config = uvicorn.Config(
            app=mcp.streamable_http_app(),
            host="0.0.0.0",
            port=port,
            log_level=uvicorn_log_level,
            access_log=ma_log_level <= logging.DEBUG,
            log_config=None,  # Disable uvicorn's default logging config
        )
        self._server = uvicorn.Server(config)
        self._serve_task: asyncio.Task[None] | None = None
        self._logger = logger

    async def start(self) -> None:
        """Start the server."""
        # Configure uvicorn loggers to use MA logger
        ma_log_level = self._logger.getEffectiveLevel()
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            uvi_logger = logging.getLogger(name)
            uvi_logger.handlers = self._logger.handlers
            uvi_logger.setLevel(ma_log_level)

        # Configure MCP/FastMCP loggers to use MA logger
        for name in ("mcp", "mcp.server", "mcp.server.fastmcp", "mcp.server.lowlevel"):
            mcp_logger = logging.getLogger(name)
            mcp_logger.handlers = self._logger.handlers
            mcp_logger.setLevel(ma_log_level)

        self._serve_task = asyncio.create_task(self._server.serve())

    async def stop(self) -> None:
        """Stop the server."""
        import contextlib  # noqa: PLC0415

        if self._serve_task is None:
            return

        # Signal graceful shutdown
        self._server.should_exit = True

        # Wait briefly for graceful shutdown
        try:
            await asyncio.wait_for(asyncio.shield(self._serve_task), timeout=2.0)
        except TimeoutError:
            # Force exit
            self._server.force_exit = True
            try:
                await asyncio.wait_for(asyncio.shield(self._serve_task), timeout=1.0)
            except TimeoutError:
                # Cancel the task
                self._serve_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._serve_task
        except asyncio.CancelledError:
            self._server.force_exit = True
            self._serve_task.cancel()

        self._serve_task = None


async def start_mcp_server(
    mass: MusicAssistant,
    port: int,
    require_auth: bool = True,
    enabled_features: dict[str, bool] | None = None,
    logger: logging.Logger | None = None,
) -> MCPServer:
    """Start the MCP server.

    :param mass: MusicAssistant instance.
    :param port: Port to run the server on.
    :param require_auth: Whether to require authentication.
    :param enabled_features: Dictionary of feature flags to enable/disable tool categories.
    :param logger: Logger to use for MCP and uvicorn logging.
    :return: MCPServer instance that can be stopped.
    """
    if logger is None:
        logger = LOGGER
    server = MCPServer(mass, port, require_auth, enabled_features, logger)
    await server.start()
    return server
