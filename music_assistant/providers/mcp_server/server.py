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
    register_library_delete_tools,
    register_library_edit_tools,
    register_library_query_tools,
    register_metadata_tools,
    register_playback_control_tools,
    register_playback_query_tools,
    register_player_control_tools,
    register_player_query_tools,
    register_playlist_delete_tools,
    register_playlist_edit_tools,
    register_playlist_query_tools,
    register_podcast_tools,
    register_queue_control_tools,
    register_queue_delete_tools,
    register_queue_edit_tools,
    register_queue_query_tools,
    register_radio_tools,
    register_volume_control_tools,
)

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant

LOGGER = logging.getLogger(__name__)


def _register_query_tools(
    mcp: FastMCP,
    mass: MusicAssistant,
    features: dict[str, bool],
    log: logging.Logger,
) -> None:
    """Register query tools based on feature flags."""
    if features.get("library_query", True):
        register_library_query_tools(mcp, mass)
        register_playback_query_tools(mcp, mass)
        register_podcast_tools(mcp, mass)
        register_radio_tools(mcp, mass)
        register_audiobook_tools(mcp, mass)
        register_metadata_tools(mcp, mass)
        log.debug("Registered library query tools")
    if features.get("player_query", True):
        register_player_query_tools(mcp, mass)
        log.debug("Registered player query tools")
    if features.get("queue_query", True):
        register_queue_query_tools(mcp, mass)
        log.debug("Registered queue query tools")
    if features.get("playlist_query", True):
        register_playlist_query_tools(mcp, mass)
        log.debug("Registered playlist query tools")


def _register_control_tools(
    mcp: FastMCP,
    mass: MusicAssistant,
    features: dict[str, bool],
    log: logging.Logger,
) -> None:
    """Register control tools based on feature flags."""
    if features.get("playback_control", True):
        register_playback_control_tools(mcp, mass)
        log.debug("Registered playback control tools")
    if features.get("volume_control", True):
        register_volume_control_tools(mcp, mass)
        log.debug("Registered volume control tools")
    if features.get("player_control", True):
        register_player_control_tools(mcp, mass)
        log.debug("Registered player control tools")
    if features.get("queue_control", True):
        register_queue_control_tools(mcp, mass)
        log.debug("Registered queue control tools")


def _register_edit_tools(
    mcp: FastMCP,
    mass: MusicAssistant,
    features: dict[str, bool],
    log: logging.Logger,
) -> None:
    """Register edit tools based on feature flags."""
    if features.get("library_edit", True):
        register_library_edit_tools(mcp, mass)
        log.debug("Registered library edit tools")
    if features.get("playlist_edit", True):
        register_playlist_edit_tools(mcp, mass)
        log.debug("Registered playlist edit tools")
    if features.get("queue_edit", True):
        register_queue_edit_tools(mcp, mass)
        log.debug("Registered queue edit tools")


def _register_delete_tools(
    mcp: FastMCP,
    mass: MusicAssistant,
    features: dict[str, bool],
    log: logging.Logger,
) -> None:
    """Register delete tools based on feature flags."""
    if features.get("library_delete", False):
        register_library_delete_tools(mcp, mass)
        log.debug("Registered library delete tools")
    if features.get("playlist_delete", False):
        register_playlist_delete_tools(mcp, mass)
        log.debug("Registered playlist delete tools")
    if features.get("queue_delete", True):
        register_queue_delete_tools(mcp, mass)
        log.debug("Registered queue delete tools")


def _register_resources_and_prompts(
    mcp: FastMCP,
    mass: MusicAssistant,
    features: dict[str, bool],
    log: logging.Logger,
) -> None:
    """Register resources and prompts based on feature flags."""
    if features.get("player_resources", True):
        register_player_resources(mcp, mass)
        log.debug("Registered player resources")
    if features.get("library_resources", True):
        register_library_resources(mcp, mass)
        log.debug("Registered library resources")
    if features.get("prompts", True):
        register_prompts(mcp, mass)
        log.debug("Registered prompts")


def create_mcp_server(
    mass: MusicAssistant,
    require_auth: bool = True,
    enabled_features: dict[str, bool] | None = None,
    logger: logging.Logger | None = None,
) -> FastMCP:
    """Create and configure the MCP server instance.

    :param mass: MusicAssistant instance.
    :param require_auth: Whether to require authentication.
    :param enabled_features: Dictionary of feature flags to enable/disable tool categories.
    :param logger: Logger instance for debug output.
    :return: Configured FastMCP server instance.
    """
    from mcp.server.transport_security import TransportSecuritySettings  # noqa: PLC0415

    log = logger or LOGGER

    log.debug("Creating MCP server instance (require_auth=%s)", require_auth)

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
        log.debug("Auth configured with issuer_url=%s", base_url)

    mcp = FastMCP(**server_kwargs)
    features = enabled_features or {}

    # Register all tools, resources, and prompts
    _register_query_tools(mcp, mass, features, log)
    _register_control_tools(mcp, mass, features, log)
    _register_edit_tools(mcp, mass, features, log)
    _register_delete_tools(mcp, mass, features, log)
    _register_resources_and_prompts(mcp, mass, features, log)

    log.debug("MCP server instance created")
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

        self._logger = logger

        self._logger.debug("Initializing MCP server on port %d", port)

        mcp = create_mcp_server(mass, require_auth, enabled_features, logger)
        mcp.settings.streamable_http_path = "/"

        # Map MA log level to uvicorn log level
        ma_log_level = logger.getEffectiveLevel()
        uvicorn_log_level = logging.getLevelName(ma_log_level).lower()

        self._logger.debug(
            "Configuring uvicorn (log_level=%s, access_log=%s)",
            uvicorn_log_level,
            ma_log_level <= logging.DEBUG,
        )

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

    async def start(self) -> None:
        """Start the server."""
        self._logger.debug("Starting MCP server...")

        # Configure uvicorn loggers to use MA logger
        ma_log_level = self._logger.getEffectiveLevel()
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            uvi_logger = logging.getLogger(name)
            uvi_logger.handlers = self._logger.handlers
            uvi_logger.setLevel(ma_log_level)
        self._logger.debug("Configured uvicorn loggers")

        # Configure MCP/FastMCP loggers to use MA logger
        for name in ("mcp", "mcp.server", "mcp.server.fastmcp", "mcp.server.lowlevel"):
            mcp_logger = logging.getLogger(name)
            mcp_logger.handlers = self._logger.handlers
            mcp_logger.setLevel(ma_log_level)
        self._logger.debug("Configured MCP loggers")

        self._serve_task = asyncio.create_task(self._server.serve())
        self._logger.debug("MCP server task started")

    async def stop(self) -> None:
        """Stop the server."""
        import contextlib  # noqa: PLC0415

        if self._serve_task is None:
            self._logger.debug("Stop called but no server task running")
            return

        self._logger.info("Stopping MCP server...")

        # Signal graceful shutdown
        self._server.should_exit = True
        self._logger.debug("Signaled graceful shutdown")

        # Wait briefly for graceful shutdown
        try:
            await asyncio.wait_for(asyncio.shield(self._serve_task), timeout=2.0)
            self._logger.debug("Graceful shutdown completed")
        except TimeoutError:
            # Force exit
            self._logger.debug("Graceful shutdown timed out, forcing exit")
            self._server.force_exit = True
            try:
                await asyncio.wait_for(asyncio.shield(self._serve_task), timeout=1.0)
                self._logger.debug("Forced shutdown completed")
            except TimeoutError:
                # Cancel the task
                self._logger.debug("Forced shutdown timed out, cancelling task")
                self._serve_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._serve_task
                self._logger.debug("Task cancelled")
        except asyncio.CancelledError:
            self._logger.debug("Shutdown cancelled, forcing exit")
            self._server.force_exit = True
            self._serve_task.cancel()

        self._serve_task = None
        self._logger.info("MCP server stopped")


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
