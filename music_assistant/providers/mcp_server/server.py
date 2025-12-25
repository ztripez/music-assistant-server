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


async def start_mcp_server(
    mass: MusicAssistant,
    port: int,
    require_auth: bool = True,
    enabled_features: dict[str, bool] | None = None,
) -> tuple[asyncio.Task[None], asyncio.Event]:
    """Start the MCP server.

    :param mass: MusicAssistant instance.
    :param port: Port to run the server on.
    :param require_auth: Whether to require authentication.
    :param enabled_features: Dictionary of feature flags to enable/disable tool categories.
    :return: Tuple of (server task, shutdown event).
    """
    import contextlib  # noqa: PLC0415

    mcp = create_mcp_server(mass, require_auth, enabled_features)
    shutdown_event = asyncio.Event()

    async def run_server() -> None:
        """Run the uvicorn server."""
        import uvicorn  # noqa: PLC0415

        # Configure the MCP server path
        mcp.settings.streamable_http_path = "/"

        config = uvicorn.Config(
            app=mcp.streamable_http_app(),
            host="0.0.0.0",
            port=port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)

        try:
            # Run server until shutdown is requested
            server_task = asyncio.create_task(server.serve())

            # Wait for shutdown signal
            await shutdown_event.wait()

            # Graceful shutdown - give connections 2 seconds to close
            server.should_exit = True
            try:
                await asyncio.wait_for(server_task, timeout=2.0)
            except TimeoutError:
                # Force exit if connections don't close in time
                server.force_exit = True
                try:
                    await asyncio.wait_for(server_task, timeout=1.0)
                except TimeoutError:
                    server_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await server_task
        except asyncio.CancelledError:
            # Handle task cancellation during shutdown
            server.force_exit = True
            raise

    task = asyncio.create_task(run_server())
    return task, shutdown_event
