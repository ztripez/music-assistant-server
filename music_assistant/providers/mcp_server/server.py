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
    from starlette.applications import Starlette

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


def create_mcp_asgi_app(
    mass: MusicAssistant,
    require_auth: bool = True,
    enabled_features: dict[str, bool] | None = None,
) -> Starlette:
    """Create the MCP ASGI application.

    :param mass: MusicAssistant instance.
    :param require_auth: Whether to require authentication.
    :param enabled_features: Dictionary of feature flags to enable/disable tool categories.
    :return: Starlette ASGI application.
    """
    from starlette.middleware.cors import CORSMiddleware  # noqa: PLC0415

    mcp = create_mcp_server(mass, require_auth, enabled_features)
    app = mcp.streamable_http_app()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    return app


async def run_mcp_server(
    mass: MusicAssistant,
    host: str,
    port: int,
    require_auth: bool = True,
    enabled_features: dict[str, bool] | None = None,
) -> asyncio.Task[None]:
    """Start the MCP server using uvicorn.

    :param mass: MusicAssistant instance.
    :param host: Host address to bind to.
    :param port: Port number to bind to.
    :param require_auth: Whether to require authentication.
    :param enabled_features: Dictionary of feature flags to enable/disable tool categories.
    :return: The asyncio task running the server.
    """
    import uvicorn  # noqa: PLC0415

    app = create_mcp_asgi_app(mass, require_auth, enabled_features)

    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)

    task = asyncio.create_task(server.serve())
    LOGGER.info("MCP server started on http://%s:%d/mcp", host, port)
    return task
