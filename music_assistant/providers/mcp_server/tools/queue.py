"""Queue management tools for MCP server."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from music_assistant.mass import MusicAssistant


def register_queue_query_tools(mcp: FastMCP, mass: MusicAssistant) -> None:
    """Register queue query tools.

    :param mcp: FastMCP server instance.
    :param mass: MusicAssistant instance.
    """

    @mcp.tool()
    async def get_queue(player_id: str, limit: int = 50) -> str:
        """Get items in a player's queue.

        :param player_id: Player ID.
        :param limit: Maximum number of items to return.
        """
        try:
            items = mass.player_queues.items(player_id, limit=limit)
            queue = mass.player_queues.get(player_id)
            current_index = queue.current_index if queue else 0

            output = {
                "queue_id": player_id,
                "current_index": current_index,
                "total_items": len(items),
                "items": [
                    {
                        "index": i,
                        "queue_item_id": item.queue_item_id,
                        "name": item.name,
                        "uri": item.uri if hasattr(item, "uri") else None,
                        "duration": item.duration,
                        "is_current": i == current_index,
                    }
                    for i, item in enumerate(items)
                ],
            }
            return json.dumps(output, indent=2)
        except Exception as e:
            return f"Error: {e}"


def register_queue_control_tools(mcp: FastMCP, mass: MusicAssistant) -> None:
    """Register queue control tools.

    :param mcp: FastMCP server instance.
    :param mass: MusicAssistant instance.
    """

    @mcp.tool()
    async def shuffle_queue(player_id: str, enabled: bool) -> str:
        """Set shuffle mode.

        :param player_id: Player ID.
        :param enabled: Enable shuffle.
        """
        try:
            await mass.player_queues.set_shuffle(player_id, enabled)
            state = "enabled" if enabled else "disabled"
            return f"Shuffle {state} on {player_id}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def repeat_queue(player_id: str, mode: str) -> str:
        """Set repeat mode.

        :param player_id: Player ID.
        :param mode: off, one, or all.
        """
        try:
            from music_assistant_models.enums import RepeatMode  # noqa: PLC0415

            mode_map = {
                "off": RepeatMode.OFF,
                "one": RepeatMode.ONE,
                "all": RepeatMode.ALL,
            }
            repeat_mode = mode_map.get(mode.lower(), RepeatMode.OFF)
            mass.player_queues.set_repeat(player_id, repeat_mode)
            return f"Repeat mode set to '{mode}' on {player_id}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def play_queue_index(player_id: str, index: int) -> str:
        """Play a specific item in the queue by index.

        :param player_id: Player ID.
        :param index: The index of the item to play (0-based).
        """
        try:
            await mass.player_queues.play_index(player_id, index)
            return f"Playing queue item at index {index}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def transfer_queue(
        source_player_id: str,
        target_player_id: str,
    ) -> str:
        """Transfer a queue from one player to another.

        :param source_player_id: The player to transfer from.
        :param target_player_id: The player to transfer to.
        """
        try:
            await mass.player_queues.transfer_queue(source_player_id, target_player_id)
            return f"Queue transferred from {source_player_id} to {target_player_id}"
        except Exception as e:
            return f"Error: {e}"


def register_queue_edit_tools(mcp: FastMCP, mass: MusicAssistant) -> None:
    """Register queue edit tools.

    :param mcp: FastMCP server instance.
    :param mass: MusicAssistant instance.
    """

    @mcp.tool()
    async def move_queue_item(
        player_id: str,
        position_shift: int,
        queue_item_id: str | None = None,
        index: int | None = None,
    ) -> str:
        """Move an item in the queue by a relative position.

        :param player_id: Player ID.
        :param position_shift: Number of positions to move (+/- for up/down).
        :param queue_item_id: The queue_item_id from get_queue output.
        :param index: Alternatively, the index of the item to move (0-based).
        """
        if queue_item_id is None and index is None:
            return "Error: Must provide either queue_item_id or index"
        try:
            if queue_item_id is None:
                items = mass.player_queues.items(player_id)
                if index is None or index < 0 or index >= len(items):
                    return f"Error: Index {index} out of range"
                queue_item_id = items[index].queue_item_id
            mass.player_queues.move_item(player_id, queue_item_id, position_shift)
            direction = "up" if position_shift < 0 else "down"
            return f"Moved item {abs(position_shift)} positions {direction}"
        except Exception as e:
            return f"Error: {e}"


def register_queue_delete_tools(mcp: FastMCP, mass: MusicAssistant) -> None:
    """Register queue delete tools.

    :param mcp: FastMCP server instance.
    :param mass: MusicAssistant instance.
    """

    @mcp.tool()
    async def clear_queue(player_id: str) -> str:
        """Clear all items from a player's queue.

        :param player_id: Player ID.
        """
        try:
            mass.player_queues.clear(player_id)
            return f"Queue cleared on {player_id}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def remove_queue_item(
        player_id: str, queue_item_id: str | None = None, index: int | None = None
    ) -> str:
        """Remove an item from the queue.

        :param player_id: Player ID.
        :param queue_item_id: The queue_item_id from get_queue output.
        :param index: Alternatively, the index of the item to remove (0-based).
        """
        try:
            if queue_item_id is not None:
                mass.player_queues.delete_item(player_id, queue_item_id)
            elif index is not None:
                mass.player_queues.delete_item(player_id, index)
            else:
                return "Error: Must provide either queue_item_id or index"
            return "Removed item from queue"
        except Exception as e:
            return f"Error: {e}"
