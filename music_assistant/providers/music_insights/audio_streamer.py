"""Audio streaming to insight sidecar via WebSocket for real-time audio embeddings."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import aiohttp
import msgpack

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant


@dataclass
class StreamSession:
    """Tracks an active WebSocket streaming session with the sidecar."""

    ws: aiohttp.ClientWebSocketResponse
    track_id: str
    queue_item_id: str
    queue_id: str
    bytes_sent: int = 0
    buffer: bytearray = field(default_factory=bytearray)


class AudioStreamer:
    """
    Manages WebSocket audio streaming sessions with the insight sidecar.

    Subscribes to audio frame events from MusicAssistant and streams PCM data
    to the sidecar via WebSocket for crash-resistant audio embedding generation.

    One session is created per track playback (identified by queue_item_id).
    """

    def __init__(
        self,
        mass: MusicAssistant,
        sidecar_url: str,
        logger: logging.Logger,
    ) -> None:
        """
        Initialize AudioStreamer.

        :param mass: The MusicAssistant instance.
        :param sidecar_url: Base URL of the insight sidecar.
        :param logger: Logger instance.
        """
        self.mass = mass
        self.sidecar_url = sidecar_url.rstrip("/")
        self.logger = logger
        self._sessions: dict[str, StreamSession] = {}
        self._http_session: aiohttp.ClientSession | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self._send_lock = asyncio.Lock()

        # Buffer ~0.5 second of audio before sending over WebSocket
        # (reduces message overhead while keeping latency low)
        # 48kHz stereo s16le = 48000 * 2 * 2 = 192KB/s
        self.min_buffer_bytes = 48000 * 2 * 2 // 2  # ~0.5 seconds

    @property
    def ws_url(self) -> str:
        """Get WebSocket URL for audio streaming."""
        # Convert http(s):// to ws(s)://
        base = self.sidecar_url
        if base.startswith("https://"):
            base = "wss://" + base[8:]
        elif base.startswith("http://"):
            base = "ws://" + base[7:]
        return f"{base}/api/v1/ws/audio"

    async def start(self) -> None:
        """Start listening for audio frames."""
        self._http_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))
        self._unsubscribe = self.mass.subscribe_audio_frames(self._on_audio_frame)
        self.logger.info("Audio streamer started (WebSocket mode), listening for audio frames")

    async def stop(self) -> None:
        """Stop listening and close all sessions."""
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None

        # Close all active WebSocket connections
        for queue_item_id in list(self._sessions.keys()):
            await self._end_session(queue_item_id)

        if self._http_session:
            await self._http_session.close()
            self._http_session = None

        self.logger.info("Audio streamer stopped")

    async def _on_audio_frame(
        self,
        queue_item_id: str,
        track_id: str,
        chunk: bytes,
        sample_rate: int,
        channels: int,
    ) -> None:
        """
        Handle incoming audio frame from MusicAssistant.

        :param queue_item_id: Unique ID for the queue item being played.
        :param track_id: Track ID from the library.
        :param chunk: Raw PCM audio bytes (s16le format).
        :param sample_rate: Sample rate in Hz.
        :param channels: Number of audio channels.
        """
        if not track_id:
            return

        session = self._sessions.get(queue_item_id)

        if session is None:
            # New track starting - end any other active sessions
            for old_queue_item_id in list(self._sessions.keys()):
                if old_queue_item_id != queue_item_id:
                    await self._end_session(old_queue_item_id)

            # Start new WebSocket session
            session = await self._start_session(queue_item_id, track_id, sample_rate, channels)
            if session is None:
                return

        # Add chunk to buffer
        session.buffer.extend(chunk)

        # Send when buffer is large enough
        if len(session.buffer) >= self.min_buffer_bytes:
            await self._send_buffer(session)

    async def _start_session(
        self,
        queue_item_id: str,
        track_id: str,
        sample_rate: int,
        channels: int,
    ) -> StreamSession | None:
        """
        Start a new WebSocket streaming session with the sidecar.

        :param queue_item_id: Unique ID for the queue item.
        :param track_id: Track ID from the library.
        :param sample_rate: Sample rate in Hz.
        :param channels: Number of audio channels.
        :return: StreamSession if successful, None otherwise.
        """
        if not self._http_session:
            return None

        # Get track metadata from the library
        try:
            track = await self.mass.music.tracks.get_library_item(track_id)
        except Exception as err:
            self.logger.debug("Could not get track %s: %s", track_id, err)
            return None

        if not track:
            return None

        # Get queue_id (player_id) from the queue item
        queue_id = "unknown"
        try:
            # Try to find the queue item to get the queue_id
            for player in self.mass.players:
                if player.active_source and player.active_source == queue_item_id:
                    queue_id = player.player_id
                    break
        except Exception as err:
            self.logger.debug("Could not determine queue_id: %s", err)

        # Build metadata for the sidecar
        metadata = {
            "name": track.name,
            "artists": [a.name for a in track.artists] if track.artists else [],
            "album": track.album.name if track.album else None,
            "genres": list(track.metadata.genres) if track.metadata.genres else [],
        }

        # Build header message
        header = {
            "queue_item_id": queue_item_id,
            "track_id": track_id,
            "queue_id": queue_id,
            "sample_rate": sample_rate,
            "channels": channels,
            "metadata": metadata,
        }

        try:
            # Connect WebSocket
            ws = await self._http_session.ws_connect(
                self.ws_url,
                timeout=aiohttp.ClientWSTimeout(ws_close=10),
            )

            # Send header as msgpack binary
            await ws.send_bytes(msgpack.packb(header))

            # Wait for ack
            ack_msg = await asyncio.wait_for(ws.receive(), timeout=5)
            if ack_msg.type == aiohttp.WSMsgType.BINARY:
                ack = msgpack.unpackb(ack_msg.data, raw=False)
                if not ack.get("accepted", False):
                    error = ack.get("error", "Unknown error")
                    self.logger.warning(
                        "Sidecar rejected stream session for %s: %s", track_id, error
                    )
                    await ws.close()
                    return None
            else:
                self.logger.warning("Unexpected ack message type: %s", ack_msg.type)
                await ws.close()
                return None

        except TimeoutError:
            self.logger.warning("Timeout connecting to sidecar WebSocket")
            return None
        except aiohttp.ClientError as err:
            self.logger.warning("Failed to connect to sidecar WebSocket: %s", err)
            return None
        except Exception as err:
            self.logger.warning("Failed to start WebSocket session: %s", err)
            return None

        session = StreamSession(
            ws=ws,
            track_id=track_id,
            queue_item_id=queue_item_id,
            queue_id=queue_id,
        )
        self._sessions[queue_item_id] = session
        self.logger.info(
            "Started WebSocket stream session for track %s (queue_item=%s)",
            track_id,
            queue_item_id[:8],
        )
        return session

    async def _send_buffer(self, session: StreamSession) -> bool:
        """
        Send buffered PCM data over the WebSocket.

        :param session: The streaming session.
        :return: True if successful, False if session is closed.
        """
        if not session.buffer:
            return True

        async with self._send_lock:
            data = bytes(session.buffer)
            session.buffer.clear()

        try:
            await session.ws.send_bytes(data)
            session.bytes_sent += len(data)
            return True
        except Exception as err:
            self.logger.debug("Failed to send audio data: %s", err)
            # WebSocket is likely closed
            return False

    async def end_session_for_track(self, queue_item_id: str, store: bool = True) -> None:
        """
        End a streaming session when track playback ends.

        :param queue_item_id: The queue item ID.
        :param store: Whether to store the resulting embeddings (ignored for WebSocket).
        """
        await self._end_session(queue_item_id)

    async def end_sessions_for_track_id(self, track_id: str, store: bool = True) -> None:
        """
        End all streaming sessions for a given track ID.

        This is used when MEDIA_ITEM_PLAYED event fires, which provides track_id
        but not queue_item_id.

        :param track_id: The track ID from the library.
        :param store: Whether to store the resulting embeddings (ignored for WebSocket).
        """
        matching_queue_ids = [
            qid for qid, session in self._sessions.items() if session.track_id == track_id
        ]
        if not matching_queue_ids:
            self.logger.debug(
                "No active sessions found for track_id %s (active sessions: %s)",
                track_id,
                list(self._sessions.keys()),
            )
            return

        self.logger.debug(
            "Found %d session(s) for track_id %s: %s",
            len(matching_queue_ids),
            track_id,
            matching_queue_ids,
        )
        for queue_item_id in matching_queue_ids:
            await self._end_session(queue_item_id)

    async def _end_session(self, queue_item_id: str) -> None:
        """
        End a WebSocket streaming session.

        Closing the WebSocket signals the sidecar to finalize the session
        and queue it for processing.

        :param queue_item_id: The queue item ID.
        """
        session = self._sessions.pop(queue_item_id, None)
        if not session:
            return

        # Send any remaining buffered data
        if session.buffer:
            await self._send_buffer(session)

        # Close WebSocket (this signals end of stream to sidecar)
        try:
            await session.ws.close()
            self.logger.info(
                "Stream session ended: track=%s, bytes_sent=%d",
                session.track_id,
                session.bytes_sent,
            )
        except Exception as err:
            self.logger.debug("Error closing WebSocket: %s", err)
