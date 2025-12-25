"""Audio streaming to insight sidecar for real-time audio embeddings."""

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


async def _get_error_body(resp: aiohttp.ClientResponse) -> str:
    """Safely extract error body from response (msgpack or text)."""
    try:
        data = await resp.read()
        try:
            result = msgpack.unpackb(data, raw=False)
            if isinstance(result, dict):
                return str(result.get("error", result))
            return str(result)
        except Exception:
            return data.decode("utf-8", errors="replace")
    except Exception:
        return "<failed to read response body>"


@dataclass
class StreamSession:
    """Tracks an active streaming session with the sidecar."""

    session_id: str
    track_id: str
    queue_item_id: str
    buffer: bytearray = field(default_factory=bytearray)
    bytes_sent: int = 0


class AudioStreamer:
    """
    Manages audio streaming sessions with the insight sidecar.

    Subscribes to audio frame events from MusicAssistant and streams PCM data
    to the sidecar's streaming API for real-time audio embedding generation.

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

        # Buffer ~1 second of audio before sending (48kHz stereo f32 = 384KB)
        self.min_buffer_bytes = 48000 * 4 * 2

    async def start(self) -> None:
        """Start listening for audio frames."""
        self._http_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        self._unsubscribe = self.mass.subscribe_audio_frames(self._on_audio_frame)
        self.logger.info("Audio streamer started, listening for audio frames")

    async def stop(self) -> None:
        """Stop listening and close all sessions."""
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None

        # End all active sessions without storing (incomplete data)
        for queue_item_id in list(self._sessions.keys()):
            await self._end_session(queue_item_id, store=False)

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
        :param chunk: Raw PCM audio bytes.
        :param sample_rate: Sample rate in Hz.
        :param channels: Number of audio channels.
        """
        if not track_id:
            return

        session = self._sessions.get(queue_item_id)

        if session is None:
            # New track starting - end any other active sessions
            # (only one track can play at a time per output)
            for old_queue_item_id in list(self._sessions.keys()):
                if old_queue_item_id != queue_item_id:
                    await self._end_session(old_queue_item_id, store=True)

            # Start new session with sidecar
            session = await self._start_session(queue_item_id, track_id, sample_rate, channels)
            if session is None:
                return

        # Add chunk to buffer
        session.buffer.extend(chunk)

        # Send when buffer is large enough to reduce HTTP overhead
        if len(session.buffer) >= self.min_buffer_bytes:
            if not await self._send_frames(session):
                # Session no longer exists on sidecar, remove from local tracking
                self._sessions.pop(queue_item_id, None)

    async def _start_session(
        self,
        queue_item_id: str,
        track_id: str,
        sample_rate: int,
        channels: int,
    ) -> StreamSession | None:
        """
        Start a new streaming session with the sidecar.

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

        # Build metadata for the sidecar
        metadata = {
            "name": track.name,
            "artists": [a.name for a in track.artists] if track.artists else [],
            "album": track.album.name if track.album else None,
            "genres": list(track.metadata.genres) if track.metadata.genres else [],
        }

        # Start session with sidecar
        payload = {
            "track_id": track_id,
            "metadata": metadata,
            "format": "pcm_f32_le",
            "sample_rate": sample_rate,
            "channels": channels,
        }

        try:
            url = f"{self.sidecar_url}/api/v1/stream/start"
            async with self._http_session.post(
                url,
                data=msgpack.packb(payload),
                headers={"Content-Type": "application/msgpack"},
            ) as resp:
                if resp.status != 200:
                    body = await _get_error_body(resp)
                    self.logger.warning(
                        "Failed to start stream session for %s: %s - %s",
                        track_id,
                        resp.status,
                        body,
                    )
                    return None

                result = msgpack.unpackb(await resp.read(), raw=False)
                session_id = result["session_id"]
        except aiohttp.ClientError as err:
            self.logger.warning("Failed to connect to sidecar: %s", err)
            return None
        except Exception as err:
            self.logger.warning("Failed to start stream session: %s", err)
            return None

        session = StreamSession(
            session_id=session_id,
            track_id=track_id,
            queue_item_id=queue_item_id,
        )
        self._sessions[queue_item_id] = session
        self.logger.info("Started stream session %s for track %s", session_id, track_id)
        return session

    async def _send_frames(self, session: StreamSession) -> bool:
        """
        Send buffered PCM frames to the sidecar.

        :param session: The streaming session.
        :return: True if successful, False if session is invalid/gone.
        """
        if not self._http_session or not session.buffer:
            return True

        # Extract buffer contents atomically (but keep a copy in case of 404)
        async with self._send_lock:
            data = bytes(session.buffer)
            session.buffer.clear()

        try:
            url = f"{self.sidecar_url}/api/v1/stream/{session.session_id}/frames"
            async with self._http_session.post(
                url,
                data=data,
                headers={"Content-Type": "application/octet-stream"},
            ) as resp:
                if resp.status == 200:
                    session.bytes_sent += len(data)
                    return True
                if resp.status == 404:
                    # Session doesn't exist on sidecar anymore (sidecar restarted?)
                    # Log at info level since this triggers reconnection
                    self.logger.info(
                        "Stream session %s lost (sidecar restarted?), will reconnect on next frame",
                        session.session_id,
                    )
                    return False
                self.logger.debug(
                    "Failed to send frames for session %s: %s",
                    session.session_id,
                    resp.status,
                )
                return True  # Non-404 errors are transient, keep trying
        except aiohttp.ClientError as err:
            self.logger.debug("Failed to send frames: %s", err)
            return True  # Network errors are transient
        except Exception as err:
            self.logger.warning("Unexpected error sending frames: %s", err)
            return True

    async def end_session_for_track(self, queue_item_id: str, store: bool = True) -> None:
        """
        End a streaming session when track playback ends.

        :param queue_item_id: The queue item ID.
        :param store: Whether to store the resulting embeddings.
        """
        await self._end_session(queue_item_id, store=store)

    async def _end_session(self, queue_item_id: str, store: bool = True) -> None:
        """
        End a streaming session with the sidecar.

        :param queue_item_id: The queue item ID.
        :param store: Whether to store the resulting embeddings.
        """
        session = self._sessions.pop(queue_item_id, None)
        if not session or not self._http_session:
            return

        # Send any remaining buffered data
        if session.buffer:
            await self._send_frames(session)

        # End session with sidecar
        payload = {
            "store": store,
            "min_duration_s": 3.0,
        }

        try:
            url = f"{self.sidecar_url}/api/v1/stream/{session.session_id}/end"
            async with self._http_session.post(
                url,
                data=msgpack.packb(payload),
                headers={"Content-Type": "application/msgpack"},
            ) as resp:
                if resp.status == 200:
                    result = msgpack.unpackb(await resp.read(), raw=False)
                    self.logger.info(
                        "Stream session ended: track=%s, audio_stored=%s, duration=%.1fs",
                        session.track_id,
                        result.get("audio_stored", False),
                        result.get("duration_s", 0),
                    )
                elif resp.status == 404:
                    # Session doesn't exist on sidecar (maybe it restarted or timed out)
                    # This is not an error - the session is effectively ended
                    self.logger.debug(
                        "Stream session %s not found on sidecar (already ended or expired)",
                        session.session_id,
                    )
                else:
                    body = await _get_error_body(resp)
                    self.logger.warning(
                        "Failed to end stream session %s: %s - %s",
                        session.session_id,
                        resp.status,
                        body,
                    )
        except aiohttp.ClientError as err:
            self.logger.debug("Failed to end stream session: %s", err)
        except Exception as err:
            self.logger.warning("Unexpected error ending stream session: %s", err)
