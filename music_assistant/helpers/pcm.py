"""PCM audio data utilities."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from music_assistant.constants import MASS_LOGGER_NAME

if TYPE_CHECKING:
    from music_assistant_models.media_items import AudioFormat

LOGGER = logging.getLogger(f"{MASS_LOGGER_NAME}.helpers.pcm")


class PCMConversionError(Exception):
    """Exception raised when PCM conversion fails."""


def align_pcm_to_frame_boundary(audio_data: bytes, pcm_format: AudioFormat) -> bytes:
    """Align audio data to frame boundaries by truncating incomplete frames.

    :param audio_data: Raw PCM audio data to align.
    :param pcm_format: AudioFormat of the audio data.
    """
    bytes_per_sample = pcm_format.bit_depth // 8
    frame_size = bytes_per_sample * pcm_format.channels
    valid_bytes = (len(audio_data) // frame_size) * frame_size
    if valid_bytes != len(audio_data):
        LOGGER.debug(
            "Truncating %d bytes from audio buffer to align to frame boundary",
            len(audio_data) - valid_bytes,
        )
        return audio_data[:valid_bytes]
    return audio_data


def pcm_to_mono(
    pcm_data: bytes,
    pcm_format: AudioFormat,
    *,
    truncate_misaligned: bool = False,
) -> npt.NDArray[np.float32]:
    """Convert PCM bytes to mono float32 numpy array.

    :param pcm_data: Raw PCM audio data (float32 format).
    :param pcm_format: Audio format specification.
    :param truncate_misaligned: If True, truncate misaligned buffers instead of raising.
    :raises PCMConversionError: If audio buffer is not properly aligned and truncate is False.
    """
    audio_array = np.frombuffer(pcm_data, dtype=np.float32)

    if pcm_format.channels > 1:
        samples_per_channel = len(audio_array) // pcm_format.channels
        valid_samples = samples_per_channel * pcm_format.channels

        if valid_samples != len(audio_array):
            if truncate_misaligned:
                LOGGER.warning(
                    "Audio buffer size (%d) not divisible by channels (%d), truncating %d samples",
                    len(audio_array),
                    pcm_format.channels,
                    len(audio_array) - valid_samples,
                )
                audio_array = audio_array[:valid_samples]
            else:
                msg = (
                    f"Audio buffer size ({len(audio_array)}) not divisible by channels "
                    f"({pcm_format.channels}). This indicates a bug in the audio pipeline."
                )
                raise PCMConversionError(msg)

        audio_array = audio_array.reshape(-1, pcm_format.channels)
        return np.asarray(np.mean(audio_array, axis=1, dtype=np.float32))

    return np.asarray(audio_array, dtype=np.float32)
