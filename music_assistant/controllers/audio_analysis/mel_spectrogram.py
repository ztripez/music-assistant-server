"""Mel Spectrogram Analyzer - generates mel spectrograms from PCM audio data.

This module provides the core mel spectrogram generation functionality using librosa.
The computation runs in a thread pool to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import librosa
import numpy as np
import numpy.typing as npt

from music_assistant.constants import MASS_LOGGER_NAME, VERBOSE_LOG_LEVEL
from music_assistant.helpers.pcm import (
    PCMConversionError,
    align_pcm_to_frame_boundary,
    pcm_to_mono,
)

if TYPE_CHECKING:
    from music_assistant_models.media_items import AudioFormat

    from music_assistant.models.audio_analysis_provider import MelSpectrogramConfig

LOGGER = logging.getLogger(f"{MASS_LOGGER_NAME}.audio_analysis.mel_spectrogram")


class AudioAnalysisError(PCMConversionError):
    """Exception raised when audio analysis fails."""


class MelSpectrogramAnalyzer:
    """Generates mel spectrograms from PCM audio data.

    Uses librosa.feature.melspectrogram for computation. The analysis runs
    in a thread pool via asyncio.to_thread to prevent blocking.
    """

    def _compute_mel_spectrogram(
        self,
        audio_array: npt.NDArray[np.float32],
        sample_rate: int,
        config: MelSpectrogramConfig,
    ) -> npt.NDArray[np.float32]:
        """Compute mel spectrogram using librosa.

        This method runs in a thread pool to avoid blocking.

        :param audio_array: Mono audio samples as float32.
        :param sample_rate: Audio sample rate in Hz.
        :param config: Mel spectrogram configuration.
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio_array,
            sr=sample_rate,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            fmin=config.fmin,
            fmax=config.fmax,
            power=config.power,
        )
        return mel_spec.astype(np.float32)

    async def analyze(
        self,
        pcm_chunk: bytes,
        pcm_format: AudioFormat,
        config: MelSpectrogramConfig,
    ) -> bytes:
        """Generate mel spectrogram from PCM audio chunk.

        :param pcm_chunk: Raw PCM audio data (1 second chunk).
        :param pcm_format: Audio format specification.
        :param config: Mel spectrogram configuration.
        :returns: Raw numpy float32 array bytes (shape: n_mels x time_frames).
        :raises AudioAnalysisError: If audio data is invalid (NaN/Inf values).
        """
        pcm_chunk = align_pcm_to_frame_boundary(pcm_chunk, pcm_format)

        LOGGER.log(
            VERBOSE_LOG_LEVEL,
            "Analyzing PCM chunk: %d bytes, %d Hz, %d channels",
            len(pcm_chunk),
            pcm_format.sample_rate,
            pcm_format.channels,
        )

        try:
            mono_audio = pcm_to_mono(pcm_chunk, pcm_format)
        except PCMConversionError as err:
            raise AudioAnalysisError(str(err)) from err

        if not np.all(np.isfinite(mono_audio)):
            msg = "Audio buffer contains non-finite values (NaN/Inf)"
            raise AudioAnalysisError(msg)

        mel_spec = await asyncio.to_thread(
            self._compute_mel_spectrogram,
            mono_audio,
            pcm_format.sample_rate,
            config,
        )

        LOGGER.log(
            VERBOSE_LOG_LEVEL,
            "Generated mel spectrogram: shape=%s",
            mel_spec.shape,
        )

        return mel_spec.tobytes()
