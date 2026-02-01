"""
DEMO/TEMPLATE Audio Analysis Provider for Music Assistant.

This is a demo audio analysis provider that shows how to receive and process
mel spectrogram data for ML/AI analysis. Use it as a reference to build
your own audio analysis providers.

Audio Analysis Providers receive mel spectrogram data during playback and can:
- Run genre/mood classification models
- Perform audio fingerprinting
- Generate audio embeddings (CLAP, OpenL3, etc.)
- Any other ML/AI audio analysis tasks

IMPORTANT NOTES:
- Audio analysis providers currently use the "plugin" provider type
  until music-assistant-models is updated with a dedicated type
- The mel spectrogram is provided as raw numpy float32 bytes
- Each mel spectrogram covers approximately 1 second of audio
- You can customize the mel spectrogram parameters by overriding
  get_mel_spectrogram_config()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from music_assistant.models.audio_analysis_provider import (
    AudioAnalysisProvider,
    MelSpectrogramConfig,
)

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigEntry, ConfigValueType, ProviderConfig
    from music_assistant_models.enums import ProviderFeature
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

# No special features needed for this demo
SUPPORTED_FEATURES: set[ProviderFeature] = set()


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return DemoAudioAnalysisProvider(mass, manifest, config, SUPPORTED_FEATURES)


async def get_config_entries(
    mass: MusicAssistant,
    instance_id: str | None = None,
    action: str | None = None,
    values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    """Return Config entries to setup this provider."""
    # ruff: noqa: ARG001
    # This demo provider has no configuration options
    return ()


class DemoAudioAnalysisProvider(AudioAnalysisProvider):
    """Demo Audio Analysis Provider.

    This provider demonstrates how to receive and process mel spectrogram data.
    It logs information about received spectrograms but doesn't perform
    actual ML analysis.

    To create your own audio analysis provider:
    1. Inherit from AudioAnalysisProvider
    2. Override get_mel_spectrogram_config() if you need custom parameters
    3. Implement on_mel_spectrogram() to process the data
    """

    def get_mel_spectrogram_config(self) -> MelSpectrogramConfig:
        """Return the mel spectrogram configuration for this provider.

        This demo uses the default librosa configuration:
        - n_mels=128 (number of mel frequency bands)
        - n_fft=2048 (FFT window size)
        - hop_length=512 (samples between frames)

        Override this to customize for your ML model. Common configurations:
        - CLAP: n_mels=64, n_fft=1024, hop_length=480
        - VGGish: n_mels=64, n_fft=400, hop_length=160
        - OpenL3: n_mels=128, n_fft=2048, hop_length=242
        """
        return MelSpectrogramConfig()

    async def on_mel_spectrogram(
        self,
        queue_id: str,
        queue_item_id: str,
        track_uri: str,
        timestamp: float,
        mel_spectrogram: bytes,
        sample_rate: int,
        config: MelSpectrogramConfig,
    ) -> None:
        """Process received mel spectrogram data.

        This is where you would run your ML model on the spectrogram.
        For this demo, we just log information about the received data.

        :param queue_id: The player queue ID.
        :param queue_item_id: The queue item ID being played.
        :param track_uri: URI of the track being analyzed.
        :param timestamp: Seconds into the track where this chunk starts.
        :param mel_spectrogram: Raw numpy float32 array bytes.
        :param sample_rate: Audio sample rate in Hz.
        :param config: The mel spectrogram configuration used.
        """
        # Decode the mel spectrogram from raw bytes
        mel_array = np.frombuffer(mel_spectrogram, dtype=np.float32)

        # Reshape the mel spectrogram array
        # Time frames depend on audio duration and hop_length
        # For 1 second of audio: time_frames ≈ sample_rate / hop_length
        try:
            mel_array = mel_array.reshape(config.n_mels, -1)
        except ValueError as err:
            msg = f"Could not reshape mel spectrogram array of size {mel_array.size}"
            raise ValueError(msg) from err

        # Log information about the received spectrogram
        self.logger.debug(
            "Received mel spectrogram for %s at %.1fs: "
            "shape=%s, sample_rate=%d, config=(n_mels=%d, n_fft=%d, hop_length=%d)",
            track_uri,
            timestamp,
            mel_array.shape,
            sample_rate,
            config.n_mels,
            config.n_fft,
            config.hop_length,
        )

        # Example: Calculate some basic statistics (replace with your ML model)
        mel_mean = float(np.mean(mel_array))
        mel_std = float(np.std(mel_array))
        mel_max = float(np.max(mel_array))

        self.logger.debug(
            "Mel spectrogram stats: mean=%.4f, std=%.4f, max=%.4f",
            mel_mean,
            mel_std,
            mel_max,
        )

        # In a real implementation, you would:
        # 1. Feed mel_array to your ML model
        # 2. Get predictions (genre, mood, embeddings, etc.)
        # 3. Store or use the results as needed
        #
        # Example pseudocode:
        # predictions = await asyncio.to_thread(self.model.predict, mel_array)
        # await self.store_predictions(track_uri, timestamp, predictions)
