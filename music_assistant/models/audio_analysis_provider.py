"""Model/base for an Audio Analysis Provider implementation.

Audio Analysis Providers receive mel spectrogram data from the audio pipeline
and can use it for ML/AI analysis such as genre classification, mood detection,
audio fingerprinting, or other signal processing tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

from .provider import Provider


@dataclass(frozen=True)
class MelSpectrogramConfig:
    """Configuration for mel spectrogram generation.

    Different ML models require different mel spectrogram configurations.
    Providers can override get_mel_spectrogram_config() to specify their needs.

    Common configurations by model type:
    - CLAP: n_mels=64, n_fft=1024, hop_length=480
    - VGGish: n_mels=64, n_fft=400, hop_length=160 (at 16kHz)
    - OpenL3: n_mels=128, n_fft=2048, hop_length=242
    - librosa default: n_mels=128, n_fft=2048, hop_length=512
    """

    n_mels: int = 128
    """Number of mel frequency bands."""

    n_fft: int = 2048
    """FFT window size in samples."""

    hop_length: int = 512
    """Number of samples between successive frames."""

    fmin: float = 0.0
    """Lowest frequency (Hz)."""

    fmax: float | None = None
    """Highest frequency (Hz). None means sample_rate / 2."""

    power: float = 2.0
    """Exponent for the magnitude spectrogram. 1 for energy, 2 for power."""


class AudioAnalysisProvider(Provider):
    """Base representation of an Audio Analysis Provider.

    Audio Analysis Provider implementations should inherit from this base model.
    These providers receive mel spectrogram data generated from the audio stream
    during playback and can use it for ML/AI analysis.

    To implement an audio analysis provider:
    1. Inherit from this class
    2. Override get_mel_spectrogram_config() if you need custom parameters
    3. Implement on_mel_spectrogram() to process the data

    Note: Until music-assistant-models is updated with AUDIO_ANALYSIS provider type,
    audio analysis providers use the PLUGIN provider type in their manifest.
    """

    def get_mel_spectrogram_config(self) -> MelSpectrogramConfig:
        """Return the mel spectrogram configuration this provider needs.

        Override this method to specify custom mel spectrogram parameters
        for your ML model. The default returns standard librosa defaults.
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
        """Handle a mel spectrogram generated for audio being played.

        This method is called for each 1-second chunk of audio during playback.
        The mel spectrogram is provided as raw numpy float32 array bytes.

        To decode the mel spectrogram in your implementation:
        ```python
        import numpy as np
        mel_array = np.frombuffer(mel_spectrogram, dtype=np.float32).reshape(config.n_mels, -1)
        ```

        :param queue_id: The player queue ID.
        :param queue_item_id: The queue item ID being played.
        :param track_uri: URI of the track being analyzed.
        :param timestamp: Seconds into the track where this chunk starts.
        :param mel_spectrogram: Raw numpy float32 array bytes.
        :param sample_rate: Audio sample rate in Hz.
        :param config: The mel spectrogram configuration used.
        """
        raise NotImplementedError
