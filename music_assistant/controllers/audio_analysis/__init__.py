"""Audio Analysis Controller package.

Provides audio analysis capabilities for Music Assistant,
including mel spectrogram generation for ML/AI providers.
"""

from .audio_analysis_controller import AudioAnalysisController
from .mel_spectrogram import AudioAnalysisError

__all__ = ["AudioAnalysisController", "AudioAnalysisError"]
