"""Audio Analysis Controller - manages audio analysis providers and mel spectrogram generation.

This controller coordinates the generation and distribution of mel spectrogram data
to audio analysis providers. It acts as the bridge between the audio streaming pipeline
and ML/AI analysis providers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from music_assistant.constants import MASS_LOGGER_NAME
from music_assistant.models.audio_analysis_provider import (
    AudioAnalysisProvider,
    MelSpectrogramConfig,
)

from .mel_spectrogram import MelSpectrogramAnalyzer

if TYPE_CHECKING:
    from music_assistant_models.media_items import AudioFormat

    from music_assistant.mass import MusicAssistant


class AudioAnalysisController:
    """Controller for audio analysis features.

    This controller manages audio analysis providers and coordinates the generation
    of mel spectrograms from the audio pipeline. It groups providers by their
    mel spectrogram configuration to avoid duplicate computation.
    """

    domain = "audio_analysis"

    def __init__(self, mass: MusicAssistant) -> None:
        """Initialize the audio analysis controller.

        :param mass: The MusicAssistant instance.
        """
        self.mass = mass
        mass_logger = logging.getLogger(MASS_LOGGER_NAME)
        self.logger = mass_logger.getChild(self.domain)
        self._analyzer = MelSpectrogramAnalyzer()

    def get_providers(self) -> list[AudioAnalysisProvider]:
        """Get all loaded audio analysis providers."""
        return [
            prov
            for prov in self.mass._providers.values()
            if isinstance(prov, AudioAnalysisProvider)
        ]

    def has_active_providers(self) -> bool:
        """Check if any audio analysis providers are loaded."""
        return len(self.get_providers()) > 0

    async def process_pcm_chunk(
        self,
        queue_id: str,
        queue_item_id: str,
        track_uri: str,
        pcm_chunk: bytes,
        pcm_format: AudioFormat,
        timestamp: float,
    ) -> None:
        """Process a PCM chunk and dispatch mel spectrograms to providers.

        This method is called from the streams controller for each 1-second
        PCM audio chunk during playback. It generates mel spectrograms
        according to each provider's configuration and dispatches them.

        :param queue_id: The player queue ID.
        :param queue_item_id: The queue item ID being played.
        :param track_uri: URI of the track being analyzed.
        :param pcm_chunk: Raw PCM audio data (1 second chunk).
        :param pcm_format: Audio format specification.
        :param timestamp: Seconds into the track where this chunk starts.
        """
        providers = self.get_providers()
        if not providers:
            return

        # Group providers by their mel spectrogram config to avoid duplicate computation
        configs_to_providers: dict[MelSpectrogramConfig, list[AudioAnalysisProvider]] = {}
        for prov in providers:
            config = prov.get_mel_spectrogram_config()
            if config not in configs_to_providers:
                configs_to_providers[config] = []
            configs_to_providers[config].append(prov)

        for config, target_providers in configs_to_providers.items():
            try:
                mel_spectrogram_bytes = await self._analyzer.analyze(pcm_chunk, pcm_format, config)

                for prov in target_providers:
                    self.mass.create_task(
                        prov.on_mel_spectrogram(
                            queue_id=queue_id,
                            queue_item_id=queue_item_id,
                            track_uri=track_uri,
                            timestamp=timestamp,
                            mel_spectrogram=mel_spectrogram_bytes,
                            sample_rate=pcm_format.sample_rate,
                            config=config,
                        )
                    )
            except Exception:
                self.logger.exception(
                    "Mel spectrogram generation failed for config %s (providers: %s)",
                    config,
                    [p.name for p in target_providers],
                )
