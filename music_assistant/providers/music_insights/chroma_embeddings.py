"""ChromaDB embeddings handler for Music Assistant."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any, cast

import chromadb
import numpy as np
import torch
from music_assistant_models.media_items import Track
from music_assistant_models.unique_list import UniqueList
from transformers import BitsAndBytesConfig
from transformers.models.clap import ClapConfig, ClapModel, ClapProcessor

from music_assistant.helpers.util import LOGGER

TRACK_COLLECTION_NAME = "tracks"
USER_INTERACTION_COLLECTION = "user_interactions"

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant


class ChromaEmbeddings:
    """Handles audio and text embeddings using ChromaDB and a CLAP model."""

    def __init__(
        self,
        mass: MusicAssistant,
        model_name: str = "laion/clap-htsat-fused",
        enable_cuda: bool = False,
        audio_window_s: float = 10.0,
        audio_hop_s: float = 10.0,
    ) -> None:
        """Initialize ChromaEmbeddings."""
        self.mass = mass
        self.enable_cuda = enable_cuda
        self.audio_window_s = audio_window_s
        self.audio_hop_s = audio_hop_s
        self.model_name = model_name
        self.logger = LOGGER.getChild("music_insights.chroma")
        db_path = os.path.join(mass.storage_path, "music_insights")
        self.chromadb_client = chromadb.PersistentClient(path=db_path)
        self.track_collection = self.chromadb_client.get_or_create_collection(
            TRACK_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        self.user_collection = self.chromadb_client.get_or_create_collection(
            USER_INTERACTION_COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    async def async_init(self) -> None:
        await self.mass.create_task(self._setup_models())

    def _setup_models(self) -> None:
        clap_config = ClapConfig.from_pretrained(self.model_name)
        if self.enable_cuda and torch.cuda.is_available():
            self.model = (
                ClapModel.from_pretrained(self.model_name, config=clap_config).half().to("cuda")
            )
        else:
            qconf: BitsAndBytesConfig = BitsAndBytesConfig(load_in_8bit=True)
            self.model = ClapModel.from_pretrained(
                self.model_name, config=clap_config, quantization_config=qconf
            )
        self.model.eval()
        self.processor = ClapProcessor.from_pretrained(self.model_name)

    async def embeddings_from_str(self, input_str: str) -> np.ndarray[Any, Any]:
        """Generate embeddings for a given text string."""
        return await asyncio.to_thread(self._embed_text, input_str)

    def _embed_text(self, input_str: str) -> np.ndarray[Any, Any]:
        inputs = self.processor(text=input_str, return_tensors="pt", padding=True)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs.to(self.model.device))
        return cast("np.ndarray[Any, Any]", feats.squeeze(0).cpu().numpy())

    async def embeddings_from_audio(
        self, samples: np.ndarray[Any, Any], sr: int = 48_000
    ) -> np.ndarray[Any, Any]:
        """Generate embeddings for given audio samples."""
        return await asyncio.to_thread(self._embed_audio, samples, sr)

    def _embed_audio(self, samples: np.ndarray[Any, Any], sr: int) -> np.ndarray[Any, Any]:
        inputs = self.processor(audios=samples, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            feats = self.model.get_audio_features(**inputs.to(self.model.device))
        return cast("np.ndarray[Any, Any]", feats.squeeze(0).cpu().numpy())

    def _window_embed(
        self, samples: np.ndarray[Any, Any], sr: int = 48_000
    ) -> np.ndarray[Any, Any]:
        step = int(sr * self.audio_hop_s)
        win = int(sr * self.audio_window_s)
        if len(samples) <= win:
            return self._embed_audio(samples, sr)
        chunks = [
            self._embed_audio(samples[i : i + win], sr)
            for i in range(0, len(samples) - win + 1, step)
        ]
        return cast("np.ndarray[Any, Any]", np.mean(np.vstack(chunks), axis=0))

    async def upsert_track(self, track: Track) -> None:
        """Generate and upsert embeddings for a track."""
        album_name = track.album.name if track.album else "Unknown Album"
        artist_names = [a.name for a in track.artists] if track.artists else []
        genres = ",".join(track.metadata.genres or [])
        mood = track.metadata.mood or ""
        text = f"{genres},{','.join(artist_names)},{track.name},{album_name},{mood}"
        text_emb = await self.embeddings_from_str(text)

        docs: list[str] = [text]
        embs: list[np.ndarray[Any, Any]] = [text_emb]
        ids: list[str] = [track.item_id]
        # waveform = None
        # sample_rate = None

        # NOTE: Audio embedding part is commented out
        # If enabled later, ensure preview object is available and handled correctly.

        await asyncio.to_thread(
            self.track_collection.upsert, documents=docs, embeddings=embs, ids=ids
        )
        self.logger.debug("Upserted track %s", track.item_id)

    async def remove_track(self, track_id: str) -> None:
        """Remove a track's embeddings from the collection."""
        await asyncio.to_thread(self.track_collection.delete, ids=[track_id])
        self.logger.debug("Removed track %s", track_id)

    async def get_similar_tracks(
        self, prov_track_id: str, limit: int = 25, cutoff: float = 0.5
    ) -> list[Track]:
        """Find tracks similar to a given track ID."""
        try:
            data = await asyncio.to_thread(
                self.track_collection.get, ids=[prov_track_id], include=["embeddings"]
            )
            embeddings = data.get("embeddings")
            if not embeddings:
                return []
            query_embedding = np.asarray(embeddings[0])
        except Exception:
            return []
        tracks = await self._query_tracks(
            query_embedding, cutoff=cutoff, limit=limit, exclude_id=prov_track_id
        )
        return list(tracks)

    async def search_tracks(
        self, search_query: str, limit: int = 50, cutoff: float = 0.4
    ) -> UniqueList[Track]:
        """Search for tracks based on a text query."""
        query_emb = await self.embeddings_from_str(search_query)
        return await self._query_tracks(query_emb, cutoff=cutoff, limit=limit)

    async def _query_tracks(
        self,
        query_embedding: np.ndarray[Any, Any],
        cutoff: float,
        limit: int | None = None,
        exclude_id: str | None = None,
    ) -> UniqueList[Track]:
        n_results = (limit or 100) + (1 if exclude_id else 0)
        try:
            res = await asyncio.to_thread(
                self.track_collection.query,
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                include=["documents", "distances"],
            )
        except Exception:
            return UniqueList()

        if not res or not res.get("ids"):
            return UniqueList()

        ids = res["ids"][0]
        distances = res.get("distances", [[]])
        dists = distances[0] if distances else []
        filtered_ids = [
            i for i, d in zip(ids, dists, strict=False) if d <= cutoff and i != exclude_id
        ]
        if limit:
            filtered_ids = filtered_ids[:limit]

        tasks = [
            self.mass.music.tracks.get(i.split("#")[0], provider_instance_id_or_domain="")
            for i in filtered_ids
        ]
        fetched = await asyncio.gather(*tasks, return_exceptions=True)
        return UniqueList([t for t in fetched if isinstance(t, Track)])

    async def cleanup(self) -> None:
        """Delete the ChromaDB collections."""
        await self._delete_collection(TRACK_COLLECTION_NAME)
        await self._delete_collection(USER_INTERACTION_COLLECTION)

    async def _delete_collection(self, collection_name: str) -> None:
        self.logger.info("Deleting ChromaDB collection %s", collection_name)
        await asyncio.to_thread(self.chromadb_client.delete_collection, name=collection_name)
