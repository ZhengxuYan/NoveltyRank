"""Utilities for loading arXiv metadata and representing similar papers."""
from __future__ import annotations

import logging
import math
import os
import pickle
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Sequence

from datasets import load_from_disk


logger = logging.getLogger("similarity_report.metadata")


@dataclass(slots=True)
class SimilarPaper:
    """Lightweight container for similar paper metadata required in prompts."""

    arxiv_id: str
    title: str
    abstract: str
    embedding: List[float]
    similarity_score: float
    publication_date: str

    @classmethod
    def from_raw(
        cls,
        entry: Dict[str, Any],
        metadata_lookup: Callable[[str], Dict[str, str]],
    ) -> "SimilarPaper":
        arxiv_id = entry.get("arxiv_id") or ""
        embedding_raw = entry.get("embedding") or []
        embedding = [float(value) for value in embedding_raw]
        similarity_score_raw = entry.get("similarity_score")
        similarity_score = float(similarity_score_raw) if similarity_score_raw is not None else math.nan
        publication_date = entry.get("publication_date") or ""
        meta = metadata_lookup(arxiv_id)
        return cls(
            arxiv_id=arxiv_id,
            title=meta.get("Title", ""),
            abstract=meta.get("Abstract", ""),
            embedding=embedding,
            similarity_score=similarity_score,
            publication_date=publication_date,
        )

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract,
            "embedding": list(self.embedding),
            "similarity_score": float(self.similarity_score),
            "publication_date": self.publication_date,
        }


class ArxivMetadataIndex:
    """Simple in-memory index mapping arXiv IDs to title/abstract metadata."""

    def __init__(self, dataset_roots: Iterable[str], cache_path: str | None = None) -> None:
        self._index: Dict[str, Dict[str, str]] = {}
        self._cache_path = cache_path
        self._roots: Sequence[str] = tuple(os.path.abspath(path) for path in dataset_roots)

        if cache_path and self._load_cache(cache_path):
            return

        for root in self._roots:
            self._load_split(root)
        logger.info(
            "Indexed %d unique arXiv IDs from %d dataset roots", len(self._index), len(self._roots)
        )
        if cache_path:
            self._save_cache(cache_path)

    def _load_split(self, root: str) -> None:
        if not os.path.exists(root):
            logger.warning("Skipping missing dataset split at %s", root)
            return
        dataset = load_from_disk(root)
        for record in dataset:
            arxiv_id = record.get("arXiv ID")
            if not arxiv_id or arxiv_id in self._index:
                continue
            self._index[arxiv_id] = {
                "Title": record.get("Title", ""),
                "Abstract": record.get("Abstract", ""),
            }

    def _load_cache(self, path: str) -> bool:
        try:
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
        except FileNotFoundError:
            return False

        cached_roots = tuple(payload.get("roots", ()))
        if cached_roots != self._roots:
            logger.info(
                "Metadata cache at %s does not match requested roots. Rebuilding index.",
                path,
            )
            return False

        self._index = payload.get("index", {})
        logger.info(
            "Loaded metadata index cache from %s with %d entries", path, len(self._index)
        )
        return True

    def _save_cache(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {"index": self._index, "roots": self._roots}
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)
        logger.info("Saved metadata index cache to %s", path)

    def get(self, arxiv_id: str) -> Dict[str, str]:
        return self._index.get(arxiv_id, {})

    def batch_lookup(self, arxiv_ids: Iterable[str]) -> Dict[str, Dict[str, str]]:
        return {arxiv_id: self.get(arxiv_id) for arxiv_id in arxiv_ids}

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._index)

    @property
    def roots(self) -> Sequence[str]:  # pragma: no cover - simple accessor
        return self._roots


__all__ = ["ArxivMetadataIndex", "SimilarPaper"]
