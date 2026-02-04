"""
Semantic search index using word embeddings (all-MiniLM-L6-v2).

Stores embeddings for string values and supports search-by-meaning:
returns top-k keys whose values are most similar to the query, above a threshold.

Requires: pip install sentence-transformers
"""

import threading
from typing import Dict, Any, Optional, List, Tuple

class EmbeddingUnavailableError(Exception):
    """Raised when semantic search is used but sentence-transformers is not installed."""
    pass


class EmbeddingIndex:
    """
    Index for search-by-meaning using the all-MiniLM-L6-v2 model.
    Stores embeddings for string values; supports top-k semantic search above a threshold.

    Requires: pip install sentence-transformers
    """

    def __init__(self):
        self._embeddings: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._model = None
        self._util = None
        self._unavailable_msg: Optional[str] = None  # set if import failed

    def _get_model(self):
        """Load model once and cache. Raises EmbeddingUnavailableError if lib not installed."""
        if self._unavailable_msg:
            raise EmbeddingUnavailableError(self._unavailable_msg)
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                from sentence_transformers import util as st_util
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
                self._util = st_util
            except ImportError as e:
                self._unavailable_msg = (
                    "Semantic search requires sentence-transformers. "
                    "Install with: pip install sentence-transformers"
                )
                raise EmbeddingUnavailableError(self._unavailable_msg) from e
        return self._model, self._util

    def index_key(self, key: str, value: Any, old_value: Any = None) -> None:
        """
        Add or update embedding for a key.
        Only string values are embedded; others are ignored.
        No-op if sentence-transformers is not available.
        """
        with self._lock:
            if old_value is not None and isinstance(old_value, str):
                self._embeddings.pop(key, None)
            if isinstance(value, str) and value.strip():
                try:
                    model, _ = self._get_model()
                    emb = model.encode(value, convert_to_numpy=True)
                    self._embeddings[key] = emb
                except (EmbeddingUnavailableError, Exception):
                    self._embeddings.pop(key, None)

    def unindex_key(self, key: str, value: Any) -> None:
        """Remove embedding for a key."""
        with self._lock:
            self._embeddings.pop(key, None)

    def semantic_search(
        self,
        query: str,
        k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Search by meaning: embed the query and return top-k keys whose value
        embeddings are most similar (cosine similarity), above the given threshold.

        Returns list of (key, score) sorted by score descending.
        Raises EmbeddingUnavailableError if sentence-transformers is not installed.
        """
        if not query or not query.strip():
            return []
        with self._lock:
            if not self._embeddings:
                return []
            model, util = self._get_model()
            query_emb = model.encode(query, convert_to_numpy=True)
            keys = list(self._embeddings.keys())
            value_embs = list(self._embeddings.values())
            scores = util.cos_sim(query_emb, value_embs)[0]
            results = [
                (keys[i], float(scores[i]))
                for i in range(len(keys))
                if float(scores[i]) >= threshold
            ]
            results.sort(key=lambda x: -x[1])
            return results[:k]

    def rebuild_from_data(self, data: Dict[str, Any]) -> None:
        """Rebuild embeddings from a data dict (e.g. after recovery). No-op if lib unavailable."""
        with self._lock:
            self._embeddings.clear()
            for key, value in data.items():
                if isinstance(value, str) and value.strip():
                    try:
                        model, _ = self._get_model()
                        emb = model.encode(value, convert_to_numpy=True)
                        self._embeddings[key] = emb
                    except (EmbeddingUnavailableError, Exception):
                        pass
