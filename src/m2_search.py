from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words (underthesea) rồi bỏ dấu "_" nối từ ghép.

    underthesea nối từ ghép bằng "_" (VD: "nghỉ_phép"). BM25 tokenize bằng split(" ")
    nên "nghỉ_phép" thành 1 token, còn query "nghỉ phép" thành 2 token -> không khớp.
    Vì vậy phải replace("_", " ") sau khi segment.
    """
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except Exception as e:
        print(f"  underthesea segmentation failed ({e}) - fallback raw text")
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = chunks
        self.corpus_tokens = [
            segment_vietnamese(c["text"]).lower().split()
            for c in chunks
        ]
        if not self.corpus_tokens:
            self.bm25 = None
            return
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []

        tokenized_query = segment_vietnamese(query).lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            SearchResult(
                text=self.documents[i]["text"],
                score=float(scores[i]),
                metadata=self.documents[i].get("metadata", {}),
                method="bm25",
            )
            for i in top_indices
            if scores[i] > 0
        ]


# bge-m3 chiếm ~1.5GB RAM → dùng chung 1 bản cho mọi DenseSearch instance
# (naive baseline + production pipeline chạy trong cùng process).
_ENCODER_CACHE: dict[str, object] = {}


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            if EMBEDDING_MODEL not in _ENCODER_CACHE:
                import torch
                from sentence_transformers import SentenceTransformer
                torch.set_num_threads(2)
                model = SentenceTransformer(EMBEDDING_MODEL)
                # bge-m3 mặc định max_seq_length=8192 → activation cực lớn khi encode
                # theo batch. Chunk của lab chỉ ~256-1000 ký tự nên 512 token là đủ,
                # đồng thời giữ RAM trong tầm ~1.5GB (máy lab 8GB).
                model.max_seq_length = min(getattr(model, "max_seq_length", 512), 512)
                _ENCODER_CACHE[EMBEDDING_MODEL] = model
            self._encoder = _ENCODER_CACHE[EMBEDDING_MODEL]
        return self._encoder

    def unload(self) -> None:
        """Giải phóng encoder khỏi RAM — gọi sau khi đã retrieve xong, trước khi load reranker."""
        import gc
        self._encoder = None
        _ENCODER_CACHE.pop(EMBEDDING_MODEL, None)
        gc.collect()

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PointStruct

        if not chunks:
            return

        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

        texts = [c["text"] for c in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=True, batch_size=4)

        points = [
            PointStruct(
                id=i,
                vector=vectors[i].tolist(),
                payload={**chunks[i].get("metadata", {}), "text": chunks[i]["text"]},
            )
            for i in range(len(chunks))
        ]
        # Upsert theo batch để tránh payload HTTP quá lớn
        for start in range(0, len(points), 128):
            self.client.upsert(collection_name=collection, points=points[start:start + 128])

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors.

        qdrant-client >= 1.10 dùng query_points(), KHÔNG phải search().
        """
        try:
            query_vector = self._get_encoder().encode(query).tolist()
            response = self.client.query_points(
                collection_name=collection, query=query_vector, limit=top_k, with_payload=True
            )
        except Exception as e:
            print(f"  Dense search failed: {e}")
            return []

        return [
            SearchResult(
                text=pt.payload.get("text", ""),
                score=float(pt.score),
                metadata=pt.payload or {},
                method="dense",
            )
            for pt in response.points
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = sum 1/(k + rank + 1).

    RRF chỉ dùng THỨ HẠNG nên không cần chuẩn hoá score giữa BM25 (không giới hạn)
    và cosine similarity (0..1).
    """
    rrf_scores: dict[str, dict] = {}

    for result_list in results_list:
        for rank, result in enumerate(result_list):
            entry = rrf_scores.setdefault(result.text, {"score": 0.0, "result": result})
            entry["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(rrf_scores.values(), key=lambda e: e["score"], reverse=True)[:top_k]

    return [
        SearchResult(
            text=e["result"].text,
            score=float(e["score"]),
            metadata=e["result"].metadata,
            method="hybrid",
        )
        for e in ranked
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
