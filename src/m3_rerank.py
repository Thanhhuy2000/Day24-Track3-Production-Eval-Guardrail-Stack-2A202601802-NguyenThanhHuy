from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


# Cache model theo tên: bge-reranker-v2-m3 chiếm ~1.5GB RAM, mỗi instance load
# lại một bản sao sẽ làm tràn RAM (access violation trên máy 8GB).
_MODEL_CACHE: dict[str, object] = {}


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy-load cross-encoder (dùng chung 1 bản model cho mọi instance).

        Dùng sentence_transformers.CrossEncoder, KHÔNG dùng FlagEmbedding:
        FlagReranker crash với transformers>=5.0 (XLMRobertaTokenizer lỗi).
        """
        if self._model is None:
            if self.model_name not in _MODEL_CACHE:
                from sentence_transformers import CrossEncoder
                _MODEL_CACHE[self.model_name] = CrossEncoder(self.model_name)
            self._model = _MODEL_CACHE[self.model_name]
        return self._model

    def unload(self) -> None:
        """Giải phóng model khỏi RAM (dùng khi cần load model khác trên máy ít RAM)."""
        import gc
        self._model = None
        _MODEL_CACHE.pop(self.model_name, None)
        gc.collect()

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 -> top-k."""
        if not documents:
            return []

        try:
            model = self._load_model()
            pairs = [(query, doc["text"]) for doc in documents]
            scores = model.predict(pairs)
        except Exception as e:
            print(f"  Rerank failed ({e}) - giữ nguyên thứ tự hybrid search")
            return [
                RerankResult(
                    text=doc["text"],
                    original_score=float(doc.get("score", 0.0)),
                    rerank_score=float(doc.get("score", 0.0)),
                    metadata=doc.get("metadata", {}),
                    rank=i,
                )
                for i, doc in enumerate(documents[:top_k])
            ]

        if isinstance(scores, (int, float)):
            scores = [scores]

        scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)

        return [
            RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []
        try:
            from flashrank import Ranker, RerankRequest
            if self._model is None:
                self._model = Ranker()
            passages = [{"id": i, "text": d["text"], "meta": d.get("metadata", {})}
                        for i, d in enumerate(documents)]
            results = self._model.rerank(RerankRequest(query=query, passages=passages))
        except Exception as e:
            print(f"  Flashrank failed: {e}")
            return []

        return [
            RerankResult(
                text=r["text"],
                original_score=float(documents[r["id"]].get("score", 0.0)),
                rerank_score=float(r["score"]),
                metadata=r.get("meta", {}),
                rank=i,
            )
            for i, r in enumerate(results[:top_k])
        ]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
