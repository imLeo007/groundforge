from functools import lru_cache

from typing import Any

from sentence_transformers import CrossEncoder

import torch

from app.core.config import settings



RetrievedChunks = dict[str, Any]



@lru_cache
def load_reranker() -> CrossEncoder:

    model = CrossEncoder(
        settings.reranker_model_name,
        backend="onnx",
        activation_fn=torch.nn.Sigmoid(),
        model_kwargs={
            "file_name": settings.onnx_filename,
            "provider": "CPUExecutionProvider",
        },
    )

    return model



def rerank_chunks(
    question: str,
    chunks: list[RetrievedChunks],
    top_k: int
) -> list[RetrievedChunks]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if not chunks:
        return []

    reranker = load_reranker()

    question_chunk_pairs = [
        (question, str(chunk["text"]))
        for chunk in chunks
    ]

    rerank_scores = reranker.predict(question_chunk_pairs, show_progress_bar=False)

    ranked_chunks: list[RetrievedChunks] = []

    for chunk, score in zip(chunks, rerank_scores):
        ranked_chunk = chunk.copy()

        ranked_chunk["rerank_score"] = float(score)

        ranked_chunks.append(ranked_chunk)

    ranked_chunks.sort(key=lambda chunk: float(chunk["rerank_score"]), reverse=True)

    return ranked_chunks[:top_k]