from functools import lru_cache
from typing import Any
import torch
from sentence_transformers import CrossEncoder
from app.core.config import settings
from app.schemas.retriever import RetrievedChunks


@lru_cache
def load_reranker() -> CrossEncoder:
    reranker = CrossEncoder(settings.reranker_model, device="cpu")

    return reranker


def rerank_chunks(
    question: str,
    chunks: list[RetrievedChunks],
    top_k: int,
) -> list[RetrievedChunks]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    
    if not chunks:
        return []

    reranker = load_reranker()
    
    question_chunk_pairs = [(question, str(chunk["text"])) for chunk in chunks]

    with torch.inference_mode():
        raw_scores = reranker.predict(
            question_chunk_pairs,
            show_progress_bar=False,
            apply_softmax=False,
        )
    
        probabilities = torch.sigmoid(torch.tensor(raw_scores)).tolist()

    ranked_chunks: list[RetrievedChunks] = []
    
    for chunk, score in zip(chunks, probabilities):
        ranked_chunk = chunk.copy()
    
        ranked_chunk["rerank_score"] = float(score)
    
        ranked_chunks.append(ranked_chunk)

    ranked_chunks.sort(key=lambda chunk: float(chunk["rerank_score"]), reverse=True)
    
    return ranked_chunks[:top_k]