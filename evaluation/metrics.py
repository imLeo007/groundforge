def recall_at_k(
    relevant_chunks: set[int],
    retrieved_chunks: list[int],
) -> float:
    if not relevant_chunks:
        return 0.0
    
    hits = retrieved_chunks.intersection(relevant_chunks)

    return len(hits) / len(relevant_chunks)


def reciprocal_rank(
    relevant_chunks: set[int],
    retrieved_chunks: list[int],
) -> float:
    if not relevant_chunks:
        return 0.0

    for rank, chunk_idx in enumerate(retrieved_chunks, start=1):
        if chunk_idx in relevant_chunks:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(reciprocal_ranks: list[float]) -> float:
    if not reciprocal_ranks:
        return 0.0

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def mean_recall_k(recall_k: list[float]) -> float:
    if not recall_k:
        return 0.0

    return sum(recall_k)/len(recall_k)