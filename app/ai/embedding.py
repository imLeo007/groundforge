import torch

from sentence_transformers import SentenceTransformer

from app.core.config import settings

torch.set_num_threads(2)

embedding_model = SentenceTransformer(settings.embedding_model, device="cpu")


def create_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    with torch.inference_mode():
        embeddings = embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    return embeddings.tolist()


def create_question_embeddings(question: str) -> list[float]:
    with torch.inference_mode():
        embedding = embedding_model.encode(
            question,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    return embedding.tolist()