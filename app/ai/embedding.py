from app.core.config import settings

from sentence_transformers import SentenceTransformer


embedding_model = SentenceTransformer(
    settings.embedding_model,
    backend="onnx",
    model_kwargs={
            "file_name": settings.onnx_filename,
            "provider": "CPUExecutionProvider",
        }
    )


def create_embeddings(text: list[str]) -> list[list[float]]:
    if not text:
        return []

    embeddings = embedding_model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings.tolist()


def create_question_embeddings(question: str) -> list[float]:
    embeddings = embedding_model.encode(
        question,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings.tolist()