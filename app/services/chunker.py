from docling.chunking import HybridChunker

from docling_core.types import DoclingDocument

from app.schemas.retriever import Chunk


hybrid_chunker = HybridChunker()


def get_chunk_page(chunk) -> int:
    for item in chunk.meta.doc_items:
        if item.prov:
            return item.prov[0].page_no

    return 1



def get_chunk_heading(chunk) -> str | None:
    headings = chunk.meta.headings

    if not headings:
        return None

    return " | ".join(headings)


def chunk_document(
    document: DoclingDocument
) -> list[Chunk]:

    grouped_chunks: list[Chunk] = []

    index = 0

    docling_chunks = list(
        hybrid_chunker.chunk(dl_doc=document)
    )

    for chunk in docling_chunks:

        page_number = get_chunk_page(chunk)

        heading = get_chunk_heading(chunk)

        grouped_chunks.append(
            {
                "text": chunk.text,
                "headings": heading,
                "page_no": page_number,
                "chunk_index": index
            }
        )

        index += 1

    return grouped_chunks