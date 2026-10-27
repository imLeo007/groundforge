from fastapi import UploadFile

import shutil

import os

from docling_core.types import DoclingDocument

from docling.datamodel.base_models import InputFormat

from docling.datamodel.pipeline_options import PdfPipelineOptions, PdfBackend

from docling.document_converter import DocumentConverter, PdfFormatOption


pipeline_options = PdfPipelineOptions(
    do_ocr=False,
    do_table_structure=False,
    generate_page_images=False,
    generate_picture_images=False,
    pdf_backend=PdfBackend.PYPDFIUM2,
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options,)
    },
)


def save_pdf(file: UploadFile, upload_dir: str) -> str:
    os.makedirs(upload_dir, exist_ok=True)

    if file.filename is None:
        raise ValueError("uploaded file must have a name")

    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return file_path


def warm_docling(file_path: str) -> None:
    converter.convert(file_path)


def extract_document(file_path: str) -> DoclingDocument:

    result = converter.convert(file_path)

    return result.document