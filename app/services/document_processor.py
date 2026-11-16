"""
Document Processor — handles PDF, TXT, DOCX, MD ingestion.
"""

import logging
from pathlib import Path
import io

logger = logging.getLogger(__name__)


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Extract plain text from uploaded file.
    Supports: .txt, .md, .pdf, .docx
    """
    ext = Path(filename).suffix.lower()

    if ext in (".txt", ".md"):
        return file_bytes.decode("utf-8", errors="ignore")

    elif ext == ".pdf":
        return _extract_pdf(file_bytes)

    elif ext == ".docx":
        return _extract_docx(file_bytes)

    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .txt, .md, .pdf, .docx")


def _extract_pdf(file_bytes: bytes) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages)
        logger.info(f"Extracted {len(reader.pages)} pages from PDF")
        return text
    except ImportError:
        raise ImportError("pypdf not installed. Run: pip install pypdf")


def _extract_docx(file_bytes: bytes) -> str:
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        logger.info(f"Extracted {len(paragraphs)} paragraphs from DOCX")
        return "\n\n".join(paragraphs)
    except ImportError:
        raise ImportError("python-docx not installed. Run: pip install python-docx")
