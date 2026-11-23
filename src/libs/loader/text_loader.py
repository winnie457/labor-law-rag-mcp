"""Plain text / Markdown Loader implementation.

This module implements loading for ``.txt`` and ``.md`` files by reading
them as UTF-8 text (``utf-8-sig`` to tolerate a leading BOM) and converting
them into the standardized ``Document`` structure.

Features:
- UTF-8 / UTF-8-BOM decoding
- doc_type differentiation (``text`` vs ``markdown``)
- Title extraction from the first Markdown heading
- SHA256-based document ID for traceability
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)

# Supported suffix → doc_type mapping
SUPPORTED_SUFFIXES: Dict[str, str] = {
    ".txt": "text",
    ".md": "markdown",
}


class TextLoader(BaseLoader):
    """Loader for plain text and Markdown files.

    Reads the file as UTF-8 (BOM-tolerant) and returns a Document whose
    ``text`` is the raw file content (Markdown is kept as-is so the
    Markdown-aware splitter works on real headings/paragraphs).

    Example:
        >>> loader = TextLoader()
        >>> doc = loader.load("notes/readme.md")
        >>> doc.metadata["doc_type"]
        'markdown'
    """

    def __init__(self, encoding: str = "utf-8-sig"):
        """Initialize Text Loader.

        Args:
            encoding: File encoding used for reading. Defaults to
                ``utf-8-sig`` so a leading BOM is stripped automatically.
        """
        self.encoding = encoding

    def load(self, file_path: str | Path) -> Document:
        """Load and parse a .txt or .md file.

        Args:
            file_path: Path to the text file.

        Returns:
            Document with raw text and metadata (source_path, doc_type,
            doc_hash, title).

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file suffix is not supported.
            RuntimeError: If the file cannot be decoded.
        """
        path = self._validate_file(file_path)

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"File is not a supported text format: {path}")

        doc_type = SUPPORTED_SUFFIXES[suffix]

        try:
            text_content = path.read_text(encoding=self.encoding)
        except UnicodeDecodeError as e:
            raise RuntimeError(
                f"Failed to decode {path} as {self.encoding}: {e}"
            ) from e

        doc_hash = self._compute_file_hash(path)
        doc_id = f"doc_{doc_hash[:16]}"

        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "doc_type": doc_type,
            "doc_hash": doc_hash,
        }

        title = self._extract_title(text_content, doc_type)
        if title:
            metadata["title"] = title

        logger.debug(f"Loaded {doc_type} file {path} ({len(text_content)} chars)")

        return Document(id=doc_id, text=text_content, metadata=metadata)

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute SHA256 hash of file content.

        Args:
            file_path: Path to file.

        Returns:
            Hex string of SHA256 hash.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                sha256.update(block)
        return sha256.hexdigest()

    @staticmethod
    def _extract_title(text: str, doc_type: str) -> Optional[str]:
        """Extract document title.

        For Markdown, the first ``# `` heading is preferred; otherwise the
        first non-empty line is used as a fallback.

        Args:
            text: File content.
            doc_type: ``text`` or ``markdown``.

        Returns:
            Title string if found, None otherwise.
        """
        lines = text.splitlines()

        if doc_type == "markdown":
            for line in lines[:20]:
                stripped = line.strip()
                if stripped.startswith("# "):
                    return stripped[2:].strip()

        for line in lines[:10]:
            stripped = line.strip()
            if stripped:
                return stripped

        return None
