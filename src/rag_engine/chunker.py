"""Hierarchical document chunking with density analysis."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from .config import ChunkerConfig


@dataclass
class Chunk:
    id: str
    text: str
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    level: int = 0
    char_offset: int = 0


class HierarchicalChunker:
    """Splits documents into hierarchical chunks preserving parent-child structure.

    Uses information density analysis to determine split points, preferring
    natural boundaries (paragraphs, sections) over fixed-size windows.
    """

    # catches markdown headings + ALL-CAPS-ish section titles in legal docs
    HEADING_PATTERN = re.compile(r"^#{1,6}\s+.+$|^[A-Z][A-Za-z\s]{3,60}$", re.MULTILINE)
    # naive sentence split but handles 95% of english/polish text correctly
    # tried spacy sentencizer - 40x slower for 2% improvement, nope
    SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    def __init__(self, config: ChunkerConfig) -> None:
        self._config = config

    def chunk_document(self, text: str, doc_id: str | None = None) -> list[Chunk]:
        doc_id = doc_id or str(uuid.uuid4())

        parent_chunks = self._create_parent_chunks(text, doc_id)
        all_chunks: list[Chunk] = []

        for parent in parent_chunks:
            child_chunks = self._split_into_children(parent)
            all_chunks.append(parent)
            all_chunks.extend(child_chunks)

        return all_chunks

    def _create_parent_chunks(self, text: str, doc_id: str) -> list[Chunk]:
        sections = self._split_by_sections(text)
        parents = []
        offset = 0

        for section in sections:
            if len(section) <= self._config.parent_max_size:
                chunk = Chunk(
                    id=f"{doc_id}:p:{len(parents)}",
                    text=section,
                    level=0,
                    char_offset=offset,
                    metadata={"doc_id": doc_id, "type": "parent"},
                )
                parents.append(chunk)
            else:
                sub_sections = self._split_by_size(
                    section, self._config.parent_max_size, overlap=0
                )
                for sub in sub_sections:
                    chunk = Chunk(
                        id=f"{doc_id}:p:{len(parents)}",
                        text=sub,
                        level=0,
                        char_offset=offset,
                        metadata={"doc_id": doc_id, "type": "parent"},
                    )
                    parents.append(chunk)
                    offset += len(sub)
                continue

            offset += len(section)

        return parents

    def _split_into_children(self, parent: Chunk) -> list[Chunk]:
        text = parent.text
        children = []

        if self._config.sentence_boundary:
            segments = self._split_at_sentences(text, self._config.max_chunk_size)
        else:
            segments = self._split_by_size(
                text, self._config.max_chunk_size, self._config.overlap
            )

        for i, segment in enumerate(segments):
            density = self._compute_density(segment)
            if density < self._config.min_density and len(segment.strip()) < 20:
                continue

            child_id = f"{parent.id}:c:{i}"
            child = Chunk(
                id=child_id,
                text=segment,
                parent_id=parent.id,
                level=1,
                char_offset=parent.char_offset,
                metadata={
                    "doc_id": parent.metadata.get("doc_id"),
                    "type": "child",
                    "density": round(density, 3),
                },
            )
            children.append(child)
            parent.children.append(child_id)

        return children

    def _split_by_sections(self, text: str) -> list[str]:
        headings = list(self.HEADING_PATTERN.finditer(text))

        if not headings:
            return [text]

        sections = []
        for i, match in enumerate(headings):
            start = match.start()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            section = text[start:end].strip()
            if section:
                sections.append(section)

        if headings and headings[0].start() > 0:
            preamble = text[: headings[0].start()].strip()
            if preamble:
                sections.insert(0, preamble)

        return sections

    def _split_at_sentences(self, text: str, max_size: int) -> list[str]:
        sentences = self.SENTENCE_BOUNDARY.split(text)
        chunks = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) > max_size and current:
                chunks.append(current.strip())
                overlap_text = current[-self._config.overlap :] if self._config.overlap else ""
                current = overlap_text + sentence
            else:
                current += (" " if current else "") + sentence

        if current.strip():
            chunks.append(current.strip())

        return chunks

    @staticmethod
    def _split_by_size(text: str, max_size: int, overlap: int) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_size, len(text))
            chunks.append(text[start:end])
            start = end - overlap if overlap else end
        return chunks

    @staticmethod
    def _compute_density(text: str) -> float:
        """Information density heuristic - filters out whitespace-heavy chunks
        (tables of contents, blank sections in PDFs) that tank retrieval precision.
        The 0.4/0.6 weights were tuned on our legal corpus - YMMV on other domains."""
        if not text:
            return 0.0

        total = len(text)
        content = len(text.strip())
        whitespace_ratio = (total - content) / total if total > 0 else 0

        words = text.split()
        unique_words = set(w.lower() for w in words)
        lexical_diversity = len(unique_words) / len(words) if words else 0

        return (1 - whitespace_ratio) * 0.4 + lexical_diversity * 0.6
