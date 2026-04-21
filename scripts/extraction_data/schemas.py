from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ImageAsset:
    image_id: str
    page_number: int
    file_path: str
    width: int
    height: int
    ext: str
    bbox: dict[str, float] | None = None
    xref: int | None = None
    page_coverage: float | None = None
    image_type: str = "unknown"
    role: str = "unknown"
    is_indexable: bool = False
    context_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TableAsset:
    table_id: str
    page_number: int
    row_count: int
    column_count: int
    csv_path: str
    json_path: str
    columns: list[str] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    preview: list[dict[str, Any]] = field(default_factory=list)
    bbox: dict[str, float] | None = None
    table_role: str = "unknown"
    is_indexable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentBlock:
    block_id: str
    page_number: int
    block_type: str
    section_title: str
    text: str
    bbox: dict[str, float] | None
    source_table_ids: list[str] = field(default_factory=list)
    source_image_ids: list[str] = field(default_factory=list)
    source_text_block_ids: list[str] = field(default_factory=list)
    is_indexable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OcrAsset:
    page_number: int
    text: str
    text_path: str | None
    image_path: str | None
    used: bool
    engine: str | None = None
    blocks: list[dict[str, Any]] = field(default_factory=list)
    words: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PageData:
    page_number: int
    width: float
    height: float
    native_text: str
    final_text: str
    native_text_chars: int
    ocr_used: bool
    ocr_text_chars: int
    table_ids: list[str] = field(default_factory=list)
    image_ids: list[str] = field(default_factory=list)
    blocks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentData:
    doc_id: str
    source_pdf: str
    output_dir: str
    pdf_type: str
    document_type: str
    page_count: int
    native_text_available: bool
    ocr_available: bool
    extraction_warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    facility: dict[str, Any] = field(default_factory=dict)
    patient: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)
    interpretation: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    pages: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    blocks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
