# Target Architecture

## Objective

This document frames the target architecture for a future multimodal and hybrid-retrieval RAG pipeline over medical laboratory PDF reports. It describes the intended flow, responsibilities of each stage, and design constraints for a realistic student project.

## End-to-End Flow

`Medical PDF Reports -> Ingestion Layer -> Multimodal Extraction -> Clinical Structuring -> Chunking -> Anonymization -> Indexing Layer -> keyword store + vector store + object store + metadata store -> Query Processing Service -> Query Embedding -> Query Routing -> Vector Search + Keyword Search -> Result Fusion -> Reranker -> Context Augmentation -> LLM Generation -> Output Guardrails -> Final Answer`

## Stage Responsibilities

### 1. Medical PDF Reports

Input documents may contain typed text, scanned sections, machine-generated tables, logos, charts, footers, and mixed page layouts.

### 2. Ingestion Layer

- Register document metadata.
- Assign a document identifier.
- Track page boundaries and source provenance.
- Prepare the PDF for downstream extraction without losing layout information.

### 3. Multimodal Extraction

- Extract text blocks in reading order.
- Detect tables and preserve row and column relationships.
- Identify image or graph regions as explicit placeholders with coordinates or page references.
- Use OCR only when native text extraction is incomplete or unavailable.

### 4. Clinical Structuring

- Normalize analyte names, values, units, and reference ranges.
- Preserve source excerpts and page references.
- Keep uncertain fields explicit rather than inferred silently.

### 5. Chunking

- Build chunks that remain medically coherent.
- Keep table rows and nearby interpretation context together when relevant.
- Avoid splitting a result from its unit, abnormal flag, or reference range.

### 6. Anonymization

- Remove or mask personally identifying data before indexing.
- Preserve document traceability through synthetic document identifiers.

### 7. Indexing Layer

Use multiple stores with distinct responsibilities:

- `keyword store` for exact lookup of analytes, units, abbreviations, and uncommon terms.
- `vector store` for semantic retrieval over normalized chunks and report passages.
- `object store` for raw PDFs, page crops, extracted artifacts, or intermediate outputs.
- `metadata store` for document-level and chunk-level provenance, page mapping, and extraction status.

### 8. Query Processing Service

- Normalize the user question.
- Detect whether the question is analytical, factual, document-specific, or cross-document.
- Decide whether the query benefits more from keyword matching, semantic matching, or both.

### 9. Retrieval and Fusion

- Generate query embeddings when semantic retrieval is useful.
- Run vector search and keyword search in parallel when appropriate.
- Fuse results with a transparent ranking policy.
- Apply reranking using evidence quality, source proximity, and document relevance.

### 10. Context Augmentation and Generation

- Assemble a grounded context window from top-ranked evidence.
- Keep page references and source excerpts attached to each evidence item.
- Instruct the LLM to answer only from retrieved evidence.

### 11. Output Guardrails

- Require source citations.
- Abstain when evidence is weak, contradictory, or incomplete.
- Avoid unsupported clinical interpretation.

## Design Principles

- Prefer traceability over aggressive normalization.
- Prefer explicit uncertainty over hidden assumptions.
- Keep retrieval hybrid by default for laboratory reports.
- Make evaluation possible from the start by preserving page-level provenance.
- Keep the first implementation academically credible and technically modest.
