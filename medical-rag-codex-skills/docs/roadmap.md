# Roadmap

## Phase 1. Repository Foundation

- Finalize architecture documents and skill boundaries.
- Define extraction rules, record schema, retrieval policy, and evaluation criteria.
- Prepare `sample-data/` conventions without committing sensitive files.

## Phase 2. Small-Scale Document Experiments

- Collect a few representative medical laboratory PDFs with permission.
- Compare native text extraction, table extraction, and OCR fallback quality.
- Record common layout patterns and extraction failure modes.

## Phase 3. Structured Output Design

- Stabilize the normalized JSON schema for lab records.
- Decide how to represent source excerpts, page references, and extraction confidence.
- Define chunking rules for indexed evidence units.

## Phase 4. Hybrid Retrieval Prototype

- Build a small offline prototype for keyword search and vector search.
- Test result fusion and reranking on a limited evaluation set.
- Measure citation quality and abstention behavior.

## Phase 5. Evaluation Pack

- Create realistic evaluation queries.
- Annotate expected supporting evidence.
- Track retrieval relevance, grounding quality, completeness, and hallucination resistance.

## Out of Scope for This Repository

- Production deployment.
- API or web interface development.
- Infrastructure orchestration.
- Clinical decision support claims.
