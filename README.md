# medical-rag-platform
Development of an intelligent system for extracting, structuring, and querying medical reports using RAG architectures.

## Backend Contract Test

- Local minimal environment can skip backend contract tests if backend dependencies are missing.
- Full backend environment should run:
  - `make test-backend-contract`
- In full CI, backend contract tests must be required (non-optional).

