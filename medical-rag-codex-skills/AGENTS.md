# AGENTS.md

## Purpose

Use this repository as a **skills-first architectural workspace** for a medical multimodal RAG graduation project. Keep every change aligned with the scope of medical laboratory PDF processing, structured extraction, hybrid retrieval, grounded answering, and evaluation.

## Working Principles

- Keep changes minimal, reviewable, and easy to verify.
- Prefer adding or refining markdown, schemas, and decision records before adding code.
- Preserve the repository as a documentation and skills scaffold unless the user explicitly requests implementation work.
- Stay within the medical PDF RAG scope defined in the root README and `docs/architecture.md`.
- Write folder and file names in English.
- Keep content professional, realistic, and suitable for a student PFE context.

## Scope Guardrails

- Do not add a backend, API server, frontend, Docker setup, database deployment, or infrastructure automation unless explicitly requested.
- Do not add fake pipelines, mock production services, or placeholder code that suggests features already exist.
- Do not introduce heavy framework choices prematurely.
- Do not dilute the skills-first structure by scattering undocumented notes outside the defined repository layout.

## Repository Conventions

- Treat each skill as an operational instruction layer tied to a specific stage of the target architecture.
- Keep `SKILL.md` files concise, imperative, and trigger-oriented.
- Keep reference files practical and reusable; prefer checklists, schemas, edge cases, and policies over theory.
- Keep agent metadata small and human-readable.
- When adding examples, make them realistic and medically plausible without implying clinical validation.

## Editing Rules

- Prefer markdown and structured schemas over code when both would satisfy the request.
- Extend existing documents instead of creating parallel duplicates.
- Preserve traceability between extraction, structuring, retrieval, answer generation, and evaluation.
- Use citations, page references, and source excerpts whenever examples involve extracted evidence.

## Definition of Done

A repository change is acceptable when it is:

- clearly within scope,
- consistent with the target architecture,
- easy for a supervisor to review,
- and directly reusable in future Codex-assisted iterations.



## Safety Rules

- Never run destructive commands such as `rm -rf`, force-delete operations, disk formatting commands, or commands that target system directories.
- Never modify files outside the current repository.
- Ask for confirmation before deleting, renaming, moving, or overwriting multiple files.
- Prefer safe, reversible edits over broad refactors.
- Do not install or configure system-level services unless explicitly requested.


## Grounding and Reliability

- Do not invent medical values, extracted content, evaluation results, or implemented features.
- When writing examples, clearly distinguish between illustrative examples and real extracted evidence.
- Preserve groundedness across the repository: extraction rules, schemas, retrieval logic, answer generation, and evaluation must stay consistent.
- Prefer abstention, uncertainty notes, or explicit limitations over unsupported claims.
- Keep all future repository changes aligned with the target multimodal and hybrid-retrieval architecture.



## Code Quality Standards

- Write production-style Python that is readable, explicit, and maintainable.
- Prefer simple, well-structured modules over clever or overly compact code.
- Use clear function and variable names that reflect medical RAG concepts precisely.
- Add type hints to public functions, core data structures, and important interfaces.
- Write short docstrings for non-trivial functions and modules.
- Keep functions focused on a single responsibility.
- Avoid deeply nested logic when a helper function would make the code clearer.
- Prefer small reusable utilities over duplicated logic.
- Validate inputs and handle errors explicitly.
- Do not silently swallow exceptions.
- Use structured return formats for extraction, normalization, retrieval, and evaluation steps.
- Keep logging informative and minimal; avoid noisy debug prints in committed code.
- Follow a consistent project style compatible with black, ruff, and modern Python conventions.
- When proposing classes, use them only if they clearly improve structure; do not introduce unnecessary abstraction.
- Keep comments useful and technical; do not comment obvious lines.
- When writing RAG-related code, preserve traceability between source document, extracted content, chunk, retrieval result, and final answer.
- Prefer grounded, testable logic over speculative heuristics.


## Python Rules for Medical RAG Work

- Keep extraction, structuring, retrieval, and evaluation concerns separated.
- Represent lab results with explicit schemas or typed models.
- Preserve source metadata such as page number, table origin, and source excerpt.
- Never hardcode fake clinical conclusions or simulated model outputs as if they were real results.
- Make normalization rules explicit and reviewable.
- Prefer deterministic preprocessing steps before LLM-based reasoning when possible.
- Keep citation-related fields first-class in the data flow.
- Make ambiguity visible instead of hiding it.


## Verification and Testing

- When adding Python code, keep it easy to run and easy to review.
- Prefer small, focused modules with basic usage examples or lightweight tests when appropriate.
- Run formatting, linting, and tests before considering implementation work complete.
- Do not mark work as finished if core paths are unverified.
- Prefer explicit failure over silent success when extraction, normalization, or retrieval steps are uncertain.


## Data Privacy and Sample Data

- Never commit raw medical PDFs containing directly identifiable patient information.
- Prefer anonymized, synthetic, or explicitly sanitized examples in documentation and sample data.
- Treat personal health information as sensitive by default.
- When creating examples, avoid real names, IDs, addresses, dates of birth, or report numbers.
- Do not present repository artifacts as clinically validated or suitable for real medical decision-making.