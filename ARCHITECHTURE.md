# Architecture

> Overview of the project structure, key modules, and design decisions.
> Read this before adding new code so you know **where things go**.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Module Responsibilities](#2-module-responsibilities)
3. [Design Principles](#3-design-principles)
4. [Key Decisions](#4-key-decisions)
5. [Adding New Code](#5-adding-new-code)

---

## 1. Project Structure

```
.
├── CODING_STYLE.md          # Code style rules (must follow)
├── CLAUDE.md                # Auto-loaded rules for Claude AI
├── GEMINI.md                # Auto-loaded rules for Gemini AI
├── ARCHITECHTURE.md         # This file
├── pyproject.toml           # Project config, dependencies, tool settings
├── uv.lock                  # Locked dependency versions (committed)
├── README.md                # Project documentation
└── bitikocr/                # Main package directory
    ├── __init__.py          # Package initialization
    └── py.typed             # Type hinting marker
```

---

## 2. Module Responsibilities

| Module        | Responsibility                                     | Depends on           |
|---------------|----------------------------------------------------|-----------------------|
| `cli`         | Application entry point, CLI setup                 | `config`, `core`      |
| `config`      | Load and validate configuration from env/files     | (none)                |
| `models/`     | Data classes, schemas, type definitions            | (none)                |
| `core/`       | Core OCR business logic and image processing       | `models`, `utils`     |
| `utils/`      | Pure helper functions, no business logic           | (none)                |


---

## 3. Design Principles

These principles guide how we structure code in this project:

### 1. Separation of Concerns
Each module has **one job**. API routes don't contain business logic. Services don't format HTTP responses. Database queries don't validate input.

### 2. Explicit over Implicit
- No magic globals or hidden state.
- Dependencies are passed explicitly (constructor injection or function arguments).
- Configuration is loaded once and passed down — not read from env vars deep in the code.

### 3. Configuration at the Edges
Configuration is loaded once at startup (in `config.py` or `cli.py`) and passed into services/modules that need it. No module reads environment variables on its own.

---

## 4. Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Package manager | `uv` | Fast, deterministic lockfile, replaces pip+virtualenv |
| Formatter | `black` | No config debates, one canonical style |
| Import Sorter | `isort` | Groups and sorts imports (profile=black) |
| Linter | `ruff` / `flake8` | Linting standard |
| Type checker | `mypy` | Catches type errors before runtime |
| Test framework | `pytest` | Industry standard, better than unittest |
| Docstring format | Google-style | Clear, readable format for APIs |

---


## 5. Adding New Code

### Checklist for new modules

- [ ] File is in the correct directory per the table above
- [ ] File and module names use `snake_case`
- [ ] Module has a docstring at the top explaining its purpose
- [ ] All public functions/classes have type annotations and docstrings
- [ ] Corresponding test file created in `tests/`
- [ ] No circular imports (follow the dependency flow)
