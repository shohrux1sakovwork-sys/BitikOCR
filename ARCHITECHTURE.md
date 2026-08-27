# Architecture

> Overview of the project structure, key modules, and design decisions.
> Read this before adding new code so you know **where things go**.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Module Responsibilities](#2-module-responsibilities)
3. [Dependency Flow](#3-dependency-flow)
4. [Design Principles](#4-design-principles)
5. [Key Decisions](#5-key-decisions)
6. [Adding New Code](#6-adding-new-code)

---

## 1. Project Structure

<!-- UPDATE THIS to match your actual project layout -->

```
.
├── CODING_STYLE.md          # Code style rules (must follow)
├── CONTRIBUTING.md          # How to contribute
├── ARCHITECTURE.md          # This file
├── AGENTS.md                # Auto-loaded rules for AI agents
├── Makefile                 # style, checks, lint, test commands
├── pyproject.toml           # Project config, dependencies, tool settings
├── uv.lock                  # Locked dependency versions (committed)
├── src/
│   └── <project>/
│       ├── __init__.py
│       ├── main.py           # Application entry point
│       ├── config.py         # Configuration loading
├── tests/
│   ├── conftest.py           # Shared pytest fixtures
│   ├── test_auth.py
│   └── test_loader.py
└── docs/
    └── conf.py               # Sphinx configuration
```

---

## 2. Module Responsibilities

<!-- UPDATE THIS to describe your actual modules -->

| Module        | Responsibility                                     | Depends on           |
|---------------|----------------------------------------------------|-----------------------|
| `main`        | Application entry point, CLI setup                 | `config`, `services`  |
| `config`      | Load and validate configuration from env/files     | (none)                |
| `models/`     | Data classes, schemas, type definitions             | (none)                |
| `services/`   | Core business logic                                | `models`, `db`        |
| `api/`        | HTTP routes, request/response handling             | `services`, `models`  |
| `db/`         | Database access, queries, repository pattern       | `models`              |
| `utils/`      | Pure helper functions, no business logic           | (none)                |


---

## 4. Design Principles

These principles guide how we structure code in this project:

### 1. Separation of Concerns
Each module has **one job**. API routes don't contain business logic. Services don't format HTTP responses. Database queries don't validate input.

### 2. Explicit over Implicit
- No magic globals or hidden state.
- Dependencies are passed explicitly (constructor injection or function arguments).
- Configuration is loaded once and passed down — not read from env vars deep in the code.

### 5. Configuration at the Edges
Configuration is loaded once at startup (in `config.py` or `main.py`) and passed into services/modules that need it. No module reads environment variables on its own.

---

## 5. Key Decisions

<!-- UPDATE THIS with your project's actual decisions -->

Document important architectural decisions here so future contributors (and AI agents) understand **why** things are the way they are.

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Package manager | `uv` | Fast, deterministic lockfile, replaces pip+virtualenv |
| Formatter | `black` | No config debates, one canonical style |
| Linter | `ruff` | Replaces flake8+isort checks, 10x faster |
| Type checker | `mypy` | Catches type errors before runtime |
| Test framework | `pytest` | Industry standard, better than unittest |
| Docstring format | reST / Sphinx | Auto-generates API docs |

<!-- Add rows for framework choices, DB choices, deployment strategy, etc. -->

---


### Checklist for new modules

- [ ] File is in the correct directory per the table above
- [ ] File and module names use `snake_case`
- [ ] Module has a docstring at the top explaining its purpose
- [ ] All public functions/classes have type annotations and docstrings
- [ ] Corresponding test file created in `tests/`
- [ ] No circular imports (follow the dependency flow)
