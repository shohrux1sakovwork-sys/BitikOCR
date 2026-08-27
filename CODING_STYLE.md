# Coding Style Guide

> This document defines the coding style and conventions for this project.
> All team members **and AI agents** must follow these rules to keep the codebase consistent.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Environment — uv](#2-environment--uv)
3. [Tooling & Automation](#3-tooling--automation)
4. [Code Formatting](#4-code-formatting)
5. [Naming Conventions](#5-naming-conventions)
6. [Type Annotations](#6-type-annotations)
7. [Imports](#7-imports)
8. [Docstrings](#8-docstrings)
9. [Comments](#9-comments)
10. [Error Handling](#10-error-handling)
11. [What to Avoid](#11-what-to-avoid)
12. [Git Commit Messages](#12-git-commit-messages)

---

## 1. Quick Start

```bash
# Set up environment
uv sync

# Auto-format code
make style

# Run all checks at once (formatting + lint + types)
make checks

# Individual checks
make style-check   # isort + black
make lint-check    # ruff
make type-check    # mypy
```

> **Before every pull request**, run `make checks` and fix all reported issues.

---

## 2. Environment — uv

We use **[uv](https://docs.astral.sh/uv/)** as our Python package and environment manager. Do **not** use `pip`, `conda`, `poetry`, or `virtualenv` directly.

### Why uv

- 10–100x faster than pip
- Deterministic lockfile (`uv.lock`)
- Manages Python versions, virtual environments, and dependencies in one tool
- Drop-in replacement for pip and virtualenv workflows

### Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the project and sync dependencies
git clone <repo-url> && cd <project>
uv sync
```

`uv sync` reads `pyproject.toml`, resolves dependencies, creates a `.venv` if needed, and installs everything — one command, done.

### Daily Usage

```bash
# Add a production dependency
uv add requests

# Add a dev dependency
uv add --dev pytest

# Remove a dependency
uv remove requests

# Run a command inside the managed environment
uv run python main.py
uv run pytest

# Update the lockfile after manual pyproject.toml edits
uv lock
```

### Rules

- **Always commit `uv.lock`** to version control. This ensures every team member and CI gets identical dependency versions.
- **Never commit `.venv/`** — add it to `.gitignore`.
- **Never run `pip install` manually.** Use `uv add` to add packages and `uv sync` to install.
- **Pin Python version** in `pyproject.toml`:
  ```toml
  [project]
  requires-python = ">=3.11"
  ```
- When adding dependencies, let `uv` handle version resolution. Do not manually pin versions in `pyproject.toml` unless you have a specific reason.

### CI / New Machine Bootstrap

```bash
# One-liner: install uv + sync all dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh && uv sync
```

---

## 3. Tooling & Automation

| Tool    | Purpose                          | Config             |
|---------|----------------------------------|---------------------|
| `black` | Code formatting                  | `pyproject.toml`    |
| `isort` | Import sorting (profile=black)   | `pyproject.toml`    |
| `ruff`  | Linting (fast, replaces flake8)  | `pyproject.toml`    |
| `mypy`  | Static type checking             | `pyproject.toml`    |

All tools are installed as dev dependencies via `uv add --dev`.

### Ruff Ignore Rules

| Rule  | Reason                                              |
|-------|-----------------------------------------------------|
| F403  | `import *` allowed in specific cases                |
| F405  | Names from `import *` may be undefined              |
| E501  | Line length enforced by `black` instead             |
| F401  | Unused imports ignored **only** in `__init__.py`    |

---

## 4. Code Formatting

- **Line length:** `80` characters
- **Formatter:** `black` (non-negotiable — no manual style debates)
- **Import sorter:** `isort` with `profile = "black"`
- **Indentation:** 4 spaces — never tabs
- **Trailing commas:** Always use in multi-line collections (enforced by `black`)

Let the tools format the code. Do not fight the formatter.

---

## 5. Naming Conventions

| Kind                  | Convention            | Example                        |
|-----------------------|-----------------------|--------------------------------|
| Variables             | `snake_case`          | `user_count`, `file_path`      |
| Functions / Methods   | `snake_case`          | `get_user()`, `parse_config()` |
| Classes               | `PascalCase`          | `UserManager`, `DataLoader`    |
| Constants             | `UPPER_SNAKE_CASE`    | `MAX_RETRIES`, `DEFAULT_PORT`  |
| Modules / Files       | `snake_case`          | `data_loader.py`, `utils.py`   |
| Packages / Dirs       | `snake_case`          | `data_utils/`, `api_client/`   |
| Private attributes    | `_leading_underscore` | `_cache`, `_internal_client`   |
| Type aliases          | `PascalCase`          | `UserId = int`                 |

**Rules:**

- Names must be **descriptive**. Avoid single-letter names except in list comprehensions or math contexts (`i`, `x`, `y`).
- Booleans should read as yes/no questions: `is_valid`, `has_permissions`, `can_retry`.
- Avoid abbreviations unless universally understood (`url`, `id`, `cfg`, `db`).
- Never use `l`, `O`, or `I` as single-character variable names (too similar to `1` and `0`).

---

## 6. Type Annotations

All public functions and methods **must** have type annotations on both parameters and return values.

```python
# ✅ Good
def get_user(user_id: int) -> User:
    ...

def process_items(items: list[str], limit: int = 10) -> dict[str, int]:
    ...

# ❌ Bad — missing annotations
def get_user(user_id):
    ...
```

**Rules:**

- Use built-in generics: `list[str]`, `dict[str, int]`, `tuple[int, ...]` — not `typing.List`, `typing.Dict` (Python 3.9+).
- Use `X | Y` union syntax — not `typing.Union[X, Y]` (Python 3.10+).
- Use `X | None` — not `typing.Optional[X]`.
- Mypy config: `ignore_missing_imports = true` — missing stubs are acceptable for third-party libs.
- Private helper functions should also have annotations when the logic is non-trivial.

---

## 7. Imports

Imports are sorted and grouped automatically by `isort` (profile=black). The order is:

1. Standard library
2. Third-party packages
3. Local / project imports

```python
import os
import sys
from pathlib import Path

import numpy as np
import torch

from myproject.models import User
from myproject.utils import helper
```

**Rules:**

- Use **absolute imports** for project code. Avoid relative imports (`from . import ...`) unless inside a package's own `__init__.py`.
- Never use `import *` in application code — only in `__init__.py` re-exports, and only when intentional.
- A blank line separates each import group (handled by `isort`).

---

## 8. Docstrings

Docstrings are **required** on all public classes, methods, and functions. They are used by **Sphinx** (via the `napoleon` extension) to auto-generate API documentation.

### Syntax

We use **Google-style** docstrings. Parameters are listed under `Args:` with their name and description — clean and readable.

```python
def read_file(path: str) -> str:
    """Read a file from disk.

    Args:
        path: The path to the file.

    Returns:
        The contents of the file as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    ...
```

### ❌ Do NOT use reST-style

```python
# ❌ Bad — verbose, hard to read
def read_file(path: str) -> str:
    """
    :param path: The path to the file.
    :returns: The contents of the file.
    :raises FileNotFoundError: If the file doesn't exist.
    """
    ...
```

### ✅ More examples

```python
def get_users(
    is_active: bool,
    user_count: int,
    role: str | None = None,
) -> list[User]:
    """Get a filtered list of users.

    Args:
        is_active: Whether to return only active users.
        user_count: Maximum number of users to return.
        role: Filter by role name. Returns all roles if None.

    Returns:
        A list of users matching the given filters.

    Raises:
        ValueError: If user_count is negative.
        DatabaseError: If the connection to the database fails.
    """
    ...
```

### Class Docstrings

```python
class DataLoader:
    """Load and preprocess data from a given source.

    Args:
        source: Path or URL to the data source.
        batch_size: Number of samples per batch.
    """

    def __init__(self, source: str, batch_size: int = 32) -> None:
        ...
```

### Boolean Parameters

Boolean parameters should be self-explanatory through naming:

```python
# ✅ Good — the name tells you what True/False means
def fetch_records(is_cached: bool, has_header: bool) -> list[Record]:
    """Fetch records from the data source.

    Args:
        is_cached: Whether to use the cached version.
        has_header: Whether the source file has a header row.

    Returns:
        A list of parsed records.
    """
    ...
```

### Cross-Document Links

In docstrings, you can link to other documented objects:

```rst
:class:`foo.Foo`    ← links to class Foo in module foo
:mod:`foo`          ← links to module foo
:func:`foo.bar`     ← links to function bar in module foo
:attr:`foo.Foo.x`   ← links to attribute x on class Foo
```

### Rules

- First line: **short one-line summary** in imperative mood ("Read a file", not "Reads a file").
- No blank line between the summary and `Args:` if the docstring is short.
- Use `Args:`, `Returns:`, `Raises:` section headers — always with a colon.
- List each parameter as `name: description` (indented 4 spaces under `Args:`).
- Do **not** repeat the type — it's already in the annotation.
- Private methods (`_name`) don't require docstrings, but add one if the logic is non-obvious.

---

## 9. Comments

- Explain **why**, not **what**. The code itself should explain what.
- Use full sentences with proper capitalization and punctuation.
- Keep comments up to date — a wrong comment is worse than no comment.

```python
# ✅ Good — explains the reasoning
# Retry up to 3 times because the upstream API occasionally returns 503.
for attempt in range(MAX_RETRIES):
    ...

# ❌ Bad — restates the code
# Loop 3 times
for attempt in range(3):
    ...
```

### Tags

| Tag     | Meaning                                    |
|---------|--------------------------------------------|
| `TODO`  | Known future work                          |
| `FIXME` | Known bug that needs fixing                |
| `NOTE`  | Important context the reader must not miss |
| `HACK`  | Temporary workaround — explain why         |

---

## 10. Error Handling

- Catch **specific** exceptions. Never bare `except:` or `except Exception:` unless you re-raise or log.
- Always include a meaningful error message.
- Use custom exception classes for domain-specific errors.

```python
# ✅ Good
try:
    data = load_config(path)
except FileNotFoundError as e:
    raise ConfigError(f"Config file not found: {path}") from e

# ❌ Bad
try:
    data = load_config(path)
except:
    pass
```

**Rules:**

- Use `raise ... from e` to preserve the original traceback.
- Do not use exceptions for flow control.
- Log exceptions at the appropriate level (`logger.error`, `logger.warning`), never `print()`.

---

## 11. What to Avoid

| Anti-pattern                        | Do this instead                                  |
|-------------------------------------|--------------------------------------------------|
| `print()` for debugging             | Use the `logging` module                         |
| Mutable default arguments           | Use `None` and assign inside the function body   |
| Magic numbers                       | Named constants: `MAX_RETRIES = 3`               |
| `import *` in application code      | Explicit imports                                 |
| Very long functions (> ~50 lines)   | Break into smaller, focused functions            |
| Deeply nested code (> 3 levels)     | Extract logic into helper functions or early return |
| `type: ignore` without a comment    | Add a comment explaining why it's needed         |
| Commented-out code                  | Delete it — version control tracks history       |
| `pip install` anything              | Use `uv add` and `uv sync`                      |
| Manual virtualenv creation          | Let `uv sync` handle it                          |

---

## 12. Git Commit Messages

Use the **Conventional Commits** format:

```
<type>(<scope>): <short summary>

[optional body]

[optional footer]
```

### Types

| Type       | Use for                                        |
|------------|------------------------------------------------|
| `feat`     | A new feature                                  |
| `fix`      | A bug fix                                      |
| `docs`     | Documentation changes only                     |
| `style`    | Formatting, no logic change                    |
| `refactor` | Code restructuring, no feature or bug change   |
| `test`     | Adding or updating tests                       |
| `chore`    | Build process, dependency updates, tooling     |

### Examples

```
feat(auth): add JWT token refresh endpoint
fix(loader): handle empty file edge case
docs(readme): update installation instructions
refactor(api): extract pagination logic into helper
```

### Rules

- Summary line: **50 characters max**, imperative mood ("add", not "added").
- Body lines: **72 characters max**.
- Reference issues in the footer: `Closes #42`, `Fixes #17`.
