# Agent Rules

You are working on a Python project. Follow these rules strictly.

## Before Writing Any Code

1. Read `CODING_STYLE.md` for formatting, naming, type, and docstring rules.
2. Read `ARCHITECTURE.md` to understand where new code should go.
3. Read `CONTRIBUTING.md` for branch naming and commit conventions.

## Mandatory Rules

### Environment
- Use `uv` for all package management. Never use `pip install`.
- Add dependencies with `uv add <package>` (or `uv add --dev <package>` for dev tools).
- Run commands with `uv run <command>`.

### Code Style
- Line length: 80 characters.
- Format with `black`. Sort imports with `isort` (profile=black).
- Run `make style` to auto-format before committing.
- Run `make checks` to verify formatting, linting, and types.

### Naming
- Variables and functions: `snake_case`.
- Classes: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Booleans: prefix with `is_`, `has_`, `can_`, `should_`.
- Be descriptive. Never use vague names like `data`, `result`, `tmp`, `info`.

### Type Annotations
- All public functions and methods must have type annotations on parameters and return values.
- Use modern syntax: `list[str]`, `dict[str, int]`, `X | None` — not `typing.List`, `typing.Optional`.

### Docstrings
- Required on all public classes, methods, and functions.
- Use Google-style format with `Args:`, `Returns:`, `Raises:` sections.
- Do NOT use reST-style (`:param:`, `:returns:`, `:raises:`).
- List parameters as `name: description` (indented under `Args:`).
- First line: short imperative summary ("Load the config", not "Loads the config").

### Imports
- Use absolute imports only.
- Never use `import *` in application code.
- Let `isort` handle grouping and ordering.

### Error Handling
- Catch specific exceptions. Never use bare `except:` or `except Exception: pass`.
- Always provide meaningful error messages.
- Use `raise ... from e` to preserve traceback when wrapping exceptions.

### Logging
- Use the `logging` module. Never use `print()` for debugging or status messages.

## Things You Must Never Do

- Do not use `pip install`. Use `uv add`.
- Do not use `print()` for debugging. Use `logging`.
- Do not use bare `except:` or swallow exceptions silently.
- Do not leave commented-out code. Delete it.
- Do not use magic numbers. Define named constants.
- Do not use mutable default arguments (`def f(x=[])`). Use `None` and assign inside.
- Do not add `type: ignore` without a comment explaining why.
- Do not put business logic in API route handlers. Use the `services/` layer.
- Do not write raw database queries in `services/`. Use the `db/` layer.
- Do not create circular imports. Dependencies flow inward (see ARCHITECTURE.md).

## Git Commits

- Use Conventional Commits: `<type>(<scope>): <summary>`
- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.
- Summary: imperative mood, 50 chars max.

## Before Finishing

- Run `make style` to auto-format.
- Run `make checks` to verify everything passes.
- Ensure all new public functions have type annotations and docstrings.
- Ensure tests exist for new functionality.
