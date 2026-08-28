# Architecture

> Overview of the project structure, key modules, and design decisions.
> Read this before adding new code so you know **where things go**.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Module Responsibilities](#2-module-responsibilities)
3. [The Synthetic Data Module](#3-the-synthetic-data-module)
4. [Design Principles](#4-design-principles)
5. [Key Decisions](#5-key-decisions)
6. [Adding New Code](#6-adding-new-code)

---

## 1. Project Structure

```
.
├── CODING_STYLE.md              # Code style rules (must follow)
├── CLAUDE.md                    # Auto-loaded rules for Claude AI
├── GEMINI.md                    # Auto-loaded rules for Gemini AI
├── ARCHITECHTURE.md             # This file
├── Makefile                     # make style / lint-check / type-check / test
├── pyproject.toml               # Project config, dependencies, tool settings
├── uv.lock                      # Locked dependency versions (committed)
├── README.md                    # Project documentation
├── bitikocr/                    # Main package
│   ├── __init__.py
│   ├── py.typed                 # Type hinting marker
│   ├── cli.py                   # Application entry point
│   ├── config.py                # The only module that reads the environment
│   ├── models/                  # Data classes and schemas
│   │   ├── geometry.py          # BoundingBox
│   │   └── annotation.py        # Line / Block / Document annotations
│   ├── utils/                   # Pure helpers, no business logic
│   │   └── image_ops.py         # Alpha boxes, photometric augmentation
│   └── synthetic/               # Synthetic training-data generation
│       ├── fonts.py             # Font discovery, coverage, metrics
│       ├── style.py             # HandwritingStyle and its sampler
│       ├── hand.py              # The handwriting renderer
│       ├── effects.py           # Signature scribbles, round office seals
│       ├── layout.py            # Page canvas + ground-truth collection
│       ├── templates.py         # Measured geometry of pre-printed forms
│       ├── system_fonts.py      # Printed fonts for stamps and serial numbers
│       ├── sample_data.py       # Built-in field values for demos and tests
│       ├── dataset.py           # Batch generation and writing to disk
│       ├── generators/          # One module per document type
│       │   ├── base.py          # DocumentGenerator contract
│       │   ├── ariza.py
│       │   └── death_certificate.py
│       └── assets/              # Shipped with the package
│           ├── fonts/           # Handwriting fonts
│           ├── backgrounds/     # Blank form scans
│           └── layouts/         # Measured field geometry, one JSON per form
└── tests/                       # pytest suite, one module per source module
```

---

## 2. Module Responsibilities

| Module          | Responsibility                                          | Depends on                  |
|-----------------|---------------------------------------------------------|-----------------------------|
| `cli`           | Application entry point, CLI setup                      | `config`, `synthetic`       |
| `config`        | Load and validate configuration from env/files          | (none)                      |
| `models/`       | Data classes, schemas, type definitions                 | (none)                      |
| `utils/`        | Pure helper functions, no business logic                | `models`                    |
| `synthetic/`    | Synthetic handwritten-document generation               | `config`, `models`, `utils` |

The dependency flow is one-way. Nothing in `models/` or `utils/` may import
from `synthetic/`, and nothing below `cli` may read the environment.

---

## 3. The Synthetic Data Module

`bitikocr/synthetic/` produces training pages together with their ground
truth. It is layered so each piece has exactly one job:

| Layer            | Knows about                                  | Does not know about        |
|------------------|----------------------------------------------|----------------------------|
| `fonts`          | Font files, coverage, proportions            | Documents, pages, styles   |
| `style`          | What varies between writers and sheets       | Rendering, layout          |
| `hand`           | Turning a string into handwritten ink        | Documents, pages           |
| `effects`        | Signatures and office seals                  | Documents, text            |
| `layout`         | Placing ink and recording what was placed    | Which document is being made |
| `templates`      | Measured geometry of one blank form          | Rendering, handwriting     |
| `generators/`    | Where things go on one kind of document      | How ink is drawn           |
| `dataset`        | Batches, file names, on-disk format          | Layout, rendering          |

### The contract

Every document type subclasses `DocumentGenerator` and implements:

```python
generate(fields, seed=None, style_overrides=None) -> SyntheticDocument
```

`fields` maps field names to text; `SyntheticDocument` pairs the rendered
image with a `DocumentAnnotation`. Because the interface is uniform, the CLI
and the dataset builder never special-case a document type.

### Reproducibility

Every page is a pure function of its seed. The seed is recorded in the
annotation and in the output file name, so any sample can be regenerated
exactly. Nothing in the pipeline calls the global `random` module: a
`random.Random` is created from the seed and threaded through explicitly.

### Ground truth

`Page` is the only place that touches both the image and the annotation. It
records a `LineAnnotation` per rendered line and a `BlockAnnotation` per
logical region, each with the tight box around the ink that was actually
drawn. Boxes are measured from the composited alpha channel rather than
predicted from metrics, so they stay correct through jitter, slant and slope.

### Form templates

A pre-printed form's geometry lives in a layout JSON asset, not in Python.
:class:`FormTemplate` loads it, so the measurements stay the single source
of truth rather than being retyped into code, and a second variant of an
existing form — the single-page death certificate next to the two-page
bilingual one — is a new JSON rather than a new class.

`DeathCertificateGenerator` therefore knows *how* a clerk fills a form, and
the template knows *where* the rules are. Adding a variant:

1. Add the blank scan to `assets/backgrounds/`.
2. Add its measured layout to `assets/layouts/<name>.json`.
3. `bitikocr synth generate death_certificate --template <name>`.

A generator opts into templates by overriding
`DocumentGenerator.with_template`; the base refuses one, so passing a
template to a free-layout document like the ariza is an error rather than a
silent no-op.

### Adding a document type

1. Add `bitikocr/synthetic/generators/<type>.py` with a `DocumentGenerator`
   subclass that defines `name`, `field_names` and `reading_order`.
2. Register it in `GENERATOR_TYPES` in `generators/__init__.py`.
3. Add its field values to `sample_data.py`.
4. Add a test module under `tests/`.

Nothing else changes: the CLI, the registry and the dataset builder pick the
new type up automatically.

---

## 4. Design Principles

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

## 5. Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Package manager | `uv` | Fast, deterministic lockfile, replaces pip+virtualenv |
| Formatter | `black` | No config debates, one canonical style |
| Import Sorter | `isort` | Groups and sorts imports (profile=black) |
| Linter | `ruff` / `flake8` | Linting standard |
| Type checker | `mypy` | Catches type errors before runtime |
| Test framework | `pytest` | Industry standard, better than unittest |
| Docstring format | Google-style | Clear, readable format for APIs |
| Imaging dependencies | `data` extra | Pillow, numpy and fontTools are only needed to *generate* data, not to consume it |
| Assets location | Inside the package | `uv run bitikocr` works from any directory, and a wheel ships everything it needs |
| Style representation | Frozen dataclass | Typed, reproducible and safe to pass around; overrides go through `replace()` |
| Ground-truth format | One JSON per image | Image and annotation share a file stem; no index to keep in sync |

---

## 6. Adding New Code

### Checklist for new modules

- [ ] File is in the correct directory per the table above
- [ ] File and module names use `snake_case`
- [ ] Module has a docstring at the top explaining its purpose
- [ ] All public functions/classes have type annotations and docstrings
- [ ] Corresponding test file created in `tests/`
- [ ] No circular imports (follow the dependency flow)
- [ ] `make checks` passes
