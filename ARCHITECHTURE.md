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
│   ├── tests/                   # Suite for the CLI and configuration
│   ├── models/                  # Data classes and schemas
│   │   ├── geometry.py          # BoundingBox
│   │   ├── annotation.py        # Line / Block / Document annotations
│   │   ├── schema.py            # The corpus interchange schema
│   │   └── tests/               # Suite for the above
│   └── data/                    # Everything to do with the corpus
│       └── synthetic/           # Synthetic training-data generation
│           ├── README.md        # Developer's guide to this module
│           ├── fonts.py         # Font discovery, coverage, metrics
│           ├── style.py         # HandwritingStyle and its sampler
│           ├── hand.py          # The handwriting renderer
│           ├── effects.py       # Signature scribbles, round office seals
│           ├── layout.py        # Page canvas + ground-truth collection
│           ├── templates.py     # Measured geometry of pre-printed forms
│           ├── system_fonts.py  # Printed fonts for stamps and serials
│           ├── scripts.py       # Latin/Cyrillic and the transliteration
│           ├── corpus.py        # Uzbek vocabulary records are drawn from
│           ├── records.py       # Sampling a document's field values
│           ├── facts.py         # Field values to structured facts
│           ├── export.py        # Internals to the corpus schema
│           ├── augment.py       # Spoiling a clean page like a scan
│           ├── ink.py           # Measuring the ink a layer carries
│           ├── dataset.py       # Records to disk, and rendering them
│           ├── generators/      # One module per document type
│           │   ├── base.py      # DocumentGenerator contract
│           │   ├── form.py      # FormGenerator: filling any printed form
│           │   ├── ariza.py
│           │   ├── birth_certificate.py
│           │   └── death_certificate.py
│           ├── assets/          # Shipped with the package
│           │   ├── fonts/       # Handwriting fonts
│           │   ├── backgrounds/ # Blank form scans
│           │   └── layouts/     # Measured field geometry, one per form
│           ├── tests/           # The generator's own suite
│           └── output/          # Generated documents; gitignored
```

Generated documents go to `bitikocr/data/synthetic/output/<type>/`, inside
the generator that makes them, and are gitignored and excluded from the
wheel. `bitikocr/data/` is where the corpus lives as a whole: real scans and
the pipelines that prepare them belong beside `synthetic/` as they arrive.

---

## 2. Module Responsibilities

| Module          | Responsibility                                          | Depends on                  |
|-----------------|---------------------------------------------------------|-----------------------------|
| `cli`           | Application entry point, CLI setup                      | `config`, `data`            |
| `config`        | Load and validate configuration from env/files          | (none)                      |
| `models/`       | Data classes, schemas, type definitions                 | (none)                      |
| `data/synthetic/` | Synthetic handwritten-document generation             | `config`, `models`          |

The dependency flow is one-way. Nothing in `models/` may import from
`data/`, and nothing below `cli` may read the environment.

There is no `utils/`. It held one live helper used only by the generator,
which now sits beside it in `data/synthetic/ink.py`. A package for shared
helpers is worth adding when something is genuinely shared; kept alive on
speculation it collects whatever has no other home.

---

## 3. The Synthetic Data Module

`bitikocr/data/synthetic/` produces training pages together with their
ground truth. It is layered so each piece has exactly one job:

| Layer            | Knows about                                  | Does not know about        |
|------------------|----------------------------------------------|----------------------------|
| `fonts`          | Font files, coverage, proportions            | Documents, pages, styles   |
| `style`          | What varies between writers and sheets       | Rendering, layout          |
| `hand`           | Turning a string into handwritten ink        | Documents, pages           |
| `effects`        | Signatures and office seals                  | Documents, text            |
| `layout`         | Placing ink and recording what was placed    | Which document is being made |
| `templates`      | Measured geometry of one blank form          | Rendering, handwriting     |
| `scripts`        | The two alphabets and how to convert         | Documents, rendering       |
| `corpus`         | Uzbek names, places, months, causes          | Documents, rendering       |
| `records`        | What one document *says*                     | How it is drawn            |
| `facts`          | Which values are of which kind               | How anything is drawn      |
| `export`         | Mapping internals onto the corpus schema     | How anything is drawn      |
| `generators/`    | Where things go on one kind of document      | What it says, how it ages  |
| `augment`        | Spoiling a finished page                     | What the page says         |
| `dataset`        | Batches, file names, on-disk format          | Layout, rendering          |

### The contract

Every document type subclasses `DocumentGenerator` and implements:

```python
generate(fields, seed=None, style_overrides=None) -> SyntheticDocument
```

`fields` maps field names to text; `SyntheticDocument` pairs the rendered
image with a `DocumentAnnotation`. Because the interface is uniform, the CLI
and the dataset builder never special-case a document type.

### Two stages

Content and rendering are separate steps, joined only by a record:

```
sample_records()  ->  facts/*.json  ->  render_records()  ->  images + annotations
```

A `DocumentRecord` is what a document *says* — field values in one alphabet,
plus the seed that will draw it. Nothing about fonts, ink or paper. That
split is what lets a batch's text be reviewed or hand-edited before the slow
step runs, and lets the same text be re-rendered with different fonts or
heavier augmentation.

Every file belonging to one document shares its id, so the three artefacts
a fine-tuning run needs — the facts, the page and its annotation — pair up
without an index. The index exists to iterate and filter the set.

### The corpus schema

`models/schema.py` is the interchange contract: a transcription record and
a facts record per document. It is deliberately independent of how a
document was produced, so a synthetic page, a real archive scan and an
augmented copy all describe themselves the same way and `source.origin`
tells them apart.

`synthetic/export.py` is the only place that maps the generator's terms —
blocks, lines, styles, seeds — onto that schema. The generator stays free to
change and the schema stays stable. Nothing there is guessed: the marks come
from what was drawn, the capture quality from the augmentation's own report,
the era from the year the record is dated.

Augmentation is likewise a step after rendering, not a generator option: a
generator produces a clean page, and `augment` decides how much of a scan it
should look like. A photometric step leaves the boxes alone; the one
geometric step, a scan skew, transforms them with the ink.

### Reproducibility

Every page is a pure function of its record's seed, augmentation included.
The seed is recorded in the annotation and in the output file name, so any
sample can be regenerated exactly. Nothing in the pipeline calls the global
`random` module: a `random.Random` is created from the seed and threaded
through explicitly.

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

`FormGenerator` therefore knows *how* a clerk fills a form — which hand
writes digits, how a value shrinks to fit, where the seal is pressed — and
the template knows *where* everything goes. Every certificate is that one
generator with a different template, so `BirthCertificateGenerator` and
`DeathCertificateGenerator` are four lines each: a document type name and a
default layout.

Adding a variant of an existing form:

1. Add the blank scan to `assets/backgrounds/`.
2. Add its measured layout to `assets/layouts/<name>.json`.
3. `bitikocr synth generate death_certificate --template <name>`.

Adding a new form is the same plus a `FormGenerator` subclass naming it.

A generator opts into templates by overriding
`DocumentGenerator.with_template`; the base refuses one, so passing a
template to a free-layout document like the ariza is an error rather than a
silent no-op.

Layouts come from whatever tool measured the scan, so the loader accepts
more than one spelling of the same fact — see the README for the dialects.
Roles are inferred from each entry's free-text `text_type`, which is what
lets a layout describe a seal, a signature zone, machine-printed text or a
printed QR code that must never be drawn over.

### Adding a document type

1. Add `bitikocr/data/synthetic/generators/<type>.py` with a
   `DocumentGenerator`
   subclass that defines `name`, `field_names` and `reading_order`.
2. Register it in `GENERATOR_TYPES` in `generators/__init__.py`.
3. Add a record sampler to `records.py` and register it in
   `RECORD_SAMPLERS`, widening `corpus.py` if it needs new vocabulary.
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

### Where tests go

Tests live beside the code they cover, one suite per layer:

| Suite | Covers |
|---|---|
| `bitikocr/tests/` | The CLI and the configuration it loads |
| `bitikocr/models/tests/` | The data classes and the corpus schema |
| `bitikocr/data/synthetic/tests/` | The generator |

There is no top-level `tests/`. One rule with no exception means a new
layer brings its own suite without anyone having to decide where it goes —
when the recogniser lands, `bitikocr/ocr/tests/` follows from the rule
rather than from a discussion.

Collecting the package finds every suite, so `pytest` and `mypy` are both
pointed at `bitikocr` alone. No suite ships in the wheel.

### Checklist for new modules

- [ ] File is in the correct directory per the table above
- [ ] File and module names use `snake_case`
- [ ] Module has a docstring at the top explaining its purpose
- [ ] All public functions/classes have type annotations and docstrings
- [ ] Corresponding test file beside the layer it covers
- [ ] No circular imports (follow the dependency flow)
- [ ] `make checks` passes
