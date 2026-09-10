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
├── scripts/                     # Entry points, grouped by area
│   └── data/
│       └── generate_synth.py    # Generate synthetic training documents
├── bitikocr/                    # Main package: library code only
│   ├── __init__.py
│   ├── py.typed                 # Type hinting marker
│   ├── config.py                # The only module that reads the environment
│   └── data/                    # Everything to do with the corpus
│       ├── models/              # The vocabulary the corpus is described in
│       │   ├── geometry.py      # BoundingBox
│       │   └── schema.py        # The corpus interchange schema
│       └── synthetic/           # Synthetic training-data generation
│           ├── README.md        # Developer's guide to this module
│           ├── fonts.py         # Font discovery, coverage, metrics
│           ├── style.py         # HandwritingStyle and its sampler
│           ├── hand.py          # The handwriting renderer
│           ├── effects.py       # Signature scribbles, round office seals
│           ├── annotation.py    # The generator's own ground truth
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
│           │   ├── letter.py    # LetterGenerator: writing on a blank sheet
│           │   ├── ariza.py
│           │   ├── consent_letter.py
│           │   ├── birth_certificate.py
│           │   └── death_certificate.py
│           ├── assets/          # Shipped with the package
│           │   ├── fonts/       # Handwriting fonts
│           │   ├── backgrounds/ # Blank form scans
│           │   └── layouts/     # Measured field geometry, one per form
│           └── output/          # Generated documents; gitignored
└── tests/                       # The whole suite, one module per module
```

Generated documents go to `bitikocr/data/synthetic/output/<type>/`, inside
the generator that makes them, and are gitignored and excluded from the
wheel. `bitikocr/data/` is where the corpus lives as a whole: real scans and
the pipelines that prepare them belong beside `synthetic/` as they arrive.

---

## 2. Module Responsibilities

| Module          | Responsibility                                          | Depends on                  |
|-----------------|---------------------------------------------------------|-----------------------------|
| `scripts/`      | Entry points: argument parsing, wiring, reporting       | `config`, `data`            |
| `config`        | Load and validate configuration from env/files          | (none)                      |
| `data/models/`  | The corpus schema and the geometry it uses              | (none)                      |
| `data/synthetic/` | Synthetic handwritten-document generation             | `config`, `data/models`     |

The dependency flow is one-way. Nothing in `data/models/` may import from a
producer beside it, and nothing below an entry point may read the
environment.

### Where entry points go

`scripts/` holds what a person runs, grouped by area: `scripts/data/` for
the corpus, and a directory of its own for training or evaluation when
those land. The package holds only library code, so nothing inside it
parses arguments or prints to a terminal.

The test is whether anything imports it. A module that only ever runs is an
entry point and belongs in `scripts/`; a module something imports belongs in
the package, where it can be reused and where a wheel will carry it.

There is no `utils/`. It held one live helper used only by the generator,
which now sits beside it in `data/synthetic/ink.py`. A package for shared
helpers is worth adding when something is genuinely shared; kept alive on
speculation it collects whatever has no other home.

`data/models/` is held to the same test, so it holds only two things: the
corpus schema and the geometry it is written in. It sits under `data/`
because it describes the corpus rather than any one way of filling it: the
schema's `origin` covers real scans and augmented copies and its `status`
covers human review, so `synthetic/` is only its first producer and real
scans land beside it under the same parent.

Keeping it at the top level would have said the opposite — that these types
serve the whole application — on the strength of a recogniser that does not
exist yet. The top-level name is better left free for one, where `models`
will mean what it usually means in a project like this.

A structure that serves one producer belongs to that producer. The
generator's own ground truth is `data/synthetic/annotation.py`, not a shared
model: `export.py` exists precisely to translate it into the schema, and
that translation is the tell. It carries whatever the renderer finds useful
— a seed, a style, a font — while the schema stays stable.

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
| `annotation`     | What was drawn, in the generator's own terms | The corpus schema          |
| `layout`         | Placing ink and recording what was placed    | Which document is being made |
| `templates`      | Measured geometry of one blank form          | Rendering, handwriting     |
| `scripts`        | The two alphabets and how to convert         | Documents, rendering       |
| `corpus`         | Uzbek names, places, months, causes          | Documents, rendering       |
| `records`        | What one document *says*                     | How it is drawn            |
| `facts`          | Which values are of which kind               | How anything is drawn      |
| `export`         | Mapping internals onto the corpus schema     | How anything is drawn      |
| `generators/`    | Where things go on one kind of document      | What it says, how it ages  |
| `generators/form`| How a clerk fills a measured printed form    | Where the cells are        |
| `generators/letter` | How a letter is arranged on a blank sheet | Which letter is being written |
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
2. Add its measured layout to `assets/layouts/<document type>_<variant>.json`,
   reusing the field ids of the existing layout wherever both forms have
   the same cell.
3. `uv run python scripts/data/generate_synth.py generate death_certificate --template <name>`.

Adding a new form is the same plus a `FormGenerator` subclass naming it.

Letters have the same shape of split without the measuring. Nothing about a
blank sheet can be read off a scan, so `LetterGenerator` arranges the page
itself — addressee block, title, body, signature — and each letter subclass
adds only its foot: an ariza carries the receiving office's registration
marks, a consent letter whatever attested to it. A new kind of letter is a
subclass and a sampler, with no asset at all.

What reaches a page can depend on facts about the document that are never
written on it. A consent letter is sealed only by an author who has a seal —
an organisation always, a citizen never, and the official who certified a
citizen's signature on their behalf. The sampler decides that once and
records it as `notes.author_kind`; the generator reads only the fields, so
it stays a question of what the record says rather than of who is drawing
it.

Layouts are named `<document type>_<variant>` and that prefix is how a
form's variants are found (`SyntheticConfig.layouts_for`). Variants differ
in more than geometry — the single-page death certificate has no
citizenship cell and joins the issue day and month into one — so the record
is the superset: the sampler emits every spelling any variant needs, the
template writes the fields it has cells for, and the facts file beside a
page is built from what reached that page, not from the whole record.

A generator opts into templates by overriding
`DocumentGenerator.with_template`; the base refuses one, so passing a
template to a free-layout document like the ariza is an error rather than a
silent no-op.

Layouts come from whatever tool measured the scan, so the loader accepts
more than one spelling of the same fact — see the README for the dialects.
Roles are inferred from each entry's free-text `text_type`, which is what
lets a layout describe a seal, a signature zone, machine-printed text or a
printed QR code that must never be drawn over.

What a form prints for itself is a layout fact too, not a rendering one. A
machine-printed area carries a `prefix` when its label is missing from the
blank: one certificate prints "I-HR №" and is given only the digits,
another prints nothing there and is given `"prefix": "№"` so the sign is
typeset with them.

### Adding a document type

1. Add `bitikocr/data/synthetic/generators/<type>.py` with a
   `DocumentGenerator`
   subclass that defines `name`, `field_names` and `reading_order`.
2. Register it in `GENERATOR_TYPES` in `generators/__init__.py`.
3. Add a record sampler to `records.py` and register it in
   `RECORD_SAMPLERS`, widening `corpus.py` if it needs new vocabulary.
4. Add `tests/test_<module>.py`, named after the module it covers.
5. A command someone runs goes in `scripts/<area>/`, not in the package.

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
| Assets location | Inside the package | The generator finds them from any working directory, and a wheel ships everything it needs |
| Style representation | Frozen dataclass | Typed, reproducible and safe to pass around; overrides go through `replace()` |
| Ground-truth format | One JSON per image | Image and annotation share a file stem; no index to keep in sync |

---

## 6. Adding New Code

### Where tests go

Every test lives in `tests/` at the repository root, one module per module
of the package and named after it. The generator's `records.py` is covered
by `tests/test_records.py`, the CLI by `tests/test_cli.py`.

The package therefore holds only shipping code, and the suite is one
directory a newcomer can read end to end. `tests/conftest.py` holds the
fixtures, and because there is a single directory every fixture reaches
every module without being imported.

Module names must stay unique across the suite, since they share one
namespace. That is what naming each after the module it covers already
guarantees.

### Checklist for new modules

- [ ] File is in the correct directory per the table above
- [ ] File and module names use `snake_case`
- [ ] Module has a docstring at the top explaining its purpose
- [ ] All public functions/classes have type annotations and docstrings
- [ ] Corresponding `tests/test_<module>.py`, named after the new module
- [ ] No circular imports (follow the dependency flow)
- [ ] `make checks` passes
