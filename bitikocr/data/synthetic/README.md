# Synthetic document generation

This module produces handwritten Uzbek documents together with exact ground
truth, so a recogniser can be trained before enough real archive scans have
been transcribed.

Everything is reproducible: a page is a pure function of its seed, ageing
included.

---

## Quick start

Generate thirty birth certificates with box overlays to look at:

```bash
uv run python scripts/data/generate_synth.py generate birth_certificate -n 30 --boxes
```

They land in `output/birth_certificate/`, beside this file. To see what is available:

```bash
uv run python scripts/data/generate_synth.py list-types
uv run python scripts/data/generate_synth.py list-templates
uv run python scripts/data/generate_synth.py list-fonts
```

---

## The two stages

Generation is two steps, and they can be run separately:

```
sample records  ->  facts/*.json  ->  render  ->  images + annotations + facts
```

```bash
uv run python scripts/data/generate_synth.py facts death_certificate -n 30
uv run python scripts/data/generate_synth.py render bitikocr/data/synthetic/output/death_certificate
```

Splitting them means the text can be read, corrected or replaced before the
slow step runs, and the same text can be re-rendered with different fonts or
heavier ageing. `synth generate` does both in one go.

Before a render, `facts/` holds the **generator's own record** — what a page
will be told to say. Rendering replaces each file with the corpus **facts
record** for the finished page. Editing a record between the two steps
changes what gets drawn.

---

## What lands on disk

```
bitikocr/data/synthetic/output/birth_certificate/
  facts/doc_000002.json        structured values on the page
  images/doc_000002.png        the rendered page
  annotations/doc_000002.json  the transcription record
  previews/doc_000002.png      box overlays, only with --boxes
  index.jsonl                  one line per page, for a data loader
```

Every file for one document shares its id, so a training pipeline pairs
them by name. `--id-prefix` namespaces a set if several will be merged.

`output/` sits inside this module, and the path is anchored to the package
rather than to the working directory, so a run writes to the same place
wherever it is started from. It is gitignored and excluded from the wheel.
Override it with `-o` or `BITIKOCR_OUTPUT_DIR`.

Which way a page was produced is recorded in the records themselves, as
`source.origin`, so a corpus that later mixes in real scans can still tell
them apart.

Both JSON records follow the schema in `bitikocr/data/models/schema.py`, which is
the typed definition and the place to look first. Boxes there are
`[x, y, width, height]`; internally the renderer works in two corners.

---

## The files, and what each is for

Read them in roughly this order — each layer knows only about the ones above
it.

| File | What it owns | What it deliberately does not know |
|---|---|---|
| `scripts.py` | The Latin and Cyrillic alphabets, and transliteration between them | Documents, rendering |
| `corpus.py` | The Uzbek vocabulary: names, places, months, causes | Documents, rendering |
| `records.py` | What one document *says* — field values, dates, era | How any of it is drawn |
| `facts.py` | Which field values are of which kind | How anything is drawn |
| `fonts.py` | Font discovery, coverage checks, size equalisation | Documents, pages |
| `style.py` | The sampled "writer": pen, ink, slant, spacing | Rendering, layout |
| `hand.py` | Turning a string into handwritten ink | Documents, pages |
| `effects.py` | Signature scribbles and round office seals | Documents, text |
| `system_fonts.py` | Printed fonts for stamps and serial numbers | Handwriting |
| `templates.py` | The measured geometry of one blank form | Rendering, handwriting |
| `ink.py` | Measuring the ink a rendered layer carries | Documents, pages |
| `annotation.py` | What was drawn, in this module's own terms | The corpus schema |
| `layout.py` | Placing ink and recording what was placed | Which document is being made |
| `generators/` | Where things go on one kind of document | What it says, how it ages |
| `augment.py` | Spoiling a finished page like a scan | What the page says |
| `export.py` | Mapping all of the above onto the corpus schema | How anything is drawn |
| `dataset.py` | Batches, file names, on-disk layout | Layout, rendering |

Inside `generators/`:

| File | What it is |
|---|---|
| `base.py` | The `DocumentGenerator` contract every type implements |
| `form.py` | `FormGenerator`: filling any pre-printed form |
| `letter.py` | `LetterGenerator`: writing any letter on a blank sheet |
| `ariza.py` | An application letter: a letter plus the office's marks |
| `consent_letter.py` | A consent letter: a letter plus whatever attests to it |
| `birth_certificate.py` | Four lines: a name and a default template |
| `death_certificate.py` | Likewise |

There are two engines because there are two kinds of page. A certificate is
a **printed form** whose cells were measured, so `FormGenerator` reads where
everything goes from a layout. A letter is written on a **blank sheet**, so
`LetterGenerator` arranges it instead: addressee block top-right, centred
title, wrapped body, signature, shrinking the hand until it fits. What each
letter adds is its foot — an ariza carries the receiving office's
registration marks, a consent letter whatever attests to it.

### Who may seal a consent letter

A seal is not decoration, and it is not on every page. Who wrote the letter
decides whether one appears at all:

| Author | Signs | Seal |
|---|---|---|
| A citizen, uncertified | Themselves | **None.** A private person has no seal |
| A citizen, certified | Themselves, then the official | The **official's** — a notary's or the mahalla's |
| An organisation | Its head | **Always**, and it is the organisation's own |

A citizen also identifies themselves by passport and telephone in the
sender's block; an organisation identifies itself by name and the post its
signatory holds, and carries neither. The subjects follow from the same
split: a citizen consents about a boundary, a privatisation or a housing
claim, while an organisation consents in the plural to work being done or
its premises being used.

Which sort of author wrote a page is recorded on the record as
`notes.author_kind`. It is never written on the page, so it is a label to
filter a corpus by, not something a recogniser is asked to read.

This module's tests are in `tests/` at the repository root, with the rest of
the project's suite, one module there per module here. Run the generator's
own with:

```bash
uv run pytest tests/test_generators.py tests/test_templates.py
```

Assets in `assets/`:

| Directory | What it holds |
|---|---|
| `fonts/` | Handwriting fonts. `CyrilicHand*` write Cyrillic, `LatinHand*` Latin, `MixHand*` both |
| `backgrounds/` | Blank form scans |
| `layouts/` | Measured field geometry, one JSON per form |

---

## Two ideas worth knowing

**Content is separate from rendering.** A `DocumentRecord` is field values
in one alphabet plus a seed. It says nothing about fonts, ink or paper. That
split is what makes the two stages possible.

**Geometry lives in data, not code.** A pre-printed form's layout is a
measured JSON in `assets/layouts/`, so the measurements stay the single
source of truth and a second variant of a form is a new file rather than a
new class. `FormGenerator` knows *how* a clerk fills a form; the template
knows *where* the lines are.

Ground truth is measured, never predicted: every box comes from the alpha
channel of the ink actually drawn, so it stays correct through jitter,
slant, line slope and scan skew.

---

## Knobs

| Flag | Stage | What it does |
|---|---|---|
| `--script` | facts | Force `latin` or `cyrillic` for the whole batch |
| `--latin-share` | facts | Share written in Latin when neither is forced (default 0.3) |
| `--seed` | facts | Make the whole run reproducible |
| `--id-prefix` | both | What documents are called (default `doc`) |
| `--template` | render | Which form variant to fill |
| `--font` | render | Force one handwriting font instead of sampling |
| `--fonts-dir` | render | Use your own fonts |
| `--ink` | render | How heavily the pen writes (default 1.4) |
| `--augment` | render | How hard to spoil each page; `0` disables it |
| `--boxes` | render | Also write a box overlay per page |

`--latin-share` is set from the font library rather than from the archive:
seven fonts can write Latin against seventeen for Cyrillic, and 0.3 gives
each font of either alphabet roughly the same number of pages. An even split
would lean the Latin half on too few hands. Move it as fonts are added.

`--ink` exists because the shipped hands vary a lot in stroke weight and the
thinnest wrote too faintly to read once a page had been aged. Raise it if a
font still comes out light.

---

## How to extend it

### Add handwriting fonts

Drop `.ttf` or `.otf` files into `assets/fonts/`. Nothing else is needed: a
font is measured on load and only offered text it can actually render, so a
Latin-only font never receives a Cyrillic record. Check what it covers with
`synth list-fonts`.

### Add a variant of an existing form

1. Put the blank scan in `assets/backgrounds/`.
2. Put its measured layout in `assets/layouts/<document type>_<variant>.json`.
   The prefix is how a form's variants are found, so keep it.
3. `uv run python scripts/data/generate_synth.py generate death_certificate --template <name>`

No code changes, as long as the new form's cells go by the names the
record sampler already emits — reuse the ids of the existing layout for
that document type wherever the two forms have the same cell. The layout's
`width`/`height` must match the scan exactly, since every coordinate is
scaled from them.

A variant may lack a cell the record has, or spell one differently. The
death certificate ships two:

| Layout | Form | Differs in |
|---|---|---|
| `death_certificate_bilingual` | two-page Latin/Cyrillic, the default | — |
| `death_certificate_cyrillic_single` | older single page, Cyrillic only | no citizenship cell; issue day and month share one cell (`issue_day_month`); a typeset `form_series` beside the serial |

The record is the superset: the sampler emits every spelling any variant
needs, the template writes the fields it has cells for, and the facts file
beside a page lists only what reached that page. The bilingual form suits
either alphabet; the single-page form is printed in Cyrillic, so it is
usually rendered with `--script cyrillic`.

### Add a whole new form

The above, plus a subclass naming the document type:

```python
class MarriageCertificateGenerator(FormGenerator):
    name = "marriage_certificate"
    default_template = "marriage_certificate_bilingual"
```

Register it in `GENERATOR_TYPES`, add a record sampler to `records.py`, and
add its field categories to `FIELD_FACTS` in `facts.py`.

### Add a new kind of letter

A letter has no form to measure, so there is no layout to add. Subclass
`LetterGenerator`, name the document type and the title, and write what goes
below the signature:

```python
class ComplaintLetterGenerator(LetterGenerator):
    name = "complaint_letter"
    default_title = "Shikoyat xati"

    def _put_foot(self, page, fields, second_hand, rng, style, y):
        ...
```

Then the same three steps: register it in `GENERATOR_TYPES`, add a sampler
to `records.py`, and map its fields in `FIELD_FACTS`. Declare `FIELD_NAMES`
and `READING_ORDER` on the class — the reading order is the order the
transcription comes out in, so it should be the order a person reads the
page.

### Widen the text

`corpus.py` holds the vocabulary in Latin and transliterates on demand.
Adding names, districts or causes there widens every document type at once.
Words the transliteration rules get wrong — the month names, for one — are
given explicitly as `(latin, cyrillic)` pairs.

### Correct a fact category

`FIELD_FACTS` in `facts.py` maps each field to its category, one line per
field. The vocabulary is versioned by `CATEGORY_SET_VERSION` and an unknown
category is refused at import, so a typo cannot reach a corpus.

---

## The layout JSON

Field geometry is measured on a scan of the blank form:

```json
{
  "template_name": "death_certificate_bilingual",
  "image": "death_certificate_bilingual.png",
  "width": 1419, "height": 1108,
  "printed_scripts": ["latin", "cyrillic"],
  "fields": [
    {"id": "surname", "baseline_y": 331, "bbox_xyxy": [94, 293, 607, 335],
     "text_type": "uzbek_latin_word"}
  ]
}
```

`text_type` is free text; the first role whose keywords it mentions wins:

| Mentions | Becomes |
|---|---|
| `qr`, `barcode`, `photo` | a keep-out zone — never drawn on |
| `stamp`, `seal` | the office seal's area |
| `signature` | where the registrar signs |
| `printed_digits`, `typographic`, `series` | machine-printed text |
| `digit` | a field written in a digit hand |
| anything else | a handwritten text field |

Order matters, so `digits_7 (typographic)` is typeset rather than written.

A printed area may carry a `prefix`, the label its value is typeset behind.
It is for forms whose blank does not print that label itself: the bilingual
certificate has "I-HR №" on the paper and takes the bare digits, while the
single-page one has nothing there and is given `"prefix": "№"`, so the page
reads `II-HR № 0024695`. Printed type shrinks to stay inside its measured
area, since the series and the serial sit side by side.

Both spellings of a box are accepted — `bbox_xyxy: [x1, y1, x2, y2]` and
`bbox: {x1, y1, x2, y2}` — and both spellings of a rule, `underline_y` (the
printed line itself, preferred) and `baseline_y`. Ids ending `_line1`,
`_line2` merge into one field written across several rules.

Whether the registrar's name shares the signature area is read from the
geometry: an area with a rule through it is a name line that also gets
signed; an open zone with no rule is signed only.

---

## Using it from Python

```python
import random
from bitikocr.config import SyntheticConfig
from bitikocr.data.synthetic import create_generator, sample_records
from bitikocr.data.synthetic.dataset import render_records
from bitikocr.data.synthetic.augment import AugmentationProfile

config = SyntheticConfig.from_env()
records = sample_records("ariza", 10, random.Random(1), script="cyrillic")

generator = create_generator("ariza", config)
summary = render_records(
    generator,
    records,
    config.dataset_dir("ariza"),
    augmentation=AugmentationProfile(),
    draw_boxes=True,
)
print(len(summary), "pages ->", summary.layout.root)
```

For one page without the dataset machinery:

```python
document = generator.generate(records[0].fields, seed=records[0].seed)
document.image.save("page.png")
print(document.annotation.text)
```

`style_overrides` pins any writer parameter:

```python
generator.generate(fields, seed=1, style_overrides={"pen": "soft", "slant": 0.3})
```

---

## Gotchas

- **A page that no font can write is skipped, not fatal.** The batch
  continues and `DatasetSummary.skipped` lists what was dropped and why.
  Check it if a run produces fewer pages than asked for.
- **Boxes are measured, so a layout bug shows up as a warning**, not a bad
  label: ink drawn off the page is clipped and logged rather than claimed.
- **Augmentation is a step after rendering**, not a generator option. A
  generator always produces a clean page.
- **Only the scan skew moves the ground truth.** Every other augmentation
  step is photometric and leaves boxes alone.
- **Run `make checks` before committing.** It covers both suites —
  transliteration, record consistency, box validity and the schema.
