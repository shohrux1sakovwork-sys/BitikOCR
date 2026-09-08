# BitikOCR

Handwritten text recognition (HTR) for Uzbek documents.

The project currently covers the first stage of the pipeline: generating
synthetic handwritten pages with pixel-accurate ground truth, so a recogniser
can be trained before enough real archive scans are transcribed.

## Setup

```bash
uv sync --extra data
```

The `data` extra pulls in Pillow, NumPy and fontTools, which are needed to
*generate* training data but not to consume it.

## Generating synthetic data

Generation is two stages, and they can be run together or apart.

**Both at once:**

```bash
uv run python scripts/data/generate_synth.py generate birth_certificate -n 30 --boxes
```

**Or separately** — sample the text first, look at it or edit it, then draw:

```bash
uv run python scripts/data/generate_synth.py facts death_certificate -n 30
```

```bash
uv run python scripts/data/generate_synth.py render bitikocr/data/synthetic/output/death_certificate --boxes
```

Splitting them means a batch can be re-rendered with different fonts or
heavier augmentation without resampling the text, and the text can be
reviewed before the slow step runs.

Useful flags:

| Flag                | Stage  | Meaning                                            |
|---------------------|--------|----------------------------------------------------|
| `-n, --count`       | facts  | How many documents.                                 |
| `-o, --output-dir`  | facts  | Where to write it (default beside the generator).   |
| `--script`          | facts  | Force `latin` or `cyrillic` (default: both).        |
| `--latin-share`     | facts  | Share written in Latin when neither is forced.      |
| `--seed`            | facts  | Make the whole run reproducible.                    |
| `--id-prefix`       | both   | What documents are called (default `doc`).          |
| `--template`        | render | Which form variant to fill.                         |
| `--font`            | render | Force one handwriting font instead of sampling.     |
| `--fonts-dir`       | render | Use your own handwriting fonts.                     |
| `--ink`             | render | How heavily the pen writes; raise it if too faint.  |
| `--augment`         | render | How hard to spoil each page; `0` disables it.       |
| `--boxes`           | render | Also write a box overlay per page.                  |
| `-v, --verbose`     | both   | Log every generated sample.                         |

To see what is available:

```bash
uv run python scripts/data/generate_synth.py list-types
```

```bash
uv run python scripts/data/generate_synth.py list-fonts
```

```bash
uv run python scripts/data/generate_synth.py list-templates
```

## Output format

A dataset directory holds both stages:

```
bitikocr/data/synthetic/output/birth_certificate/
  facts/doc_000002.json        the structured values on the page
  images/doc_000002.png        the rendered page
  annotations/doc_000002.json  the transcription record
  previews/doc_000002.png      box overlays, only with --boxes
  index.jsonl                  one line per page, for a data loader
```

Every file for one document shares its id, so a fine-tuning pipeline pairs
them directly. Use `--id-prefix` to namespace a set if several will be
merged into one corpus.

Both JSON records follow the corpus schema — see
`bitikocr/data/models/schema.py`, which is the typed definition.

An **annotation** is the transcription record: what is written, where, and
under what conditions the page was captured.

```json
{
  "id": "doc_000002",
  "image": "images/doc_000002.png",
  "source": {
    "origin": "synthetic",
    "collection": "birth_certificate",
    "era": "modern",
    "year_approx": 2019,
    "seed": 1803740873,
    "generator": "bitikocr@0.1.0",
    "font": "CyrilicHand06.otf"
  },
  "metadata": {
    "language": ["uz"],
    "scripts": ["cyrillic", "latin"],
    "primary_script": "cyrillic",
    "document_type": "birth_certificate",
    "text_mode": "mixed",
    "has_handwriting": true,
    "has_printed_text": true,
    "has_stamp": true,
    "has_signature": true,
    "layout": "two_column",
    "quality": {
      "blur": true, "rotation": -0.463, "skew": true,
      "noise": "medium", "capture": "scanner"
    }
  },
  "target": {
    "text": "whole text, reading order, one line per physical line",
    "parts": [
      {"role": "child_surname", "text": "Ҳакимова", "bbox": [637, 605, 217, 62],
       "lines": [{"text": "Ҳакимова", "bbox": [637, 605, 217, 62]}]}
    ]
  },
  "annotation": {
    "status": "gold",
    "pre_annotator": "generator:bitikocr@0.1.0",
    "reviewer": null, "reviewed_at": null,
    "revision": 1, "unclear_reason": null
  }
}
```

Boxes here are `[x, y, width, height]`. Every one is measured from the ink
actually drawn, so it stays correct through jitter, slant and scan skew.
`parts` are the page's regions; each also carries the `lines` inside it,
which is what line-level HTR training needs — a reader that only knows
`role`/`text`/`bbox` can ignore them.

Nothing in the metadata is assumed: `has_stamp` is read from what was drawn,
`quality` from what the augmentation actually did, `era` from the year the
document is dated.

A **facts** file is the structured values a reader would take off the page:

```json
{
  "id": "doc_000002",
  "image": "images/doc_000002.png",
  "facts": [
    {"category": "date", "value": "2019-06-25", "fuzzy": false,
     "evidence_text": "2019 йил июн 25", "field": "birth"},
    {"category": "person_name", "value": "Ҳакимова", "fuzzy": false,
     "evidence_text": "Ҳакимова", "field": "child_surname"}
  ]
}
```

`value` is normalised where there is a normal form — a date spread over
three cells becomes one ISO date, a year written out in words becomes the
year — and `evidence_text` is always the surface form as it appears on the
page, so a wrong fact can be traced back to the transcription. `fuzzy` is
never true for synthetic pages: the page was written from the fact.

The category vocabulary and the field-to-category mapping live in
`bitikocr/data/synthetic/facts.py`, versioned by `CATEGORY_SET_VERSION`. Correct
a wrong category there in one line.

Before a render, `facts/` instead holds the generator's own record — the
text a page will be told to say, with the seed that draws it. Rendering
replaces each with the facts record above:

```json
{
  "text": "Гулистон шаҳар ҳокими И. Й. Эрбековга\n...",
  "blocks": [{ "type": "recipient", "text": "...", "bbox": [905, 190, 1620, 350] }],
  "lines":  [{ "block": "recipient", "text": "...", "bbox": [905, 190, 1620, 262] }],
  "size": [1654, 2339],
  "document_type": "ariza",
  "script": "cyrillic",
  "seed": 647892279,
  "font": "CyrilicHand07.otf",
  "style": { "pen": "hard", "slant": 0.21, "...": "..." },
  "fields": { "recipient": "...", "body": "..." }
}
```

```json
{"document_type": "birth_certificate", "script": "cyrillic", "seed": 1803740873,
 "fields": {"child_surname": "Ҳакимова", "...": "..."},
 "notes": {"year_approx": 2019, "era": "modern", "dates": {"birth": "2019-06-25"}}}
```

`index.jsonl` is what a training loop reads: the three paths relative to the
dataset root, plus the transcription, the font, the alphabet, the era and
the seed — enough to filter or balance the set without opening every file.

Regenerating a page from its facts reproduces it byte for byte, including
the augmentation: the seed rides on the record. Editing a facts file and
re-running `synth render` draws the corrected text.

## Text and alphabets

Uzbek is written in both Latin and Cyrillic, and an archive holds both, so
records are sampled in either unless `--script` forces one. The vocabulary
lives in `corpus.py` in Latin and is transliterated on demand; widen those
lists to widen the data.

The default mix is **80% Cyrillic**, because the font library is: fourteen
Cyrillic hands against one Latin one, so Latin pages would otherwise be
drawn by the same few fonts over and over. Raise `--latin-share` as Latin
fonts are added.

Records are internally consistent — a death is registered after it happened,
a family shares a surname, an age matches the year — because inconsistent
text teaches a recogniser nothing but does make the data look wrong to a
reviewer.

A font is only used for text it can actually write, so the Cyrillic-only
hands never receive Latin records and vice versa. `synth list-fonts` shows
which alphabets each font covers.

## Ink

The shipped hands vary a lot in stroke weight, and the thinnest wrote too
faintly to read once a page had been aged and re-compressed. `--ink` sets
how heavily the pen writes; the default of 1.4 is calibrated above what the
fonts ask for so every hand stays legible. Raise it if a font still comes
out light, lower it towards 1.0 for the fonts' own weight.

## Augmentation

Rendered pages are clean. `--augment` spoils them the way a scanner and time
would: paper tint, uneven lighting, edge vignetting, dust specks, defocus,
sensor grain, JPEG artefacts and a slight scan skew.

The skew moves the ink, so it moves the bounding boxes with it — every other
step is photometric and leaves the ground truth alone. `--augment 0` turns
the whole thing off; higher values than the default `1.0` push it further.

## Using it from Python

```python
from bitikocr.config import SyntheticConfig
from bitikocr.data.synthetic import create_generator

generator = create_generator("ariza", SyntheticConfig.from_env())
document = generator.generate(
    {
        "recipient": "Гулистон шаҳар ҳокими И. Й. Эрбековга",
        "applicant": "Янгиер ш. Бахт кўчаси 4 уйда яшовчи фуқаро Усмонов С-дан",
        "body": "Аризам мазмуни шундан иборатки ...",
    },
    seed=42,
)

document.image.save("page.png")
print(document.annotation.text)
```

Use `style_overrides` to pin any writer parameter, for example
`generator.generate(fields, style_overrides={"pen": "soft", "slant": 0.3})`.

Generated documents live inside the generator that makes them, under
`bitikocr/data/synthetic/output/`. That path is anchored to the package
rather than to the working directory, so a run writes to the same place
wherever it is started from. Point it elsewhere with `-o` or
`BITIKOCR_OUTPUT_DIR`.

They are gitignored and excluded from the wheel: the fonts, blank scans and
layouts ship with the package, the pages generated from them do not.

Nothing is lost by keeping them there — every record carries
`source.origin`, which is `synthetic` for these, so a corpus that later
mixes in real scans can still tell them apart.

## Form templates

Documents that fill a pre-printed form — the birth and death certificates
today, more later — take their geometry from a **layout JSON**, not from
Python. They also share one generator: a certificate is a template plus a
four-line subclass, never a new rendering path.

Each layout is the measured position of every printed underline on one blank
form:

```json
{
  "template_name": "death_certificate_bilingual",
  "image": "death_certificate_bilingual.png",
  "width": 1419, "height": 1108,
  "fields": [
    {"id": "surname", "baseline_y": 331, "bbox_xyxy": [94, 293, 607, 335],
     "text_type": "uzbek_latin_word"},
    {"id": "death_year", "baseline_y": 487, "bbox_xyxy": [94, 449, 183, 491],
     "text_type": "digits_4"}
  ]
}
```

`text_type` is free text; the first role whose keywords it mentions wins:

| `text_type` mentions                       | Becomes                          |
|--------------------------------------------|----------------------------------|
| `qr`, `barcode`, `photo`                    | a keep-out zone — never drawn on |
| `stamp`, `seal`                             | the office seal's area           |
| `signature`                                 | where the registrar signs        |
| `printed_digits`, `typographic`, `series`   | machine-printed text             |
| `digit`                                     | a field written in a digit hand  |
| anything else                               | a handwritten text field         |

Order matters, so `digits_7 (typographic, red/black)` is typeset rather than
handwritten. A form may have several printed areas — the birth certificate
sets a series beside its number — each filled from the field of the same
name.

A printed area may also carry a `prefix`, the label its value is typeset
behind, for forms whose blank does not print that label itself. The
bilingual death certificate has "I-HR №" on the paper and takes the bare
digits; the single-page one has nothing there and is given `"prefix": "№"`,
so the page reads `II-HR № 0024695` as the real document does. Printed type
shrinks to stay inside its measured area rather than run into whatever the
form sets beside it.

Both spellings of a box are accepted, `bbox_xyxy: [x1, y1, x2, y2]` and
`bbox: {x1, y1, x2, y2}`, and both spellings of a rule, `underline_y` (the
printed line itself, preferred) and `baseline_y`.

Entries whose id ends in `_line1`, `_line2`, ... are one logical field
written across several printed rules, merged under the shared base name
(`cause_of_death_line1` + `cause_of_death_line2` → `cause_of_death`).

Values are written around the middle of the cell the layout gives them,
nudged either way within their slack, and the office seal is pressed a
little left of its printed circle with enough scatter to catch the lines
above or below it. Both are read from the measured area, so they scale with
whatever form is being filled.

Whether the registrar's name is written in the signature area or has a line
of its own is read from the geometry: an area with a rule through it is a
name line that also gets signed; an open zone with no rule is signed only,
and the form gives the name a field elsewhere.

**To add a form variant**, drop its scan in
`bitikocr/data/synthetic/assets/backgrounds/` and its measured layout in
`bitikocr/data/synthetic/assets/layouts/<document type>_<variant>.json`,
then:

```bash
uv run python scripts/data/generate_synth.py generate death_certificate --template <name> -n 20 --boxes
```

No code changes are needed, provided the new form's cells reuse the field
ids of the existing layout for that document type. The layout's
`width`/`height` must match the background scan exactly, since every
coordinate is scaled from them.

The death certificate ships two variants: `death_certificate_bilingual`,
the two-page Latin/Cyrillic form and the default, and
`death_certificate_cyrillic_single`, the older single-page Cyrillic form.
One record fills either. Where a variant has no cell for a field, or spells
one differently, the template writes what it has room for and the facts
file beside the page lists only what reached it.

**To add a whole new form** — a marriage certificate, a passport page — add
its layout and a subclass naming the document type and its default
template; see `bitikocr/data/synthetic/generators/birth_certificate.py`, which is
twenty lines including the docstring.

## Assets

Handwriting fonts, blank form scans and their layouts ship inside the
package, under `bitikocr/data/synthetic/assets/`. Point the pipeline at your own
with `--fonts-dir`, or with the `BITIKOCR_FONTS_DIR`,
`BITIKOCR_BACKGROUNDS_DIR`, `BITIKOCR_LAYOUTS_DIR` and `BITIKOCR_OUTPUT_DIR`
environment variables.

Only a font that can render every character of a page is used, so adding a
Latin-only font will not break Cyrillic documents.

## Going further

[`bitikocr/data/synthetic/README.md`](bitikocr/data/synthetic/README.md) is the
developer's guide to the generator: what every module owns, how to add a
font, a form variant or a whole document type, and the gotchas worth knowing
before changing anything.

## Development

```bash
make checks
```

Runs isort, black, ruff, mypy and pytest. Every test lives in `tests/` at
the repository root, one module per module of the package and named after
it. See [CODING_STYLE.md](CODING_STYLE.md)
for the rules and [ARCHITECHTURE.md](ARCHITECHTURE.md) for where new code goes.
