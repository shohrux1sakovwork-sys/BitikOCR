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

```bash
uv run bitikocr synth generate ariza -n 100 -o data/synthetic/ariza
```

```bash
uv run bitikocr synth generate birth_certificate -n 100 --boxes
```

```bash
uv run bitikocr synth generate death_certificate -n 100 --boxes
```

Useful flags:

| Flag                | Meaning                                                      |
|---------------------|--------------------------------------------------------------|
| `-n, --count`       | How many pages to render.                                     |
| `-o, --output-dir`  | Where to write them (default `output/<document type>`).       |
| `--seed`            | Make the whole run reproducible.                              |
| `--template`        | Which form variant to fill (form documents only).             |
| `--font`            | Force one handwriting font instead of sampling.               |
| `--fonts-dir`       | Use your own handwriting fonts.                               |
| `--boxes`           | Also write a `_boxes.png` overlay to eyeball the annotations. |
| `-v, --verbose`     | Log every generated sample.                                   |

To see what is available:

```bash
uv run bitikocr synth list-types
```

```bash
uv run bitikocr synth list-fonts
```

```bash
uv run bitikocr synth list-templates
```

## Output format

Each sample is written as `<stem>.png` next to `<stem>.json`, where the stem
carries the document type, the index and the seed:

```
ariza_0000_seed647892279.png
ariza_0000_seed647892279.json
```

The JSON holds the full page transcription, one entry per logical block, one
entry per rendered line, and each with the tight bounding box around the ink
that was actually drawn:

```json
{
  "text": "Гулистон шаҳар ҳокими И. Й. Эрбековга\n...",
  "blocks": [{ "type": "recipient", "text": "...", "bbox": [905, 190, 1620, 350] }],
  "lines":  [{ "block": "recipient", "text": "...", "bbox": [905, 190, 1620, 262] }],
  "size": [1654, 2339],
  "document_type": "ariza",
  "seed": 647892279,
  "font": "Caveat-VariableFont_wght.ttf",
  "style": { "pen": "hard", "slant": 0.21, "...": "..." },
  "fields": { "recipient": "...", "body": "..." }
}
```

Regenerating a page from its seed reproduces it byte for byte.

## Using it from Python

```python
from bitikocr.config import SyntheticConfig
from bitikocr.synthetic import create_generator

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
`bitikocr/synthetic/assets/backgrounds/` and its measured layout in
`bitikocr/synthetic/assets/layouts/<name>.json`, then:

```bash
uv run bitikocr synth generate death_certificate --template <name> -n 20 --boxes
```

No code changes are needed. The layout's `width`/`height` must match the
background scan exactly, since every coordinate is scaled from them.

**To add a whole new form** — a marriage certificate, a passport page — add
its layout and a subclass naming the document type and its default
template; see `bitikocr/synthetic/generators/birth_certificate.py`, which is
twenty lines including the docstring.

## Assets

Handwriting fonts, blank form scans and their layouts ship inside the
package, under `bitikocr/synthetic/assets/`. Point the pipeline at your own
with `--fonts-dir`, or with the `BITIKOCR_FONTS_DIR`,
`BITIKOCR_BACKGROUNDS_DIR`, `BITIKOCR_LAYOUTS_DIR` and `BITIKOCR_OUTPUT_DIR`
environment variables.

Only a font that can render every character of a page is used, so adding a
Latin-only font will not break Cyrillic documents.

## Development

```bash
make checks
```

Runs isort, black, ruff, mypy and pytest. See [CODING_STYLE.md](CODING_STYLE.md)
for the rules and [ARCHITECHTURE.md](ARCHITECHTURE.md) for where new code goes.
