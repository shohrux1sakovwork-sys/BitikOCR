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

Documents that fill a pre-printed form — the death certificate today, more
later — take their geometry from a **layout JSON**, not from Python. Each
layout is the measured position of every printed underline on one blank
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

`text_type` decides what each entry is:

| `text_type` starts with | Becomes                                        |
|-------------------------|------------------------------------------------|
| `digits`                | a handwritten field, written in a digit hand   |
| `round_stamp`           | the office seal's area                         |
| `printed_digits`        | the machine-printed serial number's area       |
| `signature`             | where the registrar signs                      |
| anything else           | a handwritten text field                       |

Entries whose id ends in `_line1`, `_line2`, ... are one logical field
written across several printed rules, merged under the shared base name
(`cause_of_death_line1` + `cause_of_death_line2` → `cause_of_death`).

**To add a form variant**, drop its scan in
`bitikocr/synthetic/assets/backgrounds/` and its measured layout in
`bitikocr/synthetic/assets/layouts/<name>.json`, then:

```bash
uv run bitikocr synth generate death_certificate --template <name> -n 20 --boxes
```

No code changes are needed. The layout's `width`/`height` must match the
background scan exactly, since every coordinate is scaled from them.

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
