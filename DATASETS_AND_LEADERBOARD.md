# BitikOCR: Datasets and Leaderboard

How each training dataset was built, what each version added, and how every
model we have tried scores on the real Adliya benchmark.

**Best model so far:** Qwen3-VL-8B fine-tuned on v3 + v4 + real pages at
2 megapixels: **56.4% of facts correct, 34.2% CER**, against 33.7% and 60.7%
for the same model untuned.

## 1. The benchmark

All scores below come from **`aktrmai/uzbek-htr-bench`**: 175 real
handwritten pages from the Adliya archive, never used for training.

| Document type | Pages | | Document type | Pages |
|---|---|---|---|---|
| `ariza` (application) | 105 | | `rozilik_xati` (consent letter) | 7 |
| `malumotnoma` (certificate of information) | 41 | | `dalolatnoma` (act) | 4 |
| `olim_guvohnomasi` (death certificate) | 7 | | others | 11 |

Each page has a full transcription and a list of human-reviewed **facts**:
the names, dates, numbers, key phrases and document type that a correct
transcript must contain. The metrics:

- **Facts:** share of those facts found in the model's transcript, overall
  and per kind. Some kinds must match exactly, others match at 85%
  similarity or more. This is the main metric; checkpoints are chosen by it.
- **CER:** character error rate of the whole transcript against the
  reference. Lower is better; above 100% means the output is longer than
  the page and mostly wrong.

Transcripts follow the archive's conventions: `<signature>` for a
signature, `<stamp>` … `</stamp>` around a stamp's text, `<deleted>` …
`</deleted>` for crossed-out text, and `[...]` for illegible text.

## 2. Datasets

All datasets are private on Hugging Face under `aktrmai`. Every synthetic
page is generated, with exact ground truth; no name or number on it belongs
to a real person.

| Version | Hugging Face | Pages | Main idea |
|---|---|---|---|
| v1 | `aktrmai/synthetic_mock` | 20,000 | First generator |
| v2 | `aktrmai/synthetic_v2` | 5,000 | Labels in the archive's conventions |
| v3 | `aktrmai/synthetic_v3` | 5,000 | Names written the way clerks write them |
| v4 | `aktrmai/synthetic_v4` | 5,000 | Pages that look like the archive's scans |
| Malumotnoma | `aktrmai/synthetic_malumotnoma` | 1,000 | Mahalla certificates of residence, a new document type |
| Real | `aktrmai/adliya-handwritten-ocr` | 96 | Human-transcribed 2023 applications |

v1–v4 each have five document types, 1,000 pages each: `ariza`,
`consent_letter`, `explanatory_letter`, `birth_certificate` and
`death_certificate`. The malumotnoma set adds a sixth, also 1,000 pages.
Files are matched by id:

```
images/<id>.jpg         the page
annotations/<id>.json   transcription record: target text, parts, metadata
facts/<id>.json         structured facts that must appear in the text
index.jsonl             one line per page, with its text and render details
```

### v1: `synthetic_mock`

The first generator: 20 handwriting fonts and plain transcriptions without
the archive's tags. Split 18,000 for training and 2,000 for evaluation;
`python load.py` downloads the images for `data/train.jsonl`.

**Lesson:** 18,000 pages scored *worse* on real pages than 5,000 pages of
v2. Labels that match the target conventions matter more than volume.

### v2: `synthetic_v2`

- **Labels in the benchmark's conventions:** forms transcribe their printed
  titles and labels, stamps are `<stamp>` … `</stamp>` blocks, signatures
  are `<signature>`, and a two-page form is read one sheet after the other.
- Three in four applications carry the receiving office's incoming stamp,
  with its filing number and date written in by hand.
- **42 handwriting fonts** instead of 20; most strokes redrawn as a
  ballpoint line with uneven pressure.
- Captured as scans (76%) or phone photos (24%), with blur, folds, stains,
  shadows, photocopies and low resolution.

### v3: `synthetic_v3`

v2 taught the model to "correct" real names into the few hundred it had
seen: Ибрагимов became Иброҳимов. v3 writes names as the clerks do:

- **A wide name pool:** 276 male and 184 female given names and 361 surname
  stems, Khorezm's among them, instead of 90, 82 and 79.
- **Each writer's own spelling:** 45% keep the standard spelling; the rest
  write Uzbek Cyrillic with Russian letters (х к г у for ҳ қ ғ ў) and
  Russian name forms (Ибрагимов, Ахмедова), consistently across a page.
  Latin writers type the tutuq as `'`, `‘` or `` ` ``.

### v4: `synthetic_v4`

Set from measurements of the 175 benchmark pages and the 96 real Adliya
pages:

| | Real pages | v4 | v3 |
|---|---|---|---|
| Paper brightness (median) | 254–255 | 250 | 207 |
| Grey-paper pages | 6–41% | 28% | cream, all |
| Pages photographed on a table | 0% | 0% | 20% |
| Ink blueness (B − R) | 19–27 | 22 | 8 |
| Line step (share of page width) | 0.045–0.048 | 0.044 | 0.058 |

- **The archive's scanner:** flat scans only, paper levelled to white or a
  light neutral grey, ink colour dulled, no stains, little dust.
- **Punch holes** down the left edge on about half the pages, never over
  text.
- **Closer lines:** line spacing 0.85–1.2 times the letter size instead of
  1.15–1.6.
- **One broken font removed** (its letters rendered as dots); 41 remain.

### Malumotnoma: `synthetic_malumotnoma`

`malumotnoma` is 41 of the 175 benchmark pages, the second most common
type, and our best model still reads it at about 46% CER against 23% for
`ariza`. Each mahalla committee prints its own blank, so the generator draws
one per page, then fills and labels it like the certificates:

- The title МАЪЛУМОТНОМА, certifying sentences worded several ways with
  blanks for holder, birth year, town, mahalla, street and house, and 3–14
  numbered family rows (name, birth year, relation). Empty rows keep their
  printed numbers, as in the real labels.
- The committee's rectangular stamp with date and number written in (or a
  printed letterhead), the round seal over the chairman's signature, and
  typeset names for the chairman and secretary.
- A plausible household: in-laws and grandchildren only for older holders,
  parents for younger ones. 92% Cyrillic handwriting, like the real forms.
- 27% of committees have Cyrillic stamps, so 74.7% of pages contain Latin
  text, against 75.6% of the real certificates.
- **Families are named the Uzbek way:** a child usually takes the paternal
  grandfather's given name as a surname (Qodirov Ali → Muhammadjonov Farrux
  → Aliyev Hamid → Farruxov Jamshid), some families keep one surname, and a
  wife takes her husband's surname or keeps her own. Birth certificates
  built from commit `863f86b` on follow the same rules; v2–v4 were built
  before, with one surname per family.

### Real: `adliya-handwritten-ocr`

96 human-reviewed 2023 applications from Qashqadaryo, Buxoro and Jizzax, in
many hands; 82 pages contain Cyrillic, 46 Latin. None overlaps the
benchmark (checked by text and image similarity).

The reviewers wrote notes in brackets (`[имзо]`, `[Штамп: ...]`), so
`scripts/prepare_mixed_data.py` rewrites them in the benchmark's tags
before training, and stops on any note it has no rule for:

| Reviewers' note | Becomes |
|---|---|
| `[имзо]`, `[imzo]`, `[қолы]` | `<signature>` |
| `[Штамп: …]`, `[Муҳр: …]` | `<stamp>` … `</stamp>` (with `...` → `[...]`) |
| `[Муҳр]` | `<stamp/>` |
| `[ўчирилган]` | `<deleted></deleted>` |
| `[Резолюция: …]`, `[Қўлда: …]` | the text alone |
| `[Фото санаси: …]`, `[Орқа фонда …]` | removed |

## 3. How to create the datasets

The generator lives on the **`data`** branch; training code on
**`train-code`**. Each synthetic set rebuilds exactly from its commit and
seed:

```bash
# On the data branch
git checkout <commit>
uv run --extra data python scripts/data/build_corpus.py all \
    -o synthetic-vN --per-type 1000 --seed <seed> [--look archive]
```

| Version | Commit (`data` branch) | Seed | Extra option |
|---|---|---|---|
| v2 | `04870f6` | 2026 | |
| v3 | `f6ffbd9` | 2027 | |
| v4 | `367b86b` | 2029 | `--look archive` |
| Malumotnoma | `961e2bc` | 2031 | `--look archive --types malumotnoma` |

5,000 pages take about 6 minutes on 36 CPU cores. The build checks every
page and reports problems at the end.

## 4. How to train

```bash
# On the train-code branch
uv sync --extra train
uv run hf auth login

# Datasets
uv run hf download aktrmai/synthetic_v3 --repo-type dataset --local-dir data/synthetic-v3
uv run hf download aktrmai/synthetic_v4 --repo-type dataset --local-dir data/synthetic-v4
uv run hf download aktrmai/adliya-handwritten-ocr --repo-type dataset --local-dir data/adliya-real
uv run hf download aktrmai/uzbek-htr-bench --repo-type dataset --local-dir data/uzbek-htr-bench

# Mix: real pages repeated to 20% of the rows (12,496 rows)
uv run python scripts/prepare_mixed_data.py \
    --synthetic data/synthetic-v3/index.jsonl data/synthetic-v4/index.jsonl \
    --repeat 26 --output data/mix-v3-v4-adliya

# Train: LoRA r=64, 500 steps of 8 pages, learning rate 1e-4,
# evaluation on the benchmark every 100 steps, best checkpoint kept
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run --locked --extra train python scripts/run_full_real_eval.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --data data/mix-v3-v4-adliya/index.jsonl --image-folder data \
    --image-max-pixels 2007040 --max-steps 500 --eval-steps 100 \
    --save-only-model --output outputs/full/<run-name> --group <group> --gpu 0
```

To score a model without training, add `--evaluation-only`, and
`--adapter <folder>` for a trained adapter. With more real pages, keep them
near 20% of the rows: `--repeat` ≈ 2,500 ÷ number of real pages.

**Resolution matters.** `--image-max-pixels 2007040` (2 MP) shows an A4
page at about 1190 × 1680 pixels; the old default of 1 MP (about 100 dpi)
lost small letters and diacritics. Evaluate an adapter at the resolution it
was trained at.

## 5. Leaderboard

175 real benchmark pages, ranked by the average of the five fact columns.
⭐ marks our models; rows without it come from the team's comparison tool.

| # | Model | Base | Training data | Name | Date | Number | Phrase | Doc_Type | **Avg** | Facts (all) | CER |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ⭐ **v3 + v4 + real, 2 MP** (step 300) | **Qwen3-VL-8B** | v3 + v4 + 96 real ×26 | **46.3** | **62.5** | **53.2** | **47.9** | 94.3 | **60.8** | **56.4** | **34.2** |
| — | ⭐ v3 + v4 + malumotnoma (fixed build²) + real, 2 MP | Qwen3-VL-8B | v3 + v4 + malumotnoma + 96 real ×29 | 49.2 | 58.8 | 48.3 | 49.3 | 93.0 | 59.7 | 55.9 | 38.2 |
| — | ⭐ v3 + v4 + malumotnoma (first build²) + real, 2 MP | Qwen3-VL-8B | v3 + v4 + malumotnoma + 96 real ×29 | 47.3 | 58.1 | 49.3 | 52.8 | 93.0 | 60.1 | 56.0 | 38.2 |
| 2 | ⭐ v3 + v4 + real, 2 MP (step 300) | Qwen2.5-VL-7B | v3 + v4 + 96 real ×26 | 44.4 | 54.4 | 50.7 | 42.1 | 93.0 | 56.9 | 52.3 | 34.4 |
| 3 | ⭐ v3 + v4 + real, 1 MP model read at 2 MP | Qwen2.5-VL-7B | v3 + v4 + 96 real ×26 | 43.2 | 50.0 | 49.3 | 42.4 | 95.6 | 56.1 | 51.1 | 37.8 |
| 4 | ⭐ v3 + v4 + real, 1 MP (step 500) | Qwen2.5-VL-7B | v3 + v4 + 96 real ×26 | 39.9 | 53.1 | 44.3 | 36.6 | 94.3 | 53.6 | 48.7 | 38.7 |
| 5 | ⭐ v4 + real (step 300) | Qwen2.5-VL-7B | v4 + 96 real ×13 | 39.1 | 51.9 | 39.3 | 38.6 | 93.0 | 52.4 | 47.7 | 45.0 |
| 6 | ⭐ v3 + real, 500 steps | Qwen2.5-VL-7B | v3 + 96 real ×13 | 39.5 | 51.6 | 39.8 | 33.1 | **97.5** | 52.3 | 47.2 | 41.8 |
| 7 | ⭐ v3 + real, 250 steps | Qwen2.5-VL-7B | v3 + 96 real ×13 | 38.9 | 51.9 | 43.3 | 33.8 | 93.0 | 52.2 | 47.2 | 40.9 |
| 8 | ⭐ v3 only (step 200) | Qwen2.5-VL-7B | v3 | 40.1 | 31.2 | 31.8 | 27.9 | 91.8 | 44.6 | 40.2 | 49.2 |
| 9 | qwen3_vl_32b | Qwen3-VL-32B | none | 25.0 | 31.6 | 48.1 | 28.9 | 72.1 | 41.1 | — | — |
| 10 | ⭐ v2 only | Qwen2.5-VL-7B | v2 | 21.8 | 33.4 | 38.3 | 20.0 | 89.9 | 40.7 | 33.7 | 59.4 |
| 11 | qwen3_8_27b | Qwen3-VL-8B | none | 33.1 | 31.6 | 41.1 | 29.9 | 54.4 | 38.0 | — | — |
| 12 | ⭐ Untuned, our pipeline | Qwen3-VL-8B | none | 29.4 | 30.6 | 34.3 | 25.9 | 66.5 | 37.3 | 33.7 | 60.7 |
| 13 | qwen2_5vl_32b | Qwen2.5-VL-32B | none | 28.6 | 25.6 | 36.1 | 21.3 | 66.5 | 35.6 | — | — |
| 14 | qwen2_5vl_7b_v2 | Qwen2.5-VL-7B | — | 32.1 | 25.0 | 41.1 | 19.6 | 54.4 | 34.4 | — | — |
| 15 | qwen2_5vl_7b | Qwen2.5-VL-7B | none | 30.9 | 23.1 | 36.6 | 21.6 | 55.1 | 33.5 | — | — |
| 16 | ⭐ Untuned, our pipeline | Qwen2.5-VL-7B | none | 27.2 | 20.0 | 23.9 | 13.4 | 58.9 | 28.7 | 25.8 | 97.8 |
| 17 | ⭐ First ablation adapter (r64) | Qwen2.5-VL-7B | v1, 1.6k × 3 epochs | 21.0 | 30.3 | 16.9 | 14.5 | 60.8 | 28.7 | 25.5 | 58.3 |
| 18 | richardyoung_olmocr2_7b_q8 | olmOCR-2-7B | none | 27.8 | 29.4 | 22.8 | 17.9 | 44.3 | 28.4 | — | — |
| 19 | qwen2_5vl_3b | Qwen2.5-VL-3B | none | 24.9 | 18.8 | 29.2 | 15.1 | 50.0 | 27.6 | — | — |
| 20 | glm_ocr_latest | GLM-OCR | none | 13.7 | 19.4 | 37.1 | 16.8 | 49.4 | 27.3 | — | — |
| 21 | ⭐ v1, 18k pages (step 1350) | Qwen2.5-VL-7B | v1 18k | 17.1 | 26.2 | 14.4 | 12.4 | 63.3 | 26.7 | 22.8 | 58.6 |
| 22 | glm_ocr | GLM-OCR | none | 12.7 | 20.6 | 34.7 | 15.5 | 48.1 | 26.3 | — | — |
| 23 | wevisdoc_4b | — | none | 11.5 | 10.3 | 24.3 | 12.7 | 31.6 | 18.1 | — | — |
| 24 | deepseek_ocr_latest | DeepSeek-OCR | none | 3.3 | 8.1 | 9.4 | 3.8 | 19.0 | 8.7 | — | — |
| 25 | paddleocr_ru | PaddleOCR | none | 5.1 | 0.3 | 1.0 | 0.0 | 22.2 | 5.7 | — | — |
| 26 | yasserrmd_nanonets_ocr_s_latest | Nanonets-OCR-s | none | 1.2 | 1.6 | 1.0 | 0.7 | 1.3 | 1.2 | — | — |

² The first malumotnoma build read the title before the stamp and merged it
into letterhead lines; the model learned that order and read real
certificates worse (malumotnoma CER 51.7% against 45.9%). The fixed build
(the set on Hugging Face) still scores 49.8%. Ignoring order, it reads the
certificates slightly better (word overlap 66.3% against 64.7%, and names
49.2% against 46.3%), but it places the top stamp's text later than the
benchmark does, which CER punishes heavily. Row 1 remains the best model.

- **Avg** is the plain average of the five fact columns. **Facts (all)**
  weights every fact equally, so it differs from Avg.
- Gemini 3.6 Flash is left out: it did not run on all pages.
- The team's tool measures slightly differently from our pipeline: the same
  untuned Qwen2.5-VL-7B scores 33.5 there and 28.7 with us. Compare within
  each group first.
- The benchmark both chose the checkpoints and scored them, so the best
  checkpoints are a little optimistic. In both 2 MP runs the score peaked at
  step 300 and then fell 1.5–2 points.

### Trained adapters on Hugging Face

| Adapter | Leaderboard row |
|---|---|
| `aktrmai/bitikocr-qwen3vl-8b-lora-v3-v4-real-2mpx` | 1 (use at 2 MP) |
| `aktrmai/bitikocr-qwen2.5vl-7b-lora-v3-v4-real-2mpx` | 2 (use at 2 MP) |
| `aktrmai/bitikocr-qwen2.5vl-7b-lora-v3-v4-real` | 3, 4 |
| `aktrmai/bitikocr-qwen2.5vl-7b-lora-v3-real` | 7 |
| `aktrmai/bitikocr-qwen2.5vl-7b-lora-r64` | 17 (by its name and scores; the run did not record it) |

## 6. What we learned

Gains on the leaderboard's Avg, largest first:

| Change | Gain |
|---|---|
| v1 → v2: labels in the archive's conventions | +14.0 |
| + 96 real pages (v3 → v3 + real) | +7.6 |
| Qwen2.5-VL-7B → Qwen3-VL-8B as base | +3.9 |
| v2 → v3: clerks' name spellings | +3.9 |
| 1 MP → 2 MP images | +3.3 |
| v3 + v4 together instead of one of them | +1.2 |
| v3 → v4 (scan look) | +0.1 |
| 250 → 500 steps | +0.1 |

- **Real pages are the strongest lever.** 96 pages gave more than any
  synthetic change after v2.
- **Quality and variety beat volume.** 18,000 v1 pages lost to 5,000 v2
  pages; a run of 500 steps sees only 4,000 pages, and scores peak around
  step 300.
- **Read at high enough resolution.** 1 MP was too low for handwriting.
- **Labels must read in the benchmark's order.** A new document type whose
  labels read the top of the page differently made the model worse on that
  very type. Check a new generator's labels against real transcripts before
  training on it.

## 7. Next steps

1. **Train with the new `malumotnoma` set:** v3 + v4 + malumotnoma + real on
   Qwen3-VL-8B, and compare its `malumotnoma` CER with the current 46%.
2. **Label more real pages,** especially `malumotnoma`: real examples of the
   type are still missing from training.
3. **Try Qwen3.5-9B** as the base: on Qwen's model card it beats the larger
   Qwen3-VL-30B-A3B on every document-reading benchmark, and the code
   already supports it.
4. **A separate real test set** of about 100 pages, so checkpoints are not
   chosen and scored on the same benchmark.
