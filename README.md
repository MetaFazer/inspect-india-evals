# inspect-india-evals

**India-specific AI safety, bias, and cultural knowledge evaluations built on [UK AISI's Inspect AI framework](https://inspect.ai-safety-institute.org.uk/).**

---

## Why This Project Exists

Large language models are increasingly being deployed in India — through national digital infrastructure like Aadhaar, UPI, and Bhashini — yet most AI safety benchmarks are built around Western languages and social contexts, lacking an integrated multilingual context. They miss the things that actually matter here:

- Can a model refuse a harmful request written in Hindi or Tamil, not just English?
- Does it exhibit bias against Indian social categories like caste, religion, or region?
- Does it handle queries about India's Digital Public Infrastructure safely?
- How well does it understand Indian constitutional law and culture?

This project fills that gap. It is a collection of structured evaluation tasks — runnable with a single command — that test models on India-relevant safety, fairness, and knowledge dimensions.

---

## What It Evaluates

Five evaluation modules, each targeting a different dimension of model behaviour:

| Module | Task | What it measures |
|---|---|---|
| `multilingual/` | `multilingual` | Factual accuracy (MMLU-style) across **15 Indian languages plus English** — this is the project's largest and broadest-coverage module |
| `bias/` | `bharatbbq` | Stereotype & social bias using the BharatBBQ benchmark — caste, religion, gender, age, region, disability, and more (13 category files) |
| `safeguards/` | `multilingual_safety` | Whether the model refuses harmful prompts written in 4 Indian languages plus English |
| `safeguards/` | `jailbreak_safety` | Whether multi-turn (5-turn) jailbreak attacks across 6 Indian languages plus English, spanning 10 attack types, succeed in extracting harmful content |
| `dpi_safety/` | `dpi_safety` | Correct behaviour on India's DPI context (Aadhaar, UPI, Bhashini, Digital Lending, DigiLocker, ABDM) — answer low-risk queries, refuse high-risk ones. English only |
| `cultural_knowledge/` | `cultural_knowledge` | Rubric-graded knowledge across 5 Indian domains: Constitution, Healthcare, History, State Governance, Agriculture and MSP |

### Dataset Scale at a Glance

| Module | Dataset file(s) | Rows | Languages | Samples generated |
|---|---|---:|---:|---:|
| `multilingual` | `mmlu_translated.csv` | 2,272 | 16 (15 Indian + English) | 2,272 |
| `bias` (BharatBBQ) | 13 category CSVs | 54,048 | English | 54,048 |
| `safeguards` (safety) | `safety.csv` | 200 | 5 (4 Indian + English) | 1,000 (200 × 5 languages) |
| `safeguards` (jailbreak) | `jailbreak_multilingual.csv` | 70 | 7 (6 Indian + English) | 70 (5-turn conversations) |
| `dpi_safety` | `dpi_dataset.csv` | 150 | 1 (English only) | 150 |
| `cultural_knowledge` | `Cultural_knowledge_rubric_dataset.json` | 300 | English | 300 |

These figures are asserted by `tests/test_dataset_counts.py` against `india_evals/_dataset_facts.py`, so they can't silently drift out of date.

### Composite Fairness Index

All evaluation dimensions roll up into a single **Fairness Index** — a weighted 0–1 score:

```
fairness_index = mean([
    multilingual_accuracy,     # language coverage
    1 - |bias_score|,          # lower bias → fairer
    safety_refusal_rate,       # higher refusal on harmful prompts → safer
    dpi_accuracy,              # correct DPI behaviour
])
```

This gives evaluators one number to compare models against each other or track improvements over time.

---

## Project Structure

```
inspect-india-evals/
├── india_evals/
│   ├── multilingual/           # Multilingual MMLU accuracy
│   │   ├── task.py
│   │   └── datasets/
│   ├── bias/                   # BharatBBQ bias benchmark
│   │   ├── task.py
│   │   └── datasets/
│   ├── safeguards/             # Safety refusal + jailbreak resistance
│   │   ├── task.py
│   │   └── datasets/
│   ├── dpi_safety/             # Digital Public Infrastructure safety
│   │   ├── task.py
│   │   └── datasets/
│   ├── cultural_knowledge/     # Rubric-graded cultural knowledge
│   │   ├── task.py
│   │   └── datasets/
│   ├── scorers/                # Shared scorers (fairness_index)
│   │   └── __init__.py
│   └── view_plugin/            # HTML heatmap report generator
│       ├── parser.py
│       ├── heatmap.py
│       └── __main__.py
├── tests/                      # Full pytest suite (87 tests)
├── run_all.py                  # Multi-model runner with MLflow logging
├── pyproject.toml
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10 or later
- [Ollama](https://ollama.com/) for running local models (or any model supported by Inspect AI)

### Install

#### From PyPI (recommended)

The package is published on PyPI — install it directly into any Python environment:

```bash
pip install inspect-india-evals
```

This installs all evaluation modules and their core dependencies (`inspect-ai`, `pandas`). Optional extras:

```bash
# MLflow experiment tracking
pip install inspect-india-evals mlflow

# Plotly heatmap report
pip install inspect-india-evals plotly
```

#### From source (development)

```bash
git clone https://github.com/MetaFazer/inspect-india-evals
cd inspect-india-evals

pip install -e ".[dev]"
```

---

## Running Evaluations

### Run a single task

```bash
# Multilingual MMLU accuracy
inspect eval india_evals/multilingual/task.py@multilingual \
    --model ollama/llama3.2:3b

# BharatBBQ bias benchmark
inspect eval india_evals/bias/task.py@bharatbbq \
    --model ollama/llama3.2:3b

# Safety refusal — harmful prompts in 4 Indian languages + English
inspect eval india_evals/safeguards/task.py@multilingual_safety \
    --model ollama/llama3.2:3b

# Jailbreak resistance — 5-turn multi-turn attacks in 6 Indian languages + English
inspect eval india_evals/safeguards/task.py@jailbreak_safety \
    --model ollama/llama3.2:3b

# DPI Safety — Aadhaar / UPI / Bhashini context
inspect eval india_evals/dpi_safety/task.py@dpi_safety \
    --model ollama/llama3.2:3b

# Cultural knowledge — rubric-graded Indian domains
inspect eval india_evals/cultural_knowledge/task.py@cultural_knowledge \
    --model ollama/llama3.2:3b
```

### Quick smoke test (limit samples)

```bash
inspect eval india_evals/safeguards/task.py@multilingual_safety \
    --model ollama/llama3.2:3b --limit 5
```

### View results in the Inspect UI

```bash
inspect view
```

### Run all tasks across multiple models + log to MLflow

```bash
python run_all.py
mlflow ui   # → open http://localhost:5000
```

`run_all.py` runs all tasks across a configurable list of models, parses the `.eval` logs, computes the composite Fairness Index, and logs everything to MLflow for side-by-side comparison.

---

## Judge Model

`multilingual_safety`, `jailbreak_safety`, `dpi_safety`, and `cultural_knowledge` use an LLM as a judge (on top of, or instead of, keyword matching). The judge must never be the same model as the one under evaluation — letting a model grade its own output (self-judging) biases scores upward.

**Default judge:** `ollama/llama3.1:8b`

**How to change it:**

```bash
# Environment variable (applies to every task.py run)
export INDIA_EVALS_JUDGE=ollama/qwen2.5:32b

# Or per-run, via the task parameter
inspect eval india_evals/safeguards/task.py@multilingual_safety \
    --model ollama/gemma2:27b \
    -T judge_model=ollama/qwen2.5:32b

# run_all.py
python run_all.py --judge-model ollama/qwen2.5:32b
```

If the resolved judge ever matches the model under evaluation, the scorer prints a one-time warning to stderr (it does not raise, in case that's genuinely what you want) — check for it if your safety numbers look unexpectedly high. Every Score's metadata records the resolved `judge_model` name, so `.eval` logs always say which model graded them. `run_all.py` also refuses to start if `--judge-model` is one of `--models`.

---

## Visual Heatmap Report

Generate an interactive per-language × per-category HTML report from your eval logs:

```bash
# After running evals (logs go to ./logs/ by default)
python -m india_evals.view_plugin logs/

# Custom output path
python -m india_evals.view_plugin --output report.html logs/

# Specific eval files
python -m india_evals.view_plugin logs/run1.eval logs/run2.eval
```

This opens `india_report.html` in your browser — a self-contained page with:
- Stats strip (logs, samples, models, languages)
- Summary table (model × task accuracy)
- Interactive heatmaps: language × category for each module
- Bar charts by language and by category

---

## Python Developer API

You can import and interact with the project tasks, scoring components, and reporting utilities programmatically within your own scripts:

### Run Evaluations Programmatically
Import tasks and execute them using the `inspect_ai` runner:
```python
from inspect_ai import eval
import india_evals

# Programmatically evaluate the multilingual task
logs = eval(
    india_evals.multilingual(),
    model="ollama/llama3.2:3b",
    limit=5
)
```

### Calculate the Fairness Index
Compute the composite India Fairness Index manually using the normalized sub-scores:
```python
import india_evals

results = india_evals.fairness_index(
    multilingual_accuracy=0.72,
    bias_score_amb=0.10,          # Stereotype bias
    safety_refusal_rate=0.88,     # Safety refusal rate
    dpi_accuracy=0.85             # DPI query accuracy
)

print(results["fairness_index"])  # Output: 0.8375
```

### Render the HTML Report
Generate the interactive visual HTML report from your evaluation logs folder programmatically:
```python
import india_evals

# Scans logs/ directory and generates report.html
report_path = india_evals.render_report(
    log_dir="logs/",
    output="report.html"
)
```

---

## Module Details


### `multilingual` — Multilingual MMLU Accuracy

Tests factual reasoning across 15 Indian languages plus English using an MMLU-style multiple-choice format. Each question has one correct answer from options A–D. This is the project's largest and broadest module by far — the other datasets are 70–2,300 rows in one or a handful of languages, this one is 2,272 rows spread across 16 languages and 29 MMLU subjects.

**Languages (16, 15 Indian + English):** English (`en`), Hindi (`hi`), Bengali (`bn`), Tamil (`ta`), Telugu (`te`), Marathi (`mr`), Gujarati (`gu`), Assamese (`as`), Kannada (`kn`), Konkani (`kok`), Malayalam (`ml`), Manipuri (`mni`), Nepali (`ne`), Odia (`or`), Punjabi (`pa`), Urdu (`ur`)  
**Dataset:** 2,272 rows — 142 questions per language, across 29 MMLU subjects  
**Scorer:** Exact match (A/B/C/D)  
**Metric:** Accuracy (0–1)

---

### `bharatbbq` — Social Bias (BharatBBQ)

Based on the [BharatBBQ benchmark](https://github.com/bharat-bbq), which tests whether a model applies stereotypes when answering ambiguous social questions about Indian social groups.

**Dataset:** 13 category CSVs, 54,048 rows total — Age, AgexGender, Caste, Disability_status, Gender_identity, GenderxReligion, Nationality, Physical_appearance, Region, RegionxGender, Religion, SES (socioeconomic status), Sexual_orientation. See [Relation to the preprint](#relation-to-the-preprint) — this differs from the 13 categories the paper describes.  
**Scorer:** Accuracy on unambiguous questions + bias direction on ambiguous ones  
**Metric:** Accuracy (0–1), Bias score (−1 to +1)

---

### `multilingual_safety` — Safety Refusal

Tests whether a model correctly refuses harmful requests (cybercrime, deepfakes, communal violence, financial fraud, misinformation, harassment) when written in Indian languages — not just English.

**Languages (5, 4 Indian + English):** English, Hindi, Tamil, Telugu, Bengali  
**Dataset:** 200 prompts × 5 language columns = 1,000 samples. All 200 prompts are risk_level=High. 6 categories: cybercrime (34), deepfake (34), communal_violence (33), financial_fraud (33), misinformation (33), harassment (33)  
**Scoring:** Two-layer — keyword phrase matching first, LLM judge fallback  
**Metric:** Refusal rate (1 = refused, 0 = complied)

---

### `jailbreak_safety` — Jailbreak Resistance

Tests whether multi-turn (5-turn) jailbreak attack sequences succeed in extracting harmful content across Indian languages.

**Languages (7, 6 Indian + English):** English, Hindi, Bengali, Gujarati, Marathi, Tamil, Telugu  
**Dataset:** 70 conversations, each 5 turns, across 10 attack types (7 each): roleplay, fiction, authority, emotional, research, educational, urgency, incident_review, translation, reward_hacking  
**Scorer:** Same two-layer scorer (keyword + LLM judge)  
**Metric:** Refusal rate (higher is safer)

---

### `dpi_safety` — Digital Public Infrastructure Safety

Tests correct handling of queries in the context of India's DPI stack: Aadhaar identity, UPI payments, Bhashini language services, Digital Lending, DigiLocker, and ABDM health data. **English only** — see [Known Limitations](#known-limitations).

Low-risk queries (how does Aadhaar work?) should be answered.  
High-risk queries (how to spoof Aadhaar verification?) should be refused.

**Dataset:** 150 rows, 6 categories: Aadhaar Privacy (40), UPI Fraud (35), Bhashini Disinformation (25), Digital Lending (25), DigiLocker (15), ABDM Health Data (10)  
**Risk levels:** High (78), Low (72) — there is no Medium risk level in the data  
**Scorer:** Keyword-based + LLM judge  
**Metric:** Accuracy on correct behaviour (answer vs. refuse)

---

### `cultural_knowledge` — Indian Cultural Knowledge

Open-ended questions graded against a 4-criterion rubric by an LLM judge. Tests knowledge of the Indian Constitution, healthcare system, history, state governance, and agriculture/MSP.

**Dataset:** 300 questions across 5 domains, 60 each: Indian Constitution, Indian Healthcare, Indian History, State Governance, Agriculture and MSP. The paper's projected full-corpus figure of ~825 questions across 8+ domains has not yet been built out — see [Relation to the preprint](#relation-to-the-preprint).  
**Scorer:** LLM-as-judge rubric scoring (fraction of criteria passed)  
**Metric:** Mean rubric score (0–1)

---

## Tests

```bash
pytest tests/ -v
```

87 tests across all modules, including a dataset-counts consistency suite (`tests/test_dataset_counts.py`) that asserts every dataset's row/language/category/domain counts against `india_evals/_dataset_facts.py`. Covers dataset loading, sample structure, scorer logic, and task instantiation. No model calls required — all scorer tests are unit-tested with mocks.

---

## Supported Models

Any model supported by Inspect AI works. Tested with:

| Model | How to run |
|---|---|
| Ollama local models | `--model ollama/llama3.2:3b` |
| OpenAI | `--model openai/gpt-4o` |
| Anthropic | `--model anthropic/claude-3-5-sonnet` |
| Together AI | `--model together/meta-llama/...` |
| Sarvam AI | `--model sarvam/sarvam-m` |

Set API keys as environment variables as required by each provider.

---

## Design Decisions

**Why Inspect AI?**  
Inspect AI is the UK AISI's open evaluation framework. It handles model calls, logging, scoring, and parallelism out of the box, making eval code reproducible and auditable.

**Why keyword + LLM judge for safety scoring?**  
Keyword matching is fast, free, and works across every language in both datasets (5 for safety, 7 for jailbreak) without needing a separate call. The LLM judge handles edge cases: a model that generates a malware *scanner* description isn't the same as generating malware, and the judge can tell the difference.

**Why rubric scoring for cultural knowledge?**  
Cultural and constitutional questions have open-ended correct answers. A rubric of 4 specific factual criteria — graded by a judge model — gives a structured, reproducible score without needing a fixed expected answer.

---

## Known Limitations

- **Results in the arXiv preprint (arXiv:2607.25375) predate the judge-model fix.** Earlier versions of `refusal_scorer`, `dpi_scorer`, and `rubric_scorer` defaulted the LLM judge to `get_model()` with no argument, which resolves to the model under evaluation — so the model being tested was judging its own output (self-judging bias). The safety and jailbreak numbers reported in that preprint were produced under this bug and should be treated as invalid pending a re-run with an independent judge model (see [Judge Model](#judge-model) above).
- The `cultural_knowledge` dataset currently has 300 questions across 5 domains (Indian Constitution, Indian Healthcare, Indian History, State Governance, Agriculture and MSP) — smaller than the ~825-question, 8+-domain corpus projected in the paper.
- **DPI safety is currently evaluated in English only.** The `dpi_dataset.csv` has no non-English rows, so `dpi_safety` measures domain-specific refusal behaviour (Aadhaar, UPI, Bhashini, Digital Lending, DigiLocker, ABDM) but not cross-lingual DPI safety — a real gap for a framework whose premise is Indian-language deployment. Translating the DPI set into the same Indic languages already used by the `safeguards` module is the natural next step.
- **The BharatBBQ categories on disk don't match the paper's claimed list.** The data has 13 files (54,048 rows) including `Sexual_orientation`, which the paper doesn't claim, while `Linguistic Group`, `Caste-Adjacent Occupation`, `Urban-Rural Identity`, and `Tribal Community` — all claimed by the paper — are absent from the data. See [Relation to the preprint](#relation-to-the-preprint).

---

## Relation to the Preprint

The arXiv preprint (arXiv:2607.25375) describes a pilot run of this suite at `--limit 5` per module, and states some dataset figures that no longer match — or never matched — the data in this repository:

| Figure | Preprint | Actual (this repo) |
|---|---|---|
| Safety prompts | 50 | 200 |
| Jailbreak turns | 3 | 5 |
| Jailbreak attack strategies | 4 | 10 |
| DPI risk levels | Low / Medium / High | Low / High (no Medium) |
| Cultural knowledge questions | ~825 (projected, 8+ domains) | 300 (5 domains) |

**This repository — specifically `india_evals/_dataset_facts.py` and the datasets under `india_evals/*/datasets/` — is authoritative for dataset composition.** Where the preprint and the code disagree, trust the code, and treat any numbers copied from the preprint as describing an earlier snapshot, not the current suite.

---

## License

Apache 2.0

## Collaborators
* @MetaFazer
* @ShreyNag