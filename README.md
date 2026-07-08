# inspect-india-evals

**India-specific AI safety, bias, and cultural knowledge evaluations built on [UK AISI's Inspect AI framework](https://inspect.ai-safety-institute.org.uk/).**

---

## Why This Project Exists

Large language models are increasingly being deployed in India — through national digital infrastructure like Aadhaar, UPI, and Bhashini — yet most AI safety benchmarks are built around Western languages and social contexts. They miss the things that actually matter here:

- Can a model refuse a harmful request written in Hindi or Tamil, not just English?
- Does it exhibit bias against Indian social categories like caste, religion, or region?
- Does it handle queries about India's Digital Public Infrastructure safely?
- How well does it understand Indian constitutional law, healthcare, and culture?

This project fills that gap. It is a collection of structured evaluation tasks — runnable with a single command — that test models on India-relevant safety, fairness, and knowledge dimensions.

---

## What It Evaluates

Five evaluation modules, each targeting a different dimension of model behaviour:

| Module | Task | What it measures |
|---|---|---|
| `multilingual/` | `multilingual` | Factual accuracy (MMLU-style) across Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, and English |
| `bias/` | `bharatbbq` | Stereotype & social bias using the BharatBBQ benchmark — caste, religion, gender, region |
| `safeguards/` | `multilingual_safety` | Whether the model refuses harmful prompts written in 5 Indian languages |
| `safeguards/` | `jailbreak_safety` | Whether multi-turn jailbreak attacks (7 languages) succeed in extracting harmful content |
| `dpi_safety/` | `dpi_safety` | Correct behaviour on India's DPI context — answer low-risk queries, refuse high-risk ones |
| `cultural_knowledge/` | `cultural_knowledge` | Rubric-graded knowledge of Indian constitution, healthcare, economy, history, and culture |

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
├── tests/                      # Full pytest suite (63 tests)
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

```bash
# Clone the repo
git clone https://github.com/ac265640/inspect-india-evals
cd inspect-india-evals

# Install dependencies
pip install inspect-ai pandas

# Optional: MLflow for experiment tracking
pip install mlflow

# Optional: Plotly for the visual heatmap report
pip install plotly
```

No `pip install -e .` is required — the tasks are run directly with `inspect eval`.

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

# Safety refusal — harmful prompts in 5 Indian languages
inspect eval india_evals/safeguards/task.py@multilingual_safety \
    --model ollama/llama3.2:3b

# Jailbreak resistance — multi-turn attacks in 7 languages
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

## Module Details

### `multilingual` — Multilingual MMLU Accuracy

Tests factual reasoning across 7 Indian languages using an MMLU-style multiple-choice format. Each question has one correct answer from options A–D.

**Languages:** English, Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati  
**Scorer:** Exact match (A/B/C/D)  
**Metric:** Accuracy (0–1)

---

### `bharatbbq` — Social Bias (BharatBBQ)

Based on the [BharatBBQ benchmark](https://github.com/bharat-bbq), which tests whether a model applies stereotypes when answering ambiguous social questions about Indian social groups.

**Categories:** Caste, Religion, Gender, Age, Region, Physical appearance  
**Scorer:** Accuracy on unambiguous questions + bias direction on ambiguous ones  
**Metric:** Accuracy (0–1), Bias score (−1 to +1)

---

### `multilingual_safety` — Safety Refusal

Tests whether a model correctly refuses harmful requests (cybercrime, violence, fraud, etc.) when written in Indian languages — not just English.

**Languages:** English, Hindi, Tamil, Telugu, Bengali  
**Scoring:** Two-layer — keyword phrase matching first, LLM judge fallback  
**Metric:** Refusal rate (1 = refused, 0 = complied)

---

### `jailbreak_safety` — Jailbreak Resistance

Tests whether multi-turn jailbreak attack sequences (roleplay, hypothetical, escalation) succeed in extracting harmful content across 7 Indian languages.

**Attack types:** Roleplay, Research framing, Hypothetical, Escalating pressure  
**Scorer:** Same two-layer scorer (keyword + LLM judge)  
**Metric:** Refusal rate (higher is safer)

---

### `dpi_safety` — Digital Public Infrastructure Safety

Tests correct handling of queries in the context of India's DPI stack: Aadhaar identity, UPI payments, and Bhashini language services.

Low-risk queries (how does Aadhaar work?) should be answered.  
High-risk queries (how to spoof Aadhaar verification?) should be refused.

**Risk levels:** Low, Medium, High  
**Scorer:** Keyword-based + LLM judge  
**Metric:** Accuracy on correct behaviour (answer vs. refuse)

---

### `cultural_knowledge` — Indian Cultural Knowledge

Open-ended questions graded against a 4-criterion rubric by an LLM judge. Tests knowledge of Indian constitutional law, healthcare system, economy, history, science & technology, geography, and culture.

**Dataset:** ~825 questions across 8+ domains  
**Scorer:** LLM-as-judge rubric scoring (fraction of criteria passed)  
**Metric:** Mean rubric score (0–1)

---

## Tests

```bash
pytest tests/ -v
```

63 tests across all modules. Covers dataset loading, sample structure, scorer logic, and task instantiation. No model calls required — all scorer tests are unit-tested with mocks.

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
Keyword matching is fast, free, and works across all 7 languages without needing a separate call. The LLM judge handles edge cases: a model that generates a malware *scanner* description isn't the same as generating malware, and the judge can tell the difference.

**Why rubric scoring for cultural knowledge?**  
Cultural and constitutional questions have open-ended correct answers. A rubric of 4 specific factual criteria — graded by a judge model — gives a structured, reproducible score without needing a fixed expected answer.

---

## License

Apache 2.0
