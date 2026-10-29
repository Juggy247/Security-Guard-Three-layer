# Three-Layer LLM Prompt Injection Firewall
 
A three-layer firewall that protects general-purpose LLM deployments against prompt injection attacks. The firewall intercepts user prompts before they reach the model, applies sequential detection layers, and monitors model responses for signs of a successful attack.
 
Diploma thesis project by Thu Htoo Zaw (supervisor: Marek Zachara).

## Origin

This repository extends a two-person course project (v1) written by Thu Htoo Zaw and Thiri Toe Toe Zin ([original repository](PASTE-LINK-HERE)). The v1 firewall is the starting point and the baseline: its decisions on every project dataset are frozen in `data/baselines/v1_decisions.csv`, and each later change is compared against that file.

Added in this repository so far:
- a pytest suite, including a known-issue test for a length shortcut in the classifier
- per-layer details for every scan (rule ids, timings, token counts, classifier probability)
- PostgreSQL logging of live API traffic (`/scan`, `/chat`) with Alembic migrations
- a script that records and compares per-prompt decisions against the v1 baseline
 
## Overview
 
LLMs are increasingly deployed in customer service chatbots, coding assistants, and internal enterprise tools, making their security a critical concern. **Prompt injection** is a class of attack where adversarial text inputs manipulate a model into overriding its intended behavior, extracting hidden configuration, or bypassing safety constraints — without crashing the system, just redirecting it.
 
This project explores how LLMs can be protected against prompt injection through an external, model-agnostic firewall rather than relying solely on model-level safety training.
 
## Architecture
 
```
User prompt
    ↓
Layer 1 — Prompt Sanitizer    (rule-based, instant)
    ↓ passes
Layer 2 — ML Classifier        (fine-tuned DistilBERT)
    ↓ passes
Ollama — Llama 3.2 generates response
    ↓
Layer 3 — Output Filter        (checks response for leakage/compliance)
    ↓ passes
Response shown to user
```
 
**Layer 1 — Prompt Sanitizer**
Rule-based, catches obvious attacks instantly using keyword matching, regex patterns, text normalization, leetspeak detection, and base64 decoding.
 
**Layer 2 — ML Classifier**
Fine-tuned DistilBERT trained on 1,428 labeled samples. Catches contextual attacks the sanitizer misses — jailbreaks, indirect probing, subtle leaking attempts.
 
**Layer 3 — Output Filter**
Checks the LLM's response for system prompt leakage and attack compliance using keyword detection and semantic similarity (sentence-transformers cosine similarity).
 
## Project Structure

```
ThreeLayer_Security_Guard/
├── api/main.py                 # FastAPI backend (/scan, /chat) + request logging
├── three_layer_firewall/
│   ├── sanitizer.py            # Layer 1
│   ├── firewall.py             # pipeline: layers 1 + 2 (+ 3), per-layer details
│   ├── output_filter.py        # Layer 3
│   └── db/                     # models.py (requests table), session.py, logger.py
├── training/                   # classifier.py, evaluate_generalization.py
├── scripts/decisions.py        # record / compare per-prompt decisions vs the v1 baseline
├── alembic/                    # database migrations
├── data/                       # final/ datasets, baselines/, raw/, interim/, scripts/
├── models/                     # trained classifier (git-ignored)
├── frontend/                   # demo.html, demo-no-ollama.html
├── tests/                      # pytest suite
├── docker-compose.yml          # PostgreSQL 17
├── .env.example                # copy to .env
├── requirements.txt
└── requirements-dev.txt
```
 
## Setup

All commands run from the project root.

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

### 2. Start the database

```bash
cp .env.example .env             # then set a password; use the same one in POSTGRES_PASSWORD and DATABASE_URL
docker compose up -d
alembic upgrade head
```

### 3. Install Ollama and pull the model (optional)

The `/chat` endpoint needs [Ollama](https://ollama.com) running locally:

```bash
ollama pull llama3.2
```

### 4. Train the classifier

The trained model is **not included** in this repository (it is large and reproducible). Train it:

```bash
python training/classifier.py
```

It trains on `data/final/dataset_train.csv`, evaluates on `dataset_test.csv` and saves the model to `models/classifier-model/`.

### 5. Run the API

```bash
python -m uvicorn api.main:app --reload --port 8000
```

### 6. Open the frontend

Open `frontend/demo.html` in a browser while the API is running on `localhost:8000`.

## Request logging

Every call to `/scan` and `/chat` is written as one row to the `requests` table in PostgreSQL: the final verdict and which layer blocked, the rule id that matched, the classifier probability, token count and whether the prompt was truncated at 128 tokens, timings per layer, and the git commit of the code that was running.

Settings in `.env`:

| Variable | Meaning |
|---|---|
| `DATABASE_URL` | connection string for the logging database |
| `LOG_TO_DB` | `true` / `false`: switch logging on or off |
| `LOG_PROMPT_TEXT` | `true` stores prompt and LLM response text; `false` (default) stores only the SHA-256 hash and numbers |
| `RUN_LABEL` | optional tag written to every row, to separate experiments |

A failure while logging never breaks an API request. Example query:

```bash
docker exec firewall_db psql -U firewall -d firewall_logs -c "select id, endpoint, blocked, blocked_by, san_rule_id, clf_attack_prob, total_ms from requests order by id desc limit 10;"
```

## Tests

```bash
docker exec firewall_db psql -U firewall -d postgres -c "create database firewall_test;"   # once
pytest -q
```

Tests that need the database use the separate `firewall_test` database and are skipped if it is not reachable. One test is marked `xfail`: it documents a known issue (a long benign prompt is blocked because of a length shortcut in the classifier).

To check that a change does not alter what the firewall decides:

```bash
python -m scripts.decisions record data/baselines/now.csv
python -m scripts.decisions compare data/baselines/v1_decisions.csv data/baselines/now.csv
```

## Dataset
 
A labeled dataset of 1,428 prompts was collected across multiple rounds: 639 safe samples and 789 attack samples, split 80/20 (stratified) into 1,142 training rows and 286 test rows.
 
| Attack Type | Count | Source |
|---|---|---|
| Direct injection | 112 | HackAPrompt (levels 1–3) |
| Jailbreak | 386 | HackAPrompt + deepset + manual |
| Encoded attacks | 78 | Generated (base64 + rot13) |
| Prompt leaking | 91 | Manual + deepset |
| Normal safe | 200 | Alpaca dataset |
| Safe from deepset | 399 | deepset/prompt-injections |
| Tricky safe | 114 | Manual |
| Plain safe | 48 | Manual |
| **Total** | **1,428** | |
 
A separate out-of-distribution generalization test set of **1,231 samples** was extracted from `xTRam1/safe-guard-prompt-injection` — never used in training, strictly deduplicated against all training data.
 
## Evaluation Results
 
| Evaluation | Samples | Accuracy / Detection | F1 |
|---|---|---|---|
| Sanitizer tests | 19 | 100% | 1.00 |
| In-distribution classifier | 286 | 97% | 0.97 |
| Out-of-distribution (xTRam1) | 1,231 | 99.8% (1,229/1,231 detected) | — |
 
**Confusion matrix (in-distribution test set):**
```
              precision    recall   f1-score
safe              0.96      0.99      0.97
attack            0.98      0.96      0.97
 
[[150   2]
 [  6 128]]
```
 
The classifier converged at epoch 4 (F1 0.9697); training beyond that point showed no meaningful improvement and increasing signs of overfitting (training loss → 0, validation loss rising). Confidence threshold for blocking is `attack_probability > 0.7`.
 
A key finding during generalization testing was that **creative-writing-wrapper attacks** (injection phrases embedded inside requested story/poem openings) were a systematic blind spot — 22 such examples were identified, added to training data, and the model retrained.

## Two Demo Frontends

This project includes two frontend demos so the firewall can be tested with or without Ollama installed.

### `frontend/demo.html` — Full demo (requires Ollama)

Calls the `/chat` endpoint. Shows the complete pipeline including real LLM 
responses from Llama 3.2 and Layer 3 (output filter) in action. If Ollama 
is not installed or running, this still works — passed prompts will show 
a placeholder message instead of a real response, since the backend 
gracefully falls back when Ollama is unavailable (see `api/main.py`).

### `frontend/demo-no-ollama.html` — Detection-only demo (no installation required)

Calls the `/scan` endpoint directly. Tests only Layers 1 and 2 (sanitizer 
+ classifier) — the core ML detection work — with zero dependency on 
Ollama. This is the fastest way to grade the firewall's detection accuracy 
without installing anything beyond the Python requirements.

![Blocked by sanitizer](docs/images/screenshot1.png)

![Blocked by sanitizer](docs/images/screenshot2.png)

### Running either demo

```bash
    python -m uvicorn api.main:app --reload --port 8000
```

Then open either `frontend/demo.html` or `frontend/demo-no-ollama.html` 
directly in a browser while the API is running on `localhost:8000`.

### Testing the `/scan` endpoint directly (no frontend needed)

```bash
curl -X POST http://localhost:8000/scan \
  -H "Content-Type: application/json" \
  -d '{"prompt": "ignore all previous instructions"}'
```

Or use the interactive Swagger UI at `http://localhost:8000/docs`.
 
## Limitations
 
- **Single-turn scope only** — no memory of prior conversation turns; multi-turn escalation attacks are not covered.
- **Indirect injection not covered** — only scans user-supplied text, not documents/web content an agent might read.
- **Harmful content is out of scope** — the firewall detects injection technique, not harmful content intent (e.g. "explain how to make malware" with no injection framing passes through, by design).
- **English only** — keyword lists, regex patterns, and training data are entirely English.
- **Static keyword lists** — both the sanitizer and output filter rely on hardcoded keywords/patterns that can't adapt to entirely novel phrasing without retraining or manual updates.
See the full report for detailed per-layer limitations and the frontier lab comparison (Anthropic Claude Opus 4 model card, OpenAI GPT-4 system card).
 
## References
 
- Perez & Ribeiro (2022) — Ignore This Title and HackAPrompt
- Schulhoff et al. (2023) — HackAPrompt dataset
- Sanh et al. (2019) — DistilBERT
- Taori et al. (2023) — Stanford Alpaca
- Anthropic Claude Opus 4 Model Card
- OpenAI GPT-4 System Card