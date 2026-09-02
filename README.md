# MGC Aurora Heights — AI Developer Task

A practical build in four parts: document assistant, database schema, lead scoring, and a web interface that ties it together.

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Gemini API key
#    Edit .env and replace the placeholder:
echo GOOGLE_API_KEY=your_key_here > .env

# 4. (Optional) Train the lead scoring model (Part 3)
python lead_scoring.py

# 5. Run the web app (Part 4)
streamlit run app.py
```

The app opens at `http://localhost:8501`. Both tabs work independently — the Document Assistant needs a valid Gemini API key; Lead Scoring works offline once the model is trained.

---

## Part 1 — Document Assistant

**Approach:** Full-context prompting with LangChain + Google Gemini.

All three MGC documents (~7 KB total) are loaded into the system prompt. This is deliberate — not laziness:
- With only 3 small documents, vector-store RAG adds complexity without benefit.
- Full context guarantees the model sees **all** documents on every query, which is critical for detecting the **transfer fee conflict** (2% in the price list vs 2.5% in the FAQ).
- For a production system with hundreds of PDFs, I'd move to a vector store (FAISS or Chroma).

**Hard cases handled:**
| Question | Behavior |
|---|---|
| Transfer fee? | Flags the conflict between documents (2% vs 2.5%), shows both sources |
| Rental yield? | Refuses — FAQ explicitly says MGC doesn't publish projections |
| Anchor tenant? | States it's unconfirmed per the brochure |

**File:** `document_assistant.py`

---

## Part 2 — Database

**Schema:** Single `leads` table with proper types and a `UNIQUE` constraint on `crm_record_hash`.

One table is defensible here — it's a flat CRM export with no relational complexity. Normalizing `source` and `city` into lookup tables is textbook-correct but overkill for a 9K-row sales tool.

**Duplicate prevention:** The `UNIQUE(crm_record_hash)` constraint means the database rejects any insert where the same lead (by identity hash) already exists — regardless of what `lead_id` the agent assigns.

**Files:** `schema.sql`, `queries.sql`

---

## Part 3 — Lead Scoring

### Data Decisions

**Dropped columns:**
- `lead_id` — identifier, no predictive value
- `created_at` — temporal ordering risks leakage
- `crm_record_hash` — internal dedup hash
- `token_amount_received_pkr` — **DATA LEAKAGE**. Token payment is a *consequence* of converting, not a cause. Including it inflates metrics to ~99% and would be dishonest.

**Cleaned:**
- **City names:** Normalized 12 variants (`ISLAMABAD`→`Islamabad`, `ISB`→`Islamabad`, `khi`→`Karachi`, `Rwp`→`Rawalpindi`, etc.)
- **Missing values:** `area` filled with "Unknown"; numerical columns (`bedrooms`, `first_response_minutes`, `agent_experience_years`, `budget_pkr_lac`) filled with median
- **Duplicates:** Removed 160 duplicate rows (same `crm_record_hash`)

**Model:** Random Forest with `class_weight='balanced'` — no tuning, honest baseline.

### Metric: F1-score (positive class)

**Why F1?** The class balance is ~93% negative / 7% positive. Accuracy would be 93% by always predicting "not converted" — completely useless. F1 balances precision and recall on the minority class (converted leads), which is what the sales team cares about: **finding leads worth calling without wasting time on false positives.**

AUC-ROC is reported as a secondary metric.

**File:** `lead_scoring.py`

---

## Part 4 — Web Interface

**Stack:** Streamlit (explicitly listed as acceptable in the brief).

Single-page app with two tabs:
1. **Document Assistant** — chat interface for Part 1
2. **Lead Scoring** — form input for Part 3, returns conversion probability with priority level

**File:** `app.py`

---

## What I'd Do With More Time

- Add a vector store for scalability when document count grows
- Hyperparameter tuning with cross-validation for Part 3
- Feature engineering: interaction terms, time-based features from `created_at`
- Add SHAP explanations to the lead scoring output ("this lead scored high because...")
- Conversation memory in the document assistant
- Input validation and error handling throughout

---

## Built With

- Python 3, LangChain, Google Gemini API
- scikit-learn (Random Forest)
- Streamlit
- SQLite (schema dialect)
