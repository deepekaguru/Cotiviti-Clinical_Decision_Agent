# Clinical Claim Decision Support Agent

**Cotiviti - Clinical Decision Making & Pattern Recognition**

---

## Overview

A Streamlit-based proof of concept that demonstrates agentic prepay claim screening for healthcare payers. The agent combines three layers to analyze a claim and produce an auditable recommendation:

1. **Rule-based risk scoring** — flags diagnosis-procedure mismatches, high claim amounts, and overutilization patterns
2. **LLM notes extraction** — GPT-4o-mini reads physician clinical documentation and extracts signals that confirm or contradict the billed diagnosis and procedure
3. **RAG + chain reasoning** — retrieves the most relevant clinical billing policy via cosine similarity over OpenAI embeddings, then generates a final recommendation with a second LLM call

---

## Demo

📹 Watch the demo video here - https://www.loom.com/share/c6334bd089e84179badb2d5a336970b3

---

## Stack

- Python 3.11
- Streamlit
- OpenAI GPT-4o-mini + text-embedding-3-small
- NumPy (cosine similarity RAG)
- python-dotenv

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/your-username/cotiviti-clinical-decision-agent.git
cd cotiviti-clinical-decision-agent
```

**2. Create and activate virtual environment**
```bash
python -m venv venv
# Mac/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your OpenAI API key**

Create a `.env` file in the root directory:
```
OPENAI_API_KEY=sk-your-key-here
```

**5. Run the app**
```bash
streamlit run app.py
```

---

## Sample Claims

Four synthetic claim scenarios are pre-loaded covering all risk levels:

| Patient | Condition | Procedure | Expected Risk |
|---|---|---|---|
| P001 | Annual physical exam | Preventive medicine exam | ✅ Routine |
| P002 | Ovarian cyst with polyp | Laparoscopic removal | ⚠️ Needs Review |
| P003 | Lung nodule unspecified | Thoracoscopic lobectomy | 🚨 High Risk |
| P004 | Acute myocardial infarction | Left heart catheterization | ⚠️ Needs Review |

---

## How It Works

```
Claims data (ICD, CPT, amount, notes)
        ↓
Layer 1: Rule-based scoring → risk flags
        ↓
Layer 2: LLM notes extraction → clinical signals
        ↓
Layer 3: RAG policy retrieval + LLM chain reasoning
        ↓
Output: Risk level + policy context + agent recommendation
```

---

## Notes

- No real patient data is used — all claims are synthetic
- ChromaDB dependency removed; RAG implemented with NumPy cosine similarity for cross-platform compatibility
- Session state preserves analysis results when switching between claims
