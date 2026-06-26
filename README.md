# Clinical Claim Decision Support Agent
### Cotiviti Intern Assessment — Topic 2: Clinical Decision Making & Pattern Recognition

**Deepeka Gurunathan | University of North Texas**

---

## Overview

A Streamlit-based proof of concept that demonstrates agentic clinical decision making for healthcare claims. The app combines three layers:

1. **Rule-based risk classification** — flags diagnosis-procedure mismatches, overutilization, and high-cost anomalies
2. **RAG policy retrieval** — retrieves relevant clinical billing policy from a ChromaDB vector store using OpenAI embeddings
3. **LLM chain reasoning** — GPT-4o-mini explains the risk decision and recommends an action (Approve / Clinical Review / Deny)

---

## Stack

- Python 3.11
- Streamlit
- OpenAI (GPT-4o-mini + text-embedding-3-small)
- ChromaDB (in-memory vector store)
- Pandas

---

## Setup

```bash
pip install -r requirements.txt
```

Add your OpenAI API key in `app.py`:
```python
OPENAI_API_KEY = "your-openai-api-key-here"
```

Run the app:
```bash
streamlit run app.py
```

---

## Sample Data

`sample_claims.csv` includes 4 synthetic claim scenarios:

| Patient | Scenario | Expected Risk |
|---|---|---|
| P001 | Pneumonia + routine office visit | Routine |
| P002 | Low back pain + knee replacement | High Risk |
| P003 | Wellness exam + high complexity visit + overutilization | High Risk |
| P004 | Heart attack + cardiac catheterization | Needs Review (high cost) |

---

## Demo Flow

1. Upload `sample_claims.csv` or use the auto-loaded sample
2. Select a claim row from the dropdown
3. Click **Analyze Claim**
4. View risk badge, flags, retrieved policy context, and agent reasoning
