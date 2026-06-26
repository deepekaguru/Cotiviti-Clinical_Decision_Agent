import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI

# ── Config ───────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(
    page_title="Clinical Claim Decision Support Agent",
    page_icon="🏥",
    layout="wide"
)

# ── Session State ─────────────────────────────────────────────────────────────
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = None
if "results" not in st.session_state:
    st.session_state.results = {}

# ── Sample Data ───────────────────────────────────────────────────────────────
DATA = {
    "patient_id":            ["P001", "P002", "P003", "P004"],
    "age":                   [32, 45, 58, 71],
    "gender":                ["F", "F", "M", "F"],
    "icd_code":              ["Z00.00", "N83.20", "J98.09", "I21.9"],
    "icd_description":       ["Annual physical exam", "Ovarian cyst with polyp", "Lung nodule unspecified", "Acute myocardial infarction"],
    "procedure_code":        ["99395", "58661", "32663", "93458"],
    "procedure_description": ["Preventive medicine exam 18-39 yrs", "Laparoscopic removal adnexal structures", "Thoracoscopic lobectomy", "Left heart catheterization"],
    "claim_amount":          [250, 12500, 68000, 18500],
    "prior_claims_30days":   [1, 2, 0, 2],
    "provider_id":           ["PR101", "PR202", "PR303", "PR404"],
    "doctor_notes":          [
        "Patient presents for annual wellness exam. No acute complaints, no chronic conditions. Preventive counseling provided. Routine labs ordered. All vitals within normal range.",
        "Patient presents with pelvic pain. Ultrasound confirmed ovarian cyst with polyp. Laparoscopic procedure recommended for removal. Pathology pending to rule out malignancy. Conservative management discussed but patient opted for surgical intervention.",
        "Incidental 4mm lung nodule found on routine chest X-ray. No biopsy performed. No PET scan or follow-up CT documented. No pulmonology consult obtained. Proceeding directly to thoracoscopic lobectomy without standard diagnostic workup.",
        "Patient admitted with sudden onset chest pain and diaphoresis. EKG shows ST elevation in leads II, III, aVF. Emergency left heart catheterization performed — confirmed RCA occlusion."
    ]
}
df = pd.DataFrame(DATA)

# ── Clinical Policy Knowledge Base ───────────────────────────────────────────
CLINICAL_POLICIES = [
    {"id": "pol_001", "text": "Annual preventive medicine exam (99395) for patients aged 18-39 is appropriate for diagnosis Z00.00 (General adult medical exam). No anomalies expected when diagnosis and procedure align with age-appropriate preventive care."},
    {"id": "pol_002", "text": "Laparoscopic removal of adnexal structures (58661) is appropriate for ovarian cyst with polyp (N83.20) when conservative management has been attempted or malignancy cannot be ruled out. Pathology review is required post-procedure."},
    {"id": "pol_003", "text": "Thoracoscopic lobectomy (32663) for lung nodule requires documented prior diagnostic workup including CT follow-up, PET scan, and biopsy or pulmonology consult before surgical intervention. Proceeding to lobectomy without standard diagnostic workup is not supported."},
    {"id": "pol_004", "text": "Left heart catheterization (93458) is an appropriate procedure for acute myocardial infarction (I21.x) diagnosis. This is a standard of care for cardiac intervention."},
    {"id": "pol_005", "text": "Claims with more than 5 visits to the same provider within 30 days may indicate overutilization or upcoding patterns and require clinical documentation review."}
]

# ── RAG ───────────────────────────────────────────────────────────────────────
@st.cache_resource
def build_vector_store():
    docs = [p["text"] for p in CLINICAL_POLICIES]
    response = client.embeddings.create(input=docs, model="text-embedding-3-small")
    embeddings = [r.embedding for r in response.data]
    return docs, embeddings

def retrieve_policy(docs_and_embeddings, query: str, n=2) -> str:
    docs, embeddings = docs_and_embeddings
    qr = client.embeddings.create(input=[query], model="text-embedding-3-small")
    qv = np.array(qr.data[0].embedding)
    scores = []
    for i, emb in enumerate(embeddings):
        dv = np.array(emb)
        score = np.dot(qv, dv) / (np.linalg.norm(qv) * np.linalg.norm(dv))
        scores.append((score, docs[i]))
    scores.sort(reverse=True)
    return "\n\n".join([f"- {d}" for _, d in scores[:n]])

# ── LLM Notes Extraction ──────────────────────────────────────────────────────
def extract_notes_signals(notes: str, icd: str, procedure: str) -> dict:
    prompt = f"""You are a clinical documentation reviewer for a healthcare payer.

Review the following doctor notes and extract key signals.

Doctor Notes: "{notes}"
Billed Diagnosis: {icd}
Billed Procedure: {procedure}

Respond ONLY in this exact JSON format, no extra text:
{{
  "supports_diagnosis": true or false,
  "supports_procedure": true or false,
  "red_flags": ["list of red flag strings, empty if none"],
  "summary": "one sentence summary of what the notes say"
}}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200
    )
    try:
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        return {"supports_diagnosis": True, "supports_procedure": True, "red_flags": [], "summary": "Could not parse notes."}

# ── Risk Scoring ──────────────────────────────────────────────────────────────
HIGH_COST_THRESHOLD = 20000
PRIOR_CLAIMS_THRESHOLD = 5
MISMATCHED_PAIRS = {"32663": ["J98.09"]}

def compute_risk(row: pd.Series, notes_signals: dict = None) -> tuple[str, list[str]]:
    flags = []
    proc = str(row["procedure_code"])
    icd  = str(row["icd_code"])
    if proc in MISMATCHED_PAIRS and icd in MISMATCHED_PAIRS[proc]:
        flags.append(f"Diagnosis-procedure mismatch: {icd} does not support {proc}")
    if float(row["claim_amount"]) > HIGH_COST_THRESHOLD:
        flags.append(f"High claim amount: ${float(row['claim_amount']):,.0f} exceeds threshold")
    if int(row["prior_claims_30days"]) > PRIOR_CLAIMS_THRESHOLD:
        flags.append(f"Overutilization: {row['prior_claims_30days']} claims in last 30 days")
    if notes_signals:
        if not notes_signals.get("supports_procedure", True):
            flags.append("Doctor notes do not support the billed procedure")
        if not notes_signals.get("supports_diagnosis", True):
            flags.append("Doctor notes contradict the billed diagnosis")
        for rf in notes_signals.get("red_flags", []):
            flags.append(f"Notes red flag: {rf}")
    level = "High Risk" if len(flags) >= 2 else "Needs Review" if len(flags) == 1 else "Routine"
    return level, flags

# ── LLM Reasoning ────────────────────────────────────────────────────────────
def run_agent(row: pd.Series, risk_level: str, flags: list[str], policy_context: str, notes_summary: str) -> str:
    flag_text = "\n".join(flags) if flags else "No flags triggered."
    prompt = f"""You are a Clinical Claim Decision Support Agent for a healthcare payer.

Claim details:
Patient: {row['patient_id']} | Age: {row['age']} | Gender: {row['gender']}
Diagnosis: {row['icd_code']} — {row['icd_description']}
Procedure: {row['procedure_code']} — {row['procedure_description']}
Claim Amount: ${float(row['claim_amount']):,.0f}
Prior Claims (30d): {row['prior_claims_30days']}
Doctor Notes Summary: {notes_summary}

Risk level: {risk_level}
Flags: {flag_text}
Policy context: {policy_context}

Task: Explain the risk level, reference policy and notes, recommend Approve / Send for Clinical Review / Deny with Documentation Request. 3-5 sentences max.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }
    .block-container {
        padding-top: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px;
    }
    section[data-testid="stAppViewContainer"] { background: #f5f0f9; }
    .main .block-container {
        background: #ffffff;
        border: 1.5px solid #d5bce8;
        border-radius: 14px;
        box-shadow: 0 2px 16px rgba(107,45,139,0.07);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .brand-header {
        background-color: #6B2D8B;
        padding: 16px 24px;
        border-radius: 10px;
        margin-bottom: 1.2rem;
    }
    .brand-logo { color: #fff; font-size: 22px; font-weight: 700; letter-spacing: 3px; text-align: center; }
    .brand-subtitle { color: #c9a8e0; font-size: 12px; margin-top: 3px; text-align: center; }
    .brand-tag { background: #00A79D; color: #003d38; font-size: 10px; font-weight: 700; padding: 4px 10px; border-radius: 5px; letter-spacing: 1px; }
    .section-divider {
        font-size: 11px; font-weight: 700; color: #6B2D8B;
        letter-spacing: 1.5px; text-transform: uppercase;
        margin: 0 0 10px; padding-bottom: 6px;
        border-bottom: 1px solid #ede0f7;
    }
    .cards-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 1.4rem; }
    .claim-card { background: #fff; border: 1px solid #e0d0ee; border-radius: 10px; padding: 12px 14px; cursor: pointer; }
    .claim-card.selected { border: 2px solid #6B2D8B; background: #faf6fd; }
    .claim-card .pid { font-size: 10px; color: #6B2D8B; font-weight: 700; margin-bottom: 4px; }
    .claim-card .dx { font-size: 12px; font-weight: 700; color: #1a1a1a; line-height: 1.3; margin-bottom: 3px; }
    .claim-card .age { font-size: 11px; color: #888; }
    .claim-card .analyzed { font-size: 10px; color: #00A79D; margin-top: 5px; font-weight: 600; }
    .detail-panel {
        background: #f9f6fd; border: 1px solid #e0d0ee;
        border-radius: 10px; padding: 14px 16px; margin-bottom: 1.2rem;
    }
    .detail-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 12px; }
    .dc { background: #fff; border-radius: 7px; padding: 8px 12px; }
    .dc .dl { font-size: 10px; color: #888; margin-bottom: 2px; }
    .dc .dv { font-size: 13px; font-weight: 700; color: #1a1a1a; }
    .notes-lbl { font-size: 10px; color: #b87d00; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 5px; }
    .stButton > button {
        background-color: #6B2D8B !important;
        color: white !important; border: none !important;
        font-weight: 700 !important; border-radius: 8px !important;
        padding: 10px 28px !important; width: 100% !important;
    }
    .stButton > button:hover { background-color: #00A79D !important; }
    [data-testid="stExpander"] {
        border: 1px solid #d5bce8 !important;
        border-radius: 8px !important;
        background-color: #f0eaf4 !important;
    }
    .divider { border: none; border-top: 1px solid #e8dff0; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Load vector store ─────────────────────────────────────────────────────────
with st.spinner("Loading clinical policy knowledge base..."):
    try:
        vector_store = build_vector_store()
        vs_ok = True
    except Exception as e:
        st.error(f"Vector store error: {e}")
        vs_ok = False
        vector_store = None

# ── Brand Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="brand-header">
    <div style="display:grid; grid-template-columns:1fr auto 1fr; align-items:center; width:100%;">
        <div></div>
        <div style="text-align:center;">
            <div class="brand-logo">C O T I V I T I</div>
            <div class="brand-subtitle">Clinical Claim Decision Support Agent</div>
        </div>
        <div style="justify-self:end;">
            <div class="brand-tag">PAYMENT ACCURACY</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── About expander ────────────────────────────────────────────────────────────
with st.expander("ℹ️  About this POC — click to expand"):
    st.markdown("""
<div style="display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-top:10px; padding-top:10px; border-top:1px solid #d5bce8;">
    <div style="font-size:12px; color:#3a1a52;">
        <div style="font-size:10px; font-weight:700; color:#6B2D8B; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px;">What it does</div>
        Rule-based risk classification · LLM notes extraction · RAG policy retrieval · LLM chain reasoning
    </div>
    <div style="font-size:12px; color:#3a1a52;">
        <div style="font-size:10px; font-weight:700; color:#6B2D8B; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px;">Stack</div>
        Streamlit · OpenAI GPT-4o-mini · text-embedding-3-small · NumPy
    </div>
    <div style="font-size:12px; color:#3a1a52;">
        <div style="font-size:10px; font-weight:700; color:#6B2D8B; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px;">Domain</div>
        Healthcare claims — Treatment, Payment & Operations (TPO)
    </div>
</div>
""", unsafe_allow_html=True)

# ── Claims Queue ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-divider">Claims queue</div>', unsafe_allow_html=True)

cards_html = "<style>"
cards_html += ".cg{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:1rem;font-family:sans-serif;}"
cards_html += ".cc{background:#fff;border:1px solid #e0d0ee;border-radius:10px;padding:12px 14px;}"
cards_html += ".cc.sel{border:2px solid #6B2D8B;background:#faf6fd;}"
cards_html += ".pid{font-size:10px;color:#6B2D8B;font-weight:700;margin-bottom:4px;}"
cards_html += ".dx{font-size:12px;font-weight:700;color:#1a1a1a;line-height:1.3;margin-bottom:3px;}"
cards_html += ".age{font-size:11px;color:#888;}"
cards_html += ".analyzed{font-size:10px;color:#00A79D;margin-top:5px;font-weight:600;}"
cards_html += "</style>"
cards_html += '<div class="cg">'
for i, r in df.iterrows():
    sel = "cc sel" if st.session_state.selected_idx == i else "cc"
    analyzed = '<div class="analyzed">✔ analyzed</div>' if r['patient_id'] in st.session_state.results else ""
    cards_html += f'<div class="{sel}"><div class="pid">{r["patient_id"]}</div><div class="dx">{r["icd_description"]}</div><div class="age">Age {r["age"]} · {r["gender"]}</div>{analyzed}</div>'
cards_html += '</div>'
components.html(cards_html, height=160)
# Invisible buttons for card selection
cols = st.columns(4)
for i, r in df.iterrows():
    with cols[i]:
        if st.button(f"Select {r['patient_id']}", key=f"card_{i}"):
            st.session_state.selected_idx = i
            st.rerun()

# ── Detail Panel (only if card selected) ─────────────────────────────────────
if st.session_state.selected_idx is not None:
    idx = st.session_state.selected_idx
    row = df.loc[idx]

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="detail-panel">
        <div class="detail-grid">
            <div class="dc"><div class="dl">Patient</div><div class="dv">{row['patient_id']} · {row['gender']} · {row['age']} yrs</div></div>
            <div class="dc"><div class="dl">Claim amount</div><div class="dv">${float(row['claim_amount']):,.0f}</div></div>
            <div class="dc"><div class="dl">Prior claims (30d)</div><div class="dv">{row['prior_claims_30days']}</div></div>
            <div class="dc"><div class="dl">Diagnosis</div><div class="dv">{row['icd_code']} — {row['icd_description']}</div></div>
            <div class="dc"><div class="dl">Procedure</div><div class="dv">{row['procedure_code']} — {row['procedure_description']}</div></div>
            <div class="dc"><div class="dl">Provider</div><div class="dv">{row['provider_id']}</div></div>
        </div>
        <div class="notes-lbl">Doctor notes</div>
    </div>
    """, unsafe_allow_html=True)

    doctor_notes = st.text_area(
        "Review or edit notes before analysis",
        value=row["doctor_notes"],
        height=80,
        key=f"notes_{idx}",
        label_visibility="collapsed"
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    if st.button("🔍 Analyze Claim", type="primary"):
        with st.spinner("Step 1 of 3 — Extracting signals from doctor notes..."):
            notes_signals = extract_notes_signals(doctor_notes, row["icd_code"], row["procedure_description"])
        with st.spinner("Step 2 of 3 — Running rule-based + notes validation..."):
            risk_level, flags = compute_risk(row, notes_signals)
        policy_context = ""
        reasoning = ""
        if vs_ok and vector_store:
            with st.spinner("Step 3 of 3 — Retrieving policy and generating recommendation..."):
                query = f"{row['icd_description']} {row['procedure_description']}"
                policy_context = retrieve_policy(vector_store, query)
                try:
                    reasoning = run_agent(row, risk_level, flags, policy_context, notes_signals.get("summary", ""))
                except Exception as e:
                    reasoning = f"LLM error: {e}"
        st.session_state.results[row['patient_id']] = {
            "risk_level": risk_level, "flags": flags,
            "notes_signals": notes_signals,
            "policy_context": policy_context, "reasoning": reasoning
        }
        st.rerun()

    # ── Result display ────────────────────────────────────────────────────────
    pid = row['patient_id']
    if pid in st.session_state.results:
        r = st.session_state.results[pid]
        risk_level    = r["risk_level"]
        flags         = r["flags"]
        notes_sig     = r["notes_signals"]
        policy_ctx    = r["policy_context"]
        reasoning     = r["reasoning"]

        badge_styles = {
            "Routine":      ("✅", "#d4f5e2", "#1a7a4a"),
            "Needs Review": ("⚠️", "#fff3cd", "#8a6000"),
            "High Risk":    ("🚨", "#ffe0e0", "#8b0000"),
        }
        icon, bg, color = badge_styles.get(risk_level, ("", "#eee", "#000"))

        flag_dots = "".join([
            f'<div class="flag"><div class="dot"></div><div>{f}</div></div>'
            for f in flags
        ]) if flags else '<div style="font-size:12px;color:#444;">No flags triggered.</div>'

        notes_section = f"""
        <div class="notes-result">
            <div class="lbl" style="color:#b87d00;">Notes extraction</div>
            {notes_sig.get('summary','')}
            {'<br><b>Red flags:</b> ' + ', '.join(notes_sig.get('red_flags',[])) if notes_sig.get('red_flags') else ''}
        </div>""" if notes_sig else ""

        policy_section = f"""
        <div class="policy-result">
            <div class="lbl" style="color:#00A79D;">Retrieved policy</div>
            {policy_ctx.replace(chr(10), '<br>')}
        </div>""" if policy_ctx else ""

        reasoning_section = f"""
        <div class="rec">
            <div class="lbl" style="color:#6B2D8B;">Agent recommendation</div>
            {reasoning}
        </div>""" if reasoning else ""

        result_html = f"""
        <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: sans-serif; }}
        .result-box {{ border:1px solid #e0d0ee; border-radius:10px; overflow:hidden; }}
        .rh {{ background:#6B2D8B; padding:10px 16px; color:#fff; font-size:13px; font-weight:700; }}
        .rb {{ padding:14px 16px; background:#fff; }}
        .badge {{ display:inline-flex; align-items:center; gap:5px; font-size:13px; font-weight:700; padding:5px 12px; border-radius:6px; margin-bottom:10px; background:{bg}; color:{color}; }}
        .flag {{ display:flex; gap:8px; font-size:12px; color:#444; margin-bottom:5px; align-items:flex-start; }}
        .dot {{ width:6px; height:6px; border-radius:50%; background:#8b0000; margin-top:4px; flex-shrink:0; }}
        .notes-result {{ background:#fff8e8; border-left:3px solid #e6a817; border-radius:0 7px 7px 0; padding:9px 12px; font-size:11px; color:#4a3800; margin:10px 0; }}
        .policy-result {{ background:#e8f7f6; border-left:3px solid #00A79D; border-radius:0 7px 7px 0; padding:9px 12px; font-size:11px; color:#1a4a48; margin:10px 0; }}
        .rec {{ background:#f0eaf4; border-radius:7px; padding:10px 14px; font-size:12px; color:#3a1a52; line-height:1.6; }}
        .lbl {{ font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px; }}
        </style>
        <div class="result-box">
            <div class="rh">Agent analysis — {pid}</div>
            <div class="rb">
                <div class="badge">{icon} {risk_level}</div>
                {flag_dots}
                {notes_section}
                {policy_section}
                {reasoning_section}
            </div>
        </div>
        """
        components.html(result_html, height=560, scrolling=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:2rem; padding-top:1rem; border-top:1px solid #e8dff0;
     display:flex; justify-content:space-between; font-size:11px; color:#999;">
    <span>Deepeka Gurunathan</span>
    <span>Cotiviti Intern Assessment · Topic 2 · 2026</span>
</div>
""", unsafe_allow_html=True)