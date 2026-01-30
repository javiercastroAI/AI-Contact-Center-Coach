AI Contact Center Coach — Prototype / Research Only
Version 3.3 — December 2025
© 2025–2026 Javier Castro (dnAI) — licensed under the MIT License (see LICENSE)

This project is classified as **High‑Risk** under Annex III §4(c) of the EU AI Act because it evaluates and monitors employees. Anyone who puts it into service becomes the **Provider** and must complete an AI Risk‑Management System, obtain CE‑marking, register the system in the EU database, and maintain human oversight and post‑market monitoring. Before processing voice or transcript data, the integrator must perform a GDPR‑compliant Data‑Protection Impact Assessment. Spanish labour rules (RDL 9/2021 and the Estatuto de los Trabajadores) require transparency about algorithms used for employee scoring and a way for agents to contest automated decisions. Operators covered by NIS2 must keep the supplied SBOM current, patch vulnerabilities promptly and report serious incidents within 72 hours. This repository is provided solely for demonstration and educational purposes. The author offers no warranty and accepts no liability for any live deployment.

Before any production use, the integrator must:
- Complete the AI RMS and Annex XI technical file
- Clear a DPIA with the organisation’s DPO or legal team
- Secure lawful grounds and contracts for storing or analysing customer calls
- Run bias and robustness audits on their own datasets
- Implement secure MLOps (static analysis, SBOM management, patching)

## Overview
AI Contact Center Coach evaluates and coaches contact‑center interactions by combining:
- Automatic **conversation segmentation** into phases (greeting, issue identification, troubleshooting, etc.)
- **Semantic script‑adherence scoring** using sentence embeddings
- **Resolution promise extraction** plus fulfillment validation (simulated)
- **Benchmarking** against target thresholds
- Optional **GPT‑4 coaching feedback** with sentiment and improvement guidance
- A **Streamlit dashboard** and a standalone HTML UI for visual insights

## Repository contents
- `AI Contact Center Coach V Dec 25.ipynb` — original notebook prototype
- `contact_center_coach.py` — pure Python analysis module (core logic)
- `streamlit_app.py` — Streamlit UI for metrics, KPIs, and insights
- `coach-dashboard.html` — standalone UI mock for visualizing outputs
- `LICENSE` — MIT license
- `.github/SECURITY.md` — security policy and PGP key

## How the system works (detailed)
1) **Input**: a diarized transcript (`Agent:` / `Customer:` lines).  
2) **Segmentation**: keyword triggers on agent lines move a `current_section` pointer. All following lines are grouped into that section until another trigger fires.  
3) **Script adherence**:
   - For each section, only the agent lines in that section are scored.
   - SentenceTransformer (`all‑MiniLM‑L6‑v2`) computes embeddings.
   - Each golden sentence is matched to the best agent sentence by cosine similarity.
   - Section adherence is the mean of those best‑match similarities.
   - Overall adherence is a weighted average across sections.
4) **Resolution promises**:
   - Regex patterns detect action promises (open a ticket, send email, check systems, etc.).
   - A simulated validator marks each promise as validated, pending, or unknown.
5) **KPIs**:
   - Overall adherence, per‑section adherence, CSAT
   - Resolution integrity and promise status counts
6) **Coaching** (optional): GPT‑4 uses scores + transcript + promise results to generate qualitative feedback.

## Quick start (Streamlit UI)
```bash
pip install streamlit sentence-transformers openai
streamlit run streamlit_app.py
```
Optional for GPT‑4 coaching:
```bash
export OPENAI_API_KEY="your_key_here"
```

## Run the analysis module directly
```python
from contact_center_coach import analyze_transcript, DEFAULT_TRANSCRIPT
results = analyze_transcript(DEFAULT_TRANSCRIPT, csat_score=4.0)
print(results["overall_adherence"], results["section_scores"])
```

## Outputs you can expect
- Section adherence breakdown + benchmarks
- KPI summary table (overall adherence, CSAT, resolution integrity)
- Resolution promise audit (validated / pending / unknown)
- Segmented transcript view
- Optional GPT‑4 coaching narrative

## Notes on limitations
- Segmentation is **heuristic** (keyword‑based) and can misclassify sections.
- Promise validation is **simulated** in this prototype.
- A single agent sentence can match multiple golden sentences (can inflate scores).

## Security
See `.github/SECURITY.md` for the responsible‑disclosure policy and PGP key.

## License
MIT License — see `LICENSE`.

## Author
Javier Castro — founder of dnAI, AI transformation architect, contact‑center analytics specialist.
"Quality is not what you say — it’s what you promise and actually deliver."
