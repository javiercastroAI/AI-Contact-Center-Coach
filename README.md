AI Contact Center Coach — Prototype / Research Only  
Version 3.4 — January 2026  
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
- Research‑grade **conversation segmentation** with semantic section prototypes + Viterbi smoothing (falls back to keyword heuristics)
- **Unique‑matching script adherence** (Hungarian assignment) with tunable similarity threshold
- **Resolution promise extraction** plus fulfillment validation (simulated)
- **Benchmarking** against target thresholds and rich per‑section diagnostics
- Optional **GPT‑4 coaching feedback** with sentiment, topic‑drift, and actionability guidance
- A **Streamlit dashboard** and a standalone HTML UI for visual insights

## Repository contents
- `AI Contact Center Coach V Dec 25.ipynb` — original notebook prototype (reference)
- `contact_center_coach.py` — core analysis module
- `contact_center_coach_research.py` — same core plus research instrumentation
- `streamlit_app.py` — Streamlit UI for metrics, KPIs, and insights
- `coach-dashboard.html` — standalone UI mock for visualizing outputs
- `LICENSE` — MIT license
- `.github/SECURITY.md` — security policy and PGP key

## How the system works (detailed)
1) **Input**: diarized transcript (`Agent:` / `Customer:` lines) + optional CSAT + section weights.
2) **Segmentation** (research‑grade):
   - Semantic similarity to section prototypes scored per turn.
   - Viterbi decoding smooths the path across sections.
   - Fallback heuristic uses keyword triggers; the chosen path is stored as `_path`/`section_path`.
3) **Script adherence**:
   - Agent lines are embedded with SentenceTransformer (`all‑MiniLM‑L6‑v2`).
   - Golden sentences are *uniquely* matched to agent sentences via Hungarian assignment to avoid double‑counting.
   - Similarity threshold is tunable (default 0.40); sub‑threshold pairs are discarded.
   - Section adherence = mean similarity of matched pairs; overall adherence = weighted average by section weights.
4) **Resolution promises**:
   - Regex patterns extract promises (ticket, email, follow‑up, checks, etc.) with turn indices.
   - A simulated validator labels each as `validated`, `pending`, or `unknown`; integrity is the validated ratio.
5) **Additional KPIs**:
   - Coverage gap, unique‑match efficiency, promise density, sentiment recovery, topic drift, promise‑fulfillment lag (simulated), escalation risk.
   - Flow transitions between sections, sentiment curve, and brevity vs completeness views.
6) **Coaching (optional)**:
   - GPT‑4 prompt includes KPIs, resolution results, and golden scripts to generate section‑level feedback.
7) **Outputs**:
   - Section bars vs benchmark (red/yellow/green logic)
   - KPI summary, resolution table, segmented transcript, coverage heatmap, flow transitions, gap waterfall, risk bar chart, promise timeline, sentiment curve, targeted suggestions, and on‑demand full JSON of KPIs.

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
Research copy (identical API, extra traces):
```python
from contact_center_coach_research import analyze_transcript
```

## Outputs you can expect
- Section adherence vs benchmarks (color‑coded)
- KPI summary (overall adherence, CSAT, resolution integrity, coverage gap, unique‑match efficiency, sentiment recovery, topic drift, escalation risk)
- Resolution promise audit with statuses
- Segmented transcript and flow transitions
- Coverage heatmap and gap waterfall
- Sentiment curve and promise timeline
- Targeted suggestions table (good sections highlighted)
- Optional GPT‑4 coaching narrative
- On‑demand full JSON KPI payload

## Notes on limitations
- Promise validation is **simulated** in this prototype.
- Topic drift and escalation risk are heuristic signals, not guarantees.
- GPT‑4 coaching quality depends on model availability and API key correctness.

## Security
See `.github/SECURITY.md` for the responsible‑disclosure policy and PGP key.

## License
MIT License — see `LICENSE`.

## Author
Javier Castro — founder of dnAI, AI transformation architect, contact‑center analytics specialist.
"Quality is not what you say — it’s what you promise and actually deliver."
