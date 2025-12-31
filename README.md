AI Contact Center Coach — Prototype / Research Only  
Version 3.3 — December 2025  
© 2025 Javier Castro (dnAI) — licensed under the MIT License (see LICENSE)

This project is classified as **High-Risk** under Annex III §4(c) of the EU AI Act because it evaluates and monitors employees. Anyone who puts it into service becomes the **Provider** and must complete an AI Risk-Management System, obtain CE-marking, register the system in the EU database and maintain human oversight and post-market monitoring. Before processing voice or transcript data, the integrator must perform a GDPR-compliant Data-Protection Impact Assessment. Spanish labour rules (RDL 9/2021 and the Estatuto de los Trabajadores) require transparency about algorithms used for employee scoring and a way for agents to contest automated decisions. Operators covered by NIS2 must keep the supplied SBOM current, patch vulnerabilities promptly and report serious incidents within 72 hours. This repository is provided solely for demonstration and educational purposes. The author offers no warranty and accepts no liability for any live deployment.

**Before any production use** the integrator must:  
• finish the full AI RMS and Annex XI technical file,  
• clear a DPIA with the organisation’s DPO or legal team,  
• secure lawful grounds and contracts for storing or analysing customer calls,  
• run bias and robustness audits on their own datasets,  
• implement a secure MLOps pipeline with static-analysis scanning, SBOM management and continuous patching.  
dnAI can assist professionally with these steps.

**Overview.** AI Contact Center Coach evaluates, coaches and improves contact-centre interactions by combining semantic golden-script adherence scoring, automatic conversation segmentation, resolution-promise extraction and fulfilment validation, GPT-powered qualitative coaching feedback and visual highlighting of unfulfilled promises. Traditional QA asks whether agents uttered the correct words; this system checks whether they said the right thing, promised the right action and actually delivered it, directly affecting CSAT, first-call resolution, churn and trust.

**Core objectives.** Measure what agents say, detect what they promise, verify fulfilment, highlight high-impact risk areas and provide actionable AI-generated coaching guidance.

**Key capabilities.** Automatic call segmentation into phases without manual tagging; semantic adherence scoring with SentenceTransformers embeddings, cosine similarity and section-plus-overall explanations; detection, validation and flagging of resolution promises; benchmarked and explainable scoring against 75th-percentile targets; GPT-4 coaching feedback covering sentiment, strengths, weaknesses and suggested golden phrases.

**Architecture flow.** Diarised transcript → heuristic segmentation → agent-utterance extraction → (a) golden-script embedding comparison for adherence scores and (b) promise extraction and back-end validation → GPT-4 coaching and feedback.

**Tech stack.** Python, NumPy, regex-based NLP, SentenceTransformers (all-MiniLM-L6-v2), OpenAI GPT-4 API, IPython / Markdown reporting.

**Setup.** Clone the repository, install dependencies from requirements.txt, set `OPENAI_API_KEY` in a `.env` file, run `ai_contact_center_coach.py`. The script outputs section-level adherence, highlights unfulfilled promises and generates a full coaching report.

**Outputs.** Semantic similarity tables per section, weighted overall adherence score, promise audit (fulfilled vs pending) and AI coaching recommendations with golden phrases.

**Security.** A signed CycloneDX SBOM (`sbom-cyclonedx.json`, SHA-256, dated 13 Dec 2025) is included. See `.github/SECURITY.md` for the responsible-disclosure policy (response target ≤ 30 days) and embedded public PGP key.

**License reminder.** Distributed under the MIT License. The software is provided **“AS IS”**, without warranty of any kind. You accept full responsibility for any deployment or derivative work.

**Author.** Javier Castro — founder of dnAI, CEO-turned-AI architect, specialist in AI-driven transformation, contact-centre analytics and operational intelligence. “Quality is not what you say — it’s what you promise and actually deliver.”

Road-map ideas: live CRM / ticketing integrations, real-time agent assist, reinforcement learning for script optimisation, multilingual support, ISO- and AI-Act-aligned governance logging.
