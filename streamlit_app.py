# Purpose: research/education prototype only.
# License: MIT (see LICENSE).
# Author: Javier Castro (dnAI).
# Security: report issues to javiercastro@aiready.es (see .github/SECURITY.md).

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Dict

import altair as alt
import pandas as pd
import streamlit as st

from contact_center_coach import (
    BENCHMARKS,
    DEFAULT_SECTION_WEIGHTS,
    DEFAULT_TRANSCRIPT,
    GOLDEN_SCRIPTS,
    analyze_transcript,
    format_golden_scripts,
    get_gpt4_coaching,
    resolution_integrity,
)


st.set_page_config(page_title="AI Contact Center Coach", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Space+Grotesk:wght@400;500;600&display=swap');

html, body, [class*="css"]  {
  font-family: 'Space Grotesk', sans-serif;
}

.hero-card {
  background: linear-gradient(130deg, #fff5e9 0%, #f6efe6 50%, #ecf7f5 100%);
  padding: 24px;
  border-radius: 24px;
  border: 1px solid #e6d8c7;
  box-shadow: 0 20px 50px rgba(30, 29, 27, 0.08);
}

.hero-title {
  font-family: 'Fraunces', serif;
  font-size: 2.2rem;
  margin-bottom: 4px;
}

.hero-subtitle {
  color: #6f6a64;
}

.section-card {
  background: #fff7ed;
  padding: 18px;
  border-radius: 18px;
  border: 1px solid #ead9c8;
  box-shadow: 0 18px 40px rgba(30, 29, 27, 0.06);
}

.pill {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  background: #fff;
  border: 1px solid #e6d8c7;
  font-size: 0.8rem;
  color: #6f6a64;
}

.bar-track {
  height: 10px;
  background: #f2e7da;
  border-radius: 999px;
  overflow: hidden;
  margin-bottom: 8px;
}

.bar-fill {
  height: 100%;
  border-radius: 999px;
}

.bar-fill.actual {
  background: #6c757d;
}

.bar-fill.benchmark {
  background: #1e88e5;
}

.bar-fill.actual.bad {
  background: #e53935;
}

.bar-fill.actual.warn {
  background: #f9a825;
}

.bar-fill.actual.good {
  background: #43a047;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero-card">
  <div class="hero-title">AI Contact Center Coach</div>
  <div class="hero-subtitle">A research prototype for script adherence, resolution integrity, and coaching insights.</div>
  <div class="hero-subtitle">Developed by Javier Castro (dnAI)</div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

# Session state to preserve latest results and settings
if "last_results" not in st.session_state:
    st.session_state.last_results = None
    st.session_state.last_api_key = ""
    st.session_state.last_model = "gpt-4"

with st.sidebar:
    st.header("Inputs")
    run_report = st.button("Run Analysis (no LLM)", type="primary")
    transcript = st.text_area("Transcript", value=DEFAULT_TRANSCRIPT, height=320)
    csat_score = st.slider("CSAT score (0-10)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)

    st.subheader("Section Weights")
    weight_inputs: Dict[str, float] = {}
    for section, weight in DEFAULT_SECTION_WEIGHTS.items():
        weight_inputs[section] = st.number_input(section, min_value=0.0, max_value=1.0, value=weight, step=0.05)

    st.subheader("Scoring Controls")
    similarity_threshold = st.slider("Similarity threshold", min_value=0.2, max_value=0.9, value=0.4, step=0.05)
    enforce_unique = st.checkbox("Enforce unique matching (research-grade)", value=True)

    st.subheader("Coaching (Optional)")
    use_coaching = st.checkbox("Generate GPT-4 coaching feedback", value=False)
    api_key = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", "") if use_coaching else "")
    model_name = st.text_input("Model name", value="gpt-4") if use_coaching else "gpt-4"
    run_coaching = st.button("Run LLM Coaching", type="secondary")



run_any = run_report or run_coaching

if run_any:
    results = analyze_transcript(
        transcript=transcript,
        csat_score=csat_score,
        weights=weight_inputs,
        benchmarks=BENCHMARKS,
        golden_scripts=GOLDEN_SCRIPTS,
        similarity_threshold=similarity_threshold,
        enforce_unique=enforce_unique,
    )
    st.session_state.last_results = results
    st.session_state.last_api_key = api_key
    st.session_state.last_model = model_name
else:
    results = st.session_state.get("last_results")

if results:
    overall_adherence = results["overall_adherence"]
    section_kpis = results.get("section_kpis", {})
    integrity_score, pending_count = resolution_integrity(results["resolution_results"])
    resolution_total = len(results["resolution_results"])
    resolution_validated = sum(1 for item in results["resolution_results"] if item.status == "validated")
    resolution_unknown = sum(1 for item in results["resolution_results"] if item.status == "unknown")
    sentiment_kpis = results.get("sentiment_kpis", {})
    promise_density = results.get("promise_density", 0.0)
    topic_drift = results.get("topic_drift", 0.0)
    promise_fulfillment_lag = results.get("promise_fulfillment_lag", 0.0)
    escalation_risk = results.get("escalation_risk", 0.0)

    st.markdown("### Performance Pulse")
    col1, col2, col3 = st.columns(3)
    col1.metric("Weighted Adherence", f"{overall_adherence:.1f}%", f"{overall_adherence - BENCHMARKS['Overall']:+.1f} vs target")
    col2.metric("CSAT", f"{csat_score:.1f} / 10", f"{csat_score - BENCHMARKS['CSAT']:+.1f} vs target")
    delta_text = f"{pending_count} pending" if pending_count else "All validated"
    col3.metric("Resolution Integrity", f"{integrity_score:.1f}%", delta_text)

    st.write("")
    st.markdown("### KPI Summary")
    kpi_rows = [
        {
            "KPI": "Overall Script Adherence",
            "Value": f"{overall_adherence:.2f}%",
            "Benchmark": f"{BENCHMARKS['Overall']:.2f}%",
            "Delta": f"{overall_adherence - BENCHMARKS['Overall']:+.2f}%",
        },
        {
            "KPI": "Customer Satisfaction (CSAT)",
            "Value": f"{csat_score:.2f} / 10",
            "Benchmark": f"{BENCHMARKS['CSAT']:.2f} / 10",
            "Delta": f"{csat_score - BENCHMARKS['CSAT']:+.2f}",
        },
        {
            "KPI": "Resolution Integrity",
            "Value": f"{integrity_score:.2f}%",
            "Benchmark": "N/A",
            "Delta": "N/A",
        },
        {
            "KPI": "Resolution Promises (Total)",
            "Value": str(resolution_total),
            "Benchmark": "N/A",
            "Delta": "N/A",
        },
        {
            "KPI": "Resolution Promises (Validated)",
            "Value": str(resolution_validated),
            "Benchmark": "N/A",
            "Delta": "N/A",
        },
        {
            "KPI": "Resolution Promises (Pending)",
            "Value": str(pending_count),
            "Benchmark": "N/A",
            "Delta": "N/A",
        },
        {
            "KPI": "Resolution Promises (Unknown)",
            "Value": str(resolution_unknown),
            "Benchmark": "N/A",
            "Delta": "N/A",
        },
        {
            "KPI": "Promise Density (per 100 agent words)",
            "Value": f"{promise_density:.2f}",
            "Benchmark": "N/A",
            "Delta": "N/A",
        },
        {
            "KPI": "Sentiment Recovery",
            "Value": f"{sentiment_kpis.get('recovery', 0.0):.2f}",
            "Benchmark": "N/A",
            "Delta": "N/A",
        },
        {
            "KPI": "Topic Drift",
            "Value": f"{topic_drift:.2f}",
            "Benchmark": "Lower is better",
            "Delta": "",
        },
        {
            "KPI": "Promise Fulfillment Lag (simulated)",
            "Value": f"{promise_fulfillment_lag:.2f}",
            "Benchmark": "Lower is better",
            "Delta": "",
        },
        {
            "KPI": "Escalation Risk",
            "Value": f"{escalation_risk:.2f}",
            "Benchmark": "Lower is better",
            "Delta": "",
        },
    ]

    section_scores = results["section_scores"]
    for section, score in section_scores.items():
        benchmark = BENCHMARKS.get(section, 75.0)
        kpi_rows.append(
            {
                "KPI": f"{section} Adherence",
                "Value": f"{score:.2f}%",
                "Benchmark": f"{benchmark:.2f}%",
                "Delta": f"{score - benchmark:+.2f}%",
            }
        )
        if section in section_kpis:
            gap = section_kpis[section]["coverage_gap"]
            match_frac = section_kpis[section]["matched_fraction"]
            kpi_rows.append(
                {
                    "KPI": f"{section} Coverage Gap",
                    "Value": f"{gap*100:.1f}%",
                    "Benchmark": "Lower is better",
                    "Delta": "",
                }
            )
            kpi_rows.append(
                {
                    "KPI": f"{section} Unique Match Efficiency",
                    "Value": f"{match_frac*100:.1f}%",
                    "Benchmark": "Higher is better",
                    "Delta": "",
                }
            )

    st.dataframe(kpi_rows, use_container_width=True)
    st.write("")
    st.markdown("### Section Adherence")
    section_columns = st.columns(2)
    for idx, (section, score) in enumerate(section_scores.items()):
        benchmark = BENCHMARKS.get(section, 75.0)
        score_pct = min(max(score, 0.0), 100.0)
        benchmark_pct = min(max(benchmark, 0.0), 100.0)
        if benchmark > 0:
            if score <= 0.8 * benchmark:
                actual_class = "bad"
            elif score < benchmark:
                actual_class = "warn"
            else:
                actual_class = "good"
        else:
            actual_class = "good" if score >= 0 else "warn"

        with section_columns[idx % 2]:
            st.markdown(f"<div class='section-card'><strong>{section}</strong>", unsafe_allow_html=True)
            st.caption("Actual")
            st.markdown(
                f"<div class='bar-track'><div class='bar-fill actual {actual_class}' style='width: {score_pct:.1f}%'></div></div>",
                unsafe_allow_html=True,
            )
            st.caption("Benchmark")
            st.markdown(
                f"<div class='bar-track'><div class='bar-fill benchmark' style='width: {benchmark_pct:.1f}%'></div></div>",
                unsafe_allow_html=True,
            )
            st.caption(f"{score:.1f}% vs benchmark {benchmark:.0f}%")
            st.markdown("</div>", unsafe_allow_html=True)
    st.write("")
    st.markdown("### Resolution Promises")
    resolution_rows = [asdict(item) for item in results["resolution_results"]]
    if resolution_rows:
        st.dataframe(resolution_rows, use_container_width=True)
    else:
        st.info("No resolution promises detected.")

    st.write("")
    st.markdown("### Segmented Transcript")
    for section_name, lines in results["sections"].items():
        with st.expander(section_name, expanded=False):
            if not lines:
                st.caption("No lines matched this section.")
            else:
                st.write("\n".join(lines))

    st.write("")
    st.markdown("### Script Adherence Comparisons")
    for section_name, comparisons in results["comparisons"].items():
        with st.expander(f"{section_name} comparisons", expanded=False):
            if not comparisons:
                st.caption("No comparisons for this section.")
                continue
            st.dataframe(comparisons, use_container_width=True)

    # Coverage heatmap (golden sentences vs similarity)
    st.write("")
    st.markdown("### Coverage Heatmap (Golden vs Similarity)")
    heat_rows = []
    for section_name, comparisons in results["comparisons"].items():
        for comp in comparisons:
            heat_rows.append(
                {
                    "Section": section_name,
                    "Golden": comp["golden_sentence"],
                    "Similarity": comp["similarity_score"],
                }
            )
    if heat_rows:
        df_heat = pd.DataFrame(heat_rows)
        chart = (
            alt.Chart(df_heat)
            .mark_rect()
            .encode(
                x="Section:N",
                y=alt.Y("Golden:N", sort=None),
                color=alt.Color("Similarity:Q", scale=alt.Scale(scheme="redyellowgreen")),
                tooltip=["Section", "Golden", alt.Tooltip("Similarity", format=".3f")],
            )
            .properties(height=360)
        )

        st.altair_chart(chart, use_container_width=True)
    else:
        st.caption("No coverage data available.")

    # Flow transitions heatmap (Topic drift view)
    st.write("")
    st.markdown("### Flow Transitions")
    path_seq = results.get("section_path", []) or results.get("sections", {}).get("_path", [])
    if not path_seq:
        st.caption("No path data available.")
    else:
        counts = {}
        for prev, curr in zip(path_seq, path_seq[1:]):
            key = (prev, curr)
            counts[key] = counts.get(key, 0) + 1
        rows = [{"From": p, "To": c, "Count": cnt} for (p, c), cnt in counts.items()]
        df_trans = pd.DataFrame(rows)
        chart = (
            alt.Chart(df_trans)
            .mark_rect()
            .encode(
                x="From:N",
                y="To:N",
                color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
                tooltip=["From", "To", "Count"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    # Gap waterfall (section contribution to shortfall)
    st.write("")
    st.markdown("### Gap Waterfall (Adherence Shortfall by Section)")
    waterfall_rows = []
    for section, score in section_scores.items():
        benchmark = BENCHMARKS.get(section, 75.0)
        shortfall = benchmark - score
        waterfall_rows.append({"Section": section, "Shortfall": shortfall})
    df_water = pd.DataFrame(waterfall_rows)
    chart = (
        alt.Chart(df_water)
        .mark_bar()
        .encode(
            x="Section:N",
            y=alt.Y("Shortfall:Q"),
            color=alt.condition("datum.Shortfall > 0", alt.value("#e53935"), alt.value("#43a047")),
            tooltip=["Section", alt.Tooltip("Shortfall", format=".2f")],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)

    # Brevity vs Completeness scatter (per section)
    st.write("")
    st.markdown("### Brevity vs Completeness")
    brev_rows = []
    for section, lines in results["sections"].items():
        if section == "_path":
            continue
            continue
        agent_utts = [l for l in lines if l.lower().startswith("agent:")]
        words = sum(len(l.split()) for l in agent_utts)
        utts = max(1, len(agent_utts))
        wpu = words / utts
        completeness = section_scores.get(section, 0.0)
        brev_rows.append({"Section": section, "Words per agent turn": wpu, "Adherence": completeness})
    df_brev = pd.DataFrame(brev_rows)
    chart = (
        alt.Chart(df_brev)
        .mark_circle(size=120)
        .encode(
            x="Words per agent turn:Q",
            y="Adherence:Q",
            color="Section:N",
            tooltip=["Section", alt.Tooltip("Words per agent turn", format=".1f"), alt.Tooltip("Adherence", format=".1f")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)

    # Risk Factors (bar proxy for radar)
    st.write("")
    st.markdown("### Risk Factors")
    radar_rows = [
        {"Factor": "Adherence Shortfall", "Score": max(0.0, BENCHMARKS["Overall"] - overall_adherence)},
        {"Factor": "Pending Promises", "Score": pending_count},
        {"Factor": "Topic Drift", "Score": topic_drift * 100},
        {"Factor": "Escalation Risk (0-100)", "Score": escalation_risk * 100},
    ]
    df_radar = pd.DataFrame(radar_rows)
    chart = (
        alt.Chart(df_radar)
        .mark_bar()
        .encode(
            x="Factor:N",
            y="Score:Q",
            color=alt.Color("Factor:N", scale=alt.Scale(scheme="tableau20")),
            tooltip=["Factor", alt.Tooltip("Score", format=".2f")],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)

    # Promise timeline
    st.write("")
    st.markdown("### Promise Timeline")
    timeline = results.get("resolution_results", [])
    timeline_rows = []
    for item in timeline:
        timeline_rows.append(
            {
                "Turn": item.turn if item.turn is not None else -1,
                "Status": item.status.title(),
                "Promise": item.promise,
            }
        )
    if timeline_rows:
        df_timeline = pd.DataFrame(timeline_rows)
        chart = (
            alt.Chart(df_timeline)
            .mark_circle(size=120)
            .encode(
                x="Turn:Q",
                y=alt.Y("Status:N"),
                color=alt.Color("Status:N", scale=alt.Scale(scheme="set1")),
                tooltip=["Turn", "Status", "Promise"],
            )
            .properties(height=200)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.caption("No promises found to plot.")

    # Sentiment curve
    st.write("")
    st.markdown("### Customer Sentiment Over Time")
    curve = results.get("sentiment_curve", [])
    if curve:
        df_curve = pd.DataFrame(curve)
        chart = (
            alt.Chart(df_curve)
            .mark_line()
            .encode(
                x="turn",
                y=alt.Y("sentiment_smooth", scale=alt.Scale(domain=[-1, 1])),
                color="speaker",
                tooltip=["turn", "speaker", "text", "sentiment_raw", "sentiment_smooth"],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.caption("No sentiment data available.")

    # Coaching summary and targeted suggestions
    st.write("")
    st.markdown("### Coaching Summary")
    st.write(results["detailed_feedback"]["Overall"]["summary"])
    for section_name, feedback in results["detailed_feedback"]["Sections"].items():
        st.write(f"**{section_name}**: {feedback}")

    st.write("")
    st.markdown("### Targeted Suggestions")
    rows = []
    for section, score in section_scores.items():
        benchmark = BENCHMARKS.get(section, 75.0)
        golden = GOLDEN_SCRIPTS.get(section, [])
        if score < benchmark and golden:
            rows.append(
                {
                    "Section": section,
                    "Score": f"{score:.2f}%",
                    "Benchmark": f"{benchmark:.2f}%",
                    "Gap (pts)": f"{score - benchmark:+.2f}",
                    "Try saying": golden[0],
                    "Or": golden[1] if len(golden) > 1 else "",
                    "Status": "Improve",
                }
            )
        else:
            rows.append(
                {
                    "Section": section,
                    "Score": f"{score:.2f}%",
                    "Benchmark": f"{benchmark:.2f}%",
                    "Gap (pts)": f"{score - benchmark:+.2f}",
                    "Try saying": "Good job!",
                    "Or": "",
                    "Status": "Good",
                }
            )
    if rows:
        rows = sorted(
            rows,
            key=lambda x: (0 if x["Status"] == "Improve" else 1, float(x["Gap (pts)"]))
        )
        df_rows = pd.DataFrame(rows)
        def highlight(row):
            if row["Status"] == "Good":
                return ["background-color: #e8f5e9; color: #1b5e20"] * len(row)
            return [""] * len(row)
        st.dataframe(df_rows.style.apply(highlight, axis=1), use_container_width=True, hide_index=True)
    else:
        st.caption("No sections to display.")

    if run_coaching:
        if use_coaching and api_key.strip():
            st.write("")
            st.markdown("### GPT-4 Coaching Feedback")
            golden_script_md = format_golden_scripts(GOLDEN_SCRIPTS)
            coaching_text = get_gpt4_coaching(
                transcript=transcript,
                section_scores=section_scores,
                csat=csat_score,
                golden_script_md=golden_script_md,
                resolution_results=results["resolution_results"],
                openai_api_key=api_key,
                model=model_name,
                extra_kpis=results.get("extra_kpis", {}),
            )
            st.write(coaching_text)
        else:
            st.warning("Enable coaching and provide OPENAI_API_KEY to generate GPT feedback.")

    st.write("")
    # Full JSON payload (on demand)
    if st.button("Show full JSON KPIs"):
        def _serialize_result(r):
            if hasattr(r, "__dict__"):
                return {k: v for k, v in r.__dict__.items()}
            if isinstance(r, dict):
                return r
            return str(r)

        serializable = {}
        for k, v in results.items():
            if k == "resolution_results":
                serializable[k] = [_serialize_result(r) for r in v]
            else:
                serializable[k] = v
        st.json(serializable, expanded=False)
else:
    st.info("Enter a transcript and click **Run Analysis** to generate insights.")
