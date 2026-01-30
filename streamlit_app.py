# Purpose: research/education prototype only.
# License: MIT (see LICENSE).
# Security: report issues to javiercastro@aiready.es (see .github/SECURITY.md).

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Dict

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

with st.sidebar:
    st.header("Inputs")
    transcript = st.text_area("Transcript", value=DEFAULT_TRANSCRIPT, height=320)
    csat_score = st.slider("CSAT score (0-10)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)

    st.subheader("Section Weights")
    weight_inputs: Dict[str, float] = {}
    for section, weight in DEFAULT_SECTION_WEIGHTS.items():
        weight_inputs[section] = st.number_input(section, min_value=0.0, max_value=1.0, value=weight, step=0.05)

    st.subheader("Coaching (Optional)")
    use_coaching = st.checkbox("Generate GPT-4 coaching feedback", value=False)
    api_key = st.text_input(
        "OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", "") if use_coaching else ""
    )
    model_name = st.text_input("Model name", value="gpt-4") if use_coaching else "gpt-4"

    run_analysis = st.button("Run Analysis", type="primary")


if run_analysis:
    results = analyze_transcript(
        transcript=transcript,
        csat_score=csat_score,
        weights=weight_inputs,
        benchmarks=BENCHMARKS,
        golden_scripts=GOLDEN_SCRIPTS,
    )

    overall_adherence = results["overall_adherence"]
    integrity_score, pending_count = resolution_integrity(results["resolution_results"])
    resolution_total = len(results["resolution_results"])
    resolution_validated = sum(1 for item in results["resolution_results"] if item.status == "validated")
    resolution_unknown = sum(1 for item in results["resolution_results"] if item.status == "unknown")

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

    st.write("")
    st.markdown("### Coaching Summary")
    st.write(results["detailed_feedback"]["Overall"]["summary"])
    for section_name, feedback in results["detailed_feedback"]["Sections"].items():
        st.write(f"**{section_name}**: {feedback}")

    if use_coaching:
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
        )
        st.write(coaching_text)

    st.write("")
    st.markdown("### Raw Results")
    st.json(
        {
            "overall_adherence": overall_adherence,
            "section_scores": section_scores,
            "resolution_results": resolution_rows,
        }
    )
else:
    st.info("Enter a transcript and click **Run Analysis** to generate insights.")
