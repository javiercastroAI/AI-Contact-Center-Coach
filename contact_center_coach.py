# Purpose: research/education prototype only.
# License: MIT (see LICENSE).
# Author: Javier Castro (dnAI).
# Security: report issues to javiercastro@aiready.es (see .github/SECURITY.md).

from __future__ import annotations

import html as html_lib
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

import numpy as np
import math
try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

try:
    from sentence_transformers import SentenceTransformer, util
except Exception as exc:  # pragma: no cover - handled at runtime
    SentenceTransformer = None
    util = None
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = exc

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


GOLDEN_SCRIPTS: Dict[str, List[str]] = {
    "Greeting Section": [
        "Hello, thank you for calling ConnectPlus. My name is [agent name]. How may I assist you today? "
        "Good morning, and welcome to ConnectPlus support, where we are dedicated to resolving your issues "
        "efficiently. I am [agent name], and I am here to ensure you have a seamless experience. Whether you need "
        "technical support or general assistance, I am here to help with any inquiries you might have.",
        "Hi, thank you for reaching ConnectPlus. My name is [agent name]. What can I do for you today? "
        "Welcome to ConnectPlus customer service - I am [agent name] and I look forward to helping you get connected "
        "and solving any issues you may face.",
    ],
    "Issue Identification Section": [
        "I am sorry to hear you are experiencing difficulties with your service. Could you please describe in detail "
        "the issues you are facing? For instance, are you having trouble connecting or is your connection unstable? "
        "I would like to verify your account details to understand the root cause better. Your input is invaluable in "
        "helping me address the issue promptly.",
        "I understand you are encountering problems with your connection. Please explain what you are experiencing and "
        "any error messages you might have seen. Let me also check your account and recent activity so we can pinpoint "
        "the problem.",
    ],
    "Troubleshooting Section": [
        "Based on our initial check, it appears there is a disruption affecting your area. Let's perform a few "
        "diagnostic tests to pinpoint the issue. Please check if your modem's indicator lights are steady or blinking; "
        "this will help determine whether the signal is stable. We'll then proceed with some troubleshooting steps "
        "such as rebooting your router if necessary.",
        "Our system shows some instability in your neighborhood. To isolate the problem, kindly confirm whether the "
        "lights on your modem are flashing or remain constant. We will try a few troubleshooting measures together, "
        "and your cooperation is much appreciated.",
    ],
    "Solution Delivery Section": [
        "Thank you for your patience. Our analysis indicates that the issue is due to a local network disruption. Our "
        "technical team has been alerted and is actively working on restoring normal service. We expect the problem to "
        "be resolved within the next two hours, and I will ensure you receive timely updates via text.",
        "I appreciate your cooperation. It appears that a network disruption is affecting your service, and our "
        "engineers are addressing it as a priority. We anticipate the issue will be resolved shortly, and you will be "
        "notified once service is restored.",
    ],
    "Resolution (Ticket Creation) Section": [
        "I understand your issue requires further investigation. Let me open a support ticket for your case right away, "
        "and I will send you a confirmation email shortly.",
        "Your issue has been logged in our system. A support ticket has been created, and you will receive an email "
        "confirmation within a few minutes.",
    ],
    "Upsell (AIDA) Section": [
        "In addition to resolving your issue, I would like to inform you about a special promotion we are currently "
        "running. We are offering an exclusive discount on our Premium Internet plan, which provides faster speeds, "
        "enhanced connectivity, and additional features like a complimentary streaming subscription. Would you be "
        "interested in receiving more information about this offer?",
        "Furthermore, we have a limited-time offer on our upgraded service package that includes bonus features and "
        "faster speeds. If you are interested, I would be happy to send you the details so you can take advantage of "
        "this promotion.",
    ],
    "Closing Section": [
        "Before we conclude, is there anything else I can assist you with today? Thank you for choosing ConnectPlus; it "
        "has been a pleasure helping you. Should you have any further questions, please feel free to call us again. "
        "Have a wonderful day, and thank you for calling ConnectPlus.",
        "I hope I have addressed all your concerns. Thank you for contacting ConnectPlus. We value your business and "
        "are committed to your satisfaction. If you need further assistance in the future, do not hesitate to reach "
        "out. Enjoy your day and stay safe!",
    ],
}

BENCHMARKS: Dict[str, float] = {
    "Greeting Section": 80.0,
    "Issue Identification Section": 70.0,
    "Troubleshooting Section": 65.0,
    "Solution Delivery Section": 60.0,
    "Resolution (Ticket Creation) Section": 70.0,
    "Upsell (AIDA) Section": 55.0,
    "Closing Section": 75.0,
    "Overall": 75.0,
    "CSAT": 7.5,
}

SECTION_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("Greeting Section", ["good afternoon", "thank you for calling", "my name", "how may i help"]),
    ("Issue Identification Section", ["frustrating", "describe", "dropping", "issue", "account"]),
    ("Troubleshooting Section", ["checking our systems", "instability", "modem", "blinking", "reboot"]),
    ("Solution Delivery Section", ["engineers estimate", "notify you via text", "fixed", "up and running"]),
    ("Resolution (Ticket Creation) Section", ["open a support ticket", "ticket", "log your issue"]),
    ("Upsell (AIDA) Section", ["basic plan", "20% off", "promotion", "offer", "upgrade"]),
    ("Closing Section", ["before we finish", "thanks for calling", "great rest of your day", "bye", "take care"]),
]

DEFAULT_SECTION_WEIGHTS: Dict[str, float] = {
    "Greeting Section": 0.1,
    "Issue Identification Section": 0.2,
    "Troubleshooting Section": 0.2,
    "Solution Delivery Section": 0.2,
    "Resolution (Ticket Creation) Section": 0.1,
    "Upsell (AIDA) Section": 0.1,
    "Closing Section": 0.1,
}

DEFAULT_TRANSCRIPT = """
Agent: Good afternoon! You've reached the support line for ConnectPlus. My name's Monica. How may I help you today?
Customer: Hi. My connection has been dropping all morning.
Agent: Oh no, that must be frustrating. Let's get it sorted. Can you describe what exactly is happening?
Customer: Well, it disconnects every few minutes, especially when I'm on video calls.
Agent: Got it. Give me a second to access your account and see if anything unusual pops up... Alright, I've pulled up your details.
Agent: I'm checking our systems now. There's some instability reported in your neighborhood since early morning.
Customer: So it's a local outage?
Agent: It appears so. But to be sure, let's do a quick check. Can you confirm whether your modem's Internet light is blinking or steady?
Customer: It's blinking.
Agent: Okay, that confirms it. The signal isn't stable. Just to double-check, can you reboot the router for me?
Customer: Sure. Done.
Agent: Thanks! It'll take a minute to reconnect... Looks like it's trying to stabilize now.
Customer: When will it be fixed?
Agent: The engineers estimate about 90 minutes. We'll notify you via text once everything's up and running.
Customer: Alright.
Agent: I understand your issue requires further investigation. Let me open a support ticket for your case right away.
Customer: Yes, please open a ticket.
Agent: I understand your issue requires further investigation. Let me open a support ticket for your case right away, and I will send you a confirmation email shortly.
Customer: Thank you.
Agent: While we're at it, I noticed you're on the Basic Plan. We're currently offering 20% off on our Premium plan with double the speed and a streaming service included.
Customer: Oh? Interesting.
Agent: Would you like me to send over the details by email or walk you through it briefly?
Customer: Email's good.
Agent: Perfect! I've sent it to the address we have on file. You can upgrade in just one click if you're interested.
Customer: Thanks.
Agent: Before we finish, is there anything else I can assist you with today?
Customer: Nope, that's all.
Agent: Thanks for calling ConnectPlus, [Customer Name]. It was a pleasure helping you. Have a great rest of your day!
Customer: You too. Bye.
""".strip()

POSITIVE_CUES = [
    "thank", "thanks", "appreciate", "pleasure", "great", "resolved", "fixed",
    "glad", "happy", "wonderful", "excellent", "perfect", "good job"
]
NEGATIVE_CUES = [
    "frustrat", "problem", "issue", "error", "disconnect", "dropping",
    "angry", "upset", "annoyed", "bad", "terrible", "awful", "hate", "cannot"
]


@dataclass
class ResolutionResult:
    promise: str
    action: str
    status: str
    details: str
    turn: int | None = None


@lru_cache(maxsize=1)
def get_embedding_model():
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is required for scoring. "
            f"Import error: {_SENTENCE_TRANSFORMERS_IMPORT_ERROR}"
        )
    return SentenceTransformer("all-MiniLM-L6-v2")


def replace_agent_name(golden_sentence: str, agent_utterance: str) -> str:
    if "[agent name]" not in golden_sentence:
        return golden_sentence
    match = re.search(r"\\b([A-Z][a-z]+)\\b", agent_utterance)
    if match:
        agent_name = match.group(1)
        return golden_sentence.replace("[agent name]", agent_name)
    return golden_sentence


def format_golden_scripts(scripts: Dict[str, List[str]]) -> str:
    md = "**Golden Script for Each Section:**\\n\\n"
    for section, sentences in scripts.items():
        md += f"- **{section}:**\\n"
        for i, sentence in enumerate(sentences, 1):
            md += f"   - Option {i}: \\\"{sentence}\\\"\\n"
        md += "\\n"
    return md


def extract_agent_name(lines: List[str]) -> str | None:
    for line in lines:
        if not line.lower().startswith("agent:"):
            continue
        match = re.search(r"(?:my name is|my name's|this is)\\s+([A-Z][a-z]+)", line, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def build_section_prototypes(
    golden_scripts: Dict[str, List[str]],
    section_keywords: List[Tuple[str, List[str]]],
    model: SentenceTransformer,
) -> Dict[str, np.ndarray]:
    prototypes: Dict[str, np.ndarray] = {}
    keyword_map = {section: keywords for section, keywords in section_keywords}
    for section, sentences in golden_scripts.items():
        keyword_phrases = keyword_map.get(section, [])
        reference_texts = sentences + keyword_phrases
        if not reference_texts:
            continue
        embeddings = model.encode(reference_texts, convert_to_numpy=True)
        prototypes[section] = np.mean(embeddings, axis=0)
    return prototypes


def sentiment_score_lines(transcript: str, window: int = 3) -> List[Dict[str, object]]:
    lines = [line.strip() for line in transcript.splitlines() if line.strip()]
    scores = []
    for idx, line in enumerate(lines):
        text = line.split(":", 1)[1].strip() if ":" in line else line
        lower = text.lower()
        score = 0.0
        for cue in POSITIVE_CUES:
            if cue in lower:
                score += 1.0
        for cue in NEGATIVE_CUES:
            if cue in lower:
                score -= 1.0
        score = math.tanh(score)  # keep in [-1,1]
        speaker = line.split(":", 1)[0].strip() if ":" in line else "Unknown"
        scores.append({"turn": idx, "speaker": speaker, "text": text, "sentiment_raw": score})
    if not scores:
        return []
    # simple moving average smoothing
    smoothed = []
    half = window // 2
    for i in range(len(scores)):
        start = max(0, i - half)
        end = min(len(scores), i + half + 1)
        avg = np.mean([scores[j]["sentiment_raw"] for j in range(start, end)])
        item = scores[i].copy()
        item["sentiment_smooth"] = float(avg)
        smoothed.append(item)
    return smoothed


def segment_transcript_semantic(
    transcript: str,
    model: SentenceTransformer,
    golden_scripts: Dict[str, List[str]],
    section_keywords: List[Tuple[str, List[str]]],
    keyword_boost: float = 0.2,
    transition_penalty: float = 0.15,
    backtrack_penalty: float = 0.35,
    skip_penalty: float = 0.25,
    customer_weight: float = 0.75,
) -> Dict[str, List[str]]:
    sections = {sec: [] for sec, _ in section_keywords}
    lines = [line.strip() for line in transcript.splitlines() if line.strip()]
    if not lines:
        return sections

    order = [sec for sec, _ in section_keywords]
    prototypes = build_section_prototypes(golden_scripts, section_keywords, model)
    if not prototypes:
        return auto_segment_transcript(transcript, section_keywords)
    fallback_vector = next(iter(prototypes.values()))
    section_embeddings = np.stack(
        [prototypes.get(sec, fallback_vector) for sec in order]
    )

    line_embeddings = model.encode(lines, convert_to_numpy=True)

    emission_scores = np.zeros((len(lines), len(order)), dtype=float)
    for i, line in enumerate(lines):
        line_vec = line_embeddings[i]
        sim = util.cos_sim(line_vec, section_embeddings).cpu().numpy().flatten()
        speaker_weight = customer_weight if line.lower().startswith("customer:") else 1.0
        for sec_idx, sec in enumerate(order):
            score = float(sim[sec_idx])
            keywords = dict(section_keywords).get(sec, [])
            if keywords and any(keyword in line.lower() for keyword in keywords):
                score += keyword_boost
            emission_scores[i, sec_idx] = score * speaker_weight

    n_lines, n_sections = emission_scores.shape
    dp = np.full((n_lines, n_sections), -1e9, dtype=float)
    back = np.zeros((n_lines, n_sections), dtype=int)

    dp[0] = emission_scores[0]

    for i in range(1, n_lines):
        for curr in range(n_sections):
            best_score = -1e9
            best_prev = 0
            for prev in range(n_sections):
                diff = curr - prev
                if diff == 0:
                    penalty = 0.0
                elif diff > 0:
                    penalty = transition_penalty + skip_penalty * (diff - 1)
                else:
                    penalty = backtrack_penalty * abs(diff)
                score = dp[i - 1, prev] - penalty + emission_scores[i, curr]
                if score > best_score:
                    best_score = score
                    best_prev = prev
            dp[i, curr] = best_score
            back[i, curr] = best_prev

    best_last = int(dp[-1].argmax())
    path = [best_last]
    for i in range(n_lines - 1, 0, -1):
        best_last = int(back[i, best_last])
        path.append(best_last)
    path.reverse()

    for line, sec_idx in zip(lines, path):
        sections[order[sec_idx]].append(line)
    sections["_path"] = [order[idx] for idx in path]
    return sections


def segment_transcript(
    transcript: str,
    golden_scripts: Dict[str, List[str]] | None = None,
    section_keywords: List[Tuple[str, List[str]]] | None = None,
    segmentation_mode: str = "semantic",
    model: SentenceTransformer | None = None,
) -> Dict[str, List[str]]:
    golden_scripts = golden_scripts or GOLDEN_SCRIPTS
    section_keywords = section_keywords or SECTION_KEYWORDS
    if segmentation_mode == "heuristic" or SentenceTransformer is None:
        return auto_segment_transcript(transcript, section_keywords)
    model = model or get_embedding_model()
    return segment_transcript_semantic(transcript, model, golden_scripts, section_keywords)
def auto_segment_transcript(
    transcript: str, section_keywords: List[Tuple[str, List[str]]] | None = None
) -> Dict[str, List[str]]:
    section_keywords = section_keywords or SECTION_KEYWORDS
    sections = {sec: [] for sec, _ in section_keywords}
    path = []
    current_section = "Greeting Section"
    lines = transcript.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("agent:"):
            matched = False
            for sec, keywords in section_keywords:
                for keyword in keywords:
                    if keyword in line.lower():
                        current_section = sec
                        matched = True
                        break
                if matched:
                    break
        sections[current_section].append(line)
        path.append(current_section)
    sections["_path"] = path
    return sections


def extract_agent_utterances(section_lines: List[str]) -> List[str]:
    agent_lines = []
    for line in section_lines:
        if line.startswith("Agent:"):
            agent_lines.append(line[len("Agent:") :].strip())
    return agent_lines


def extract_resolution_promises(transcript: str) -> List[Tuple[str, int]]:
    promises = []
    pattern = (
        r"\b("
        r"let me open|"
        r"i will open|"
        r"open (a )?(support )?ticket|"
        r"log (your|this) (issue|problem)|"
        r"create (a )?(support )?ticket|"
        r"raise (a )?(support )?ticket|"
        r"file (a )?(support )?ticket|"
        r"submit (a )?(support )?ticket|"
        r"i\\'?ll (create|open|log) (a )?(support )?ticket|"
        r"send (a )?confirmation email|"
        r"email( you)? (the )?details|"
        r"confirm( via email)?|"
        r"update (our )?system|"
        r"check (our )?(logs|system)|"
        r"record (your|this) (issue|problem|concern)|"
        r"document (your|this) (issue|problem|concern)|"
        r"raise (an )?incident|"
        r"initiate (a )?case|"
        r"i\\'?ll (follow up|escalate|track|notify|verify|update|process|handle|resolve)|"
        r"make sure (to )?(open|create|log) (a )?(support )?ticket|"
        r"checking (our )?systems( now| immediately)?|"
        r"sent( it)? to (the )?address"
        r")\b"
    )
    lines = transcript.splitlines()
    for idx, line in enumerate(lines):
        if line.lower().startswith("agent:"):
            if re.search(pattern, line, re.IGNORECASE):
                promises.append((line.strip(), idx))
    return promises


def check_resolution_action(promise: str, turn: int | None = None) -> ResolutionResult:
    promise_lower = promise.lower()
    if (
        "ticket" in promise_lower
        or "support ticket" in promise_lower
        or "log your issue" in promise_lower
        or "open a support ticket" in promise_lower
        or "create a support ticket" in promise_lower
    ):
        return ResolutionResult(
            promise=promise,
            action="support ticket creation",
            status="validated",
            details="Support ticket #T123 processed at 15:05.",
            turn=turn,
        )
    if "sent" in promise_lower and "address" in promise_lower:
        return ResolutionResult(
            promise=promise,
            action="email confirmation",
            status="pending",
            details="Latest email was not sent.",
            turn=turn,
        )
    if "checking our systems" in promise_lower:
        return ResolutionResult(
            promise=promise,
            action="system check",
            status="validated",
            details="Instability in the area confirmed in system logs.",
            turn=turn,
        )
    return ResolutionResult(
        promise=promise,
        action="other",
        status="unknown",
        details="No matching log found.",
        turn=turn,
    )


def compute_section_adherence(
    golden_sentences: List[str],
    agent_sentences: List[str],
    model: SentenceTransformer,
    similarity_threshold: float = 0.4,
    enforce_unique: bool = True,
    agent_name: str | None = None,
) -> Tuple[float, List[float], List[Dict[str, str | float]]]:
    if not golden_sentences:
        return 0.0, [], []
    if not agent_sentences:
        comparisons = [
            {
                "golden_sentence": sentence,
                "agent_sentence": "No matching sentence found",
                "similarity_score": 0.0,
            }
            for sentence in golden_sentences
        ]
        return 0.0, [0.0 for _ in golden_sentences], comparisons

    name = agent_name or "Agent"
    processed_golden = [
        sentence.replace("[agent name]", name) if "[agent name]" in sentence else sentence
        for sentence in golden_sentences
    ]

    golden_embeddings = model.encode(processed_golden, convert_to_tensor=True)
    agent_embeddings = model.encode(agent_sentences, convert_to_tensor=True)
    sim_matrix = util.cos_sim(golden_embeddings, agent_embeddings).cpu().numpy()

    best_match_scores: List[float] = []
    comparisons: List[Dict[str, str | float]] = []

    if enforce_unique and linear_sum_assignment is not None:
        num_golden, num_agents = sim_matrix.shape
        if num_agents < num_golden:
            pad = np.zeros((num_golden, num_golden - num_agents))
            sim_padded = np.concatenate([sim_matrix, pad], axis=1)
        else:
            sim_padded = sim_matrix

        cost = -sim_padded
        row_ind, col_ind = linear_sum_assignment(cost)
        assignment = {row: col for row, col in zip(row_ind, col_ind)}

        for i, sentence in enumerate(golden_sentences):
            col = assignment.get(i)
            if col is None or col >= num_agents:
                best_match_scores.append(0.0)
                comparisons.append(
                    {
                        "golden_sentence": sentence,
                        "agent_sentence": "No matching sentence found",
                        "similarity_score": 0.0,
                    }
                )
                continue
            similarity = float(sim_matrix[i, col])
            if similarity < similarity_threshold:
                best_match_scores.append(0.0)
                comparisons.append(
                    {
                        "golden_sentence": sentence,
                        "agent_sentence": "No matching sentence found",
                        "similarity_score": similarity,
                    }
                )
            else:
                best_match_scores.append(similarity)
                comparisons.append(
                    {
                        "golden_sentence": sentence,
                        "agent_sentence": agent_sentences[col],
                        "similarity_score": similarity,
                    }
                )
    else:
        for i, sentence in enumerate(golden_sentences):
            similarities = sim_matrix[i]
            best_idx = int(similarities.argmax())
            best_similarity = float(similarities[best_idx])
            if best_similarity < similarity_threshold:
                best_match_scores.append(0.0)
                comparisons.append(
                    {
                        "golden_sentence": sentence,
                        "agent_sentence": "No matching sentence found",
                        "similarity_score": best_similarity,
                    }
                )
            else:
                best_match_scores.append(best_similarity)
                comparisons.append(
                    {
                        "golden_sentence": sentence,
                        "agent_sentence": agent_sentences[best_idx],
                        "similarity_score": best_similarity,
                    }
                )

    adherence_percentage = float(np.mean(best_match_scores) * 100)
    return adherence_percentage, best_match_scores, comparisons


def compute_overall_adherence(sections_adherence: Dict[str, float], weights: Dict[str, float] | None = None) -> float:
    if not sections_adherence:
        return 0.0
    if weights:
        weighted_sum = 0.0
        total_weight = 0.0
        for section, adherence in sections_adherence.items():
            weight = weights.get(section, 1.0)
            weighted_sum += adherence * weight
            total_weight += weight
        return weighted_sum / total_weight if total_weight else 0.0
    return sum(sections_adherence.values()) / len(sections_adherence)


def generate_section_feedback(section_name: str, adherence: float, benchmark: float) -> str:
    diff = adherence - benchmark
    if diff >= 0:
        return (
            f"Your adherence score is {adherence:.2f}% (benchmark: {benchmark:.2f}%). "
            f"You are {diff:.2f} points above the target. Great job!"
        )
    return (
        f"Your adherence score is {adherence:.2f}% (benchmark: {benchmark:.2f}%). "
        f"You are {abs(diff):.2f} points below the target. "
        "Consider using these golden script options."
    )


def generate_overall_feedback(overall_adherence: float, csat: float, overall_benchmark: float, csat_benchmark: float) -> str:
    overall_diff = overall_adherence - overall_benchmark
    csat_diff = csat - csat_benchmark
    feedback = []
    if overall_diff >= 0:
        feedback.append(
            f"Overall Script Adherence is {overall_adherence:.2f}% (benchmark: {overall_benchmark:.2f}%). "
            f"You are {overall_diff:.2f} points above the target."
        )
    else:
        feedback.append(
            f"Overall Script Adherence is {overall_adherence:.2f}% (benchmark: {overall_benchmark:.2f}%). "
            f"You are {abs(overall_diff):.2f} points below the target."
        )
    if csat_diff >= 0:
        feedback.append(
            f"Customer Satisfaction (CSAT) is {csat:.2f} out of 10 (benchmark: {csat_benchmark:.2f}). "
            f"You are {csat_diff:.2f} points above the target."
        )
    else:
        feedback.append(
            f"Customer Satisfaction (CSAT) is {csat:.2f} out of 10 (benchmark: {csat_benchmark:.2f}). "
            f"You are {abs(csat_diff):.2f} points below the target."
        )
    return " ".join(feedback)


def generate_detailed_feedback(
    section_scores: Dict[str, float],
    overall_adherence: float,
    csat: float,
    benchmarks: Dict[str, float],
) -> Dict[str, Dict[str, str]]:
    feedback: Dict[str, Dict[str, str]] = {}
    feedback["Overall"] = {
        "summary": generate_overall_feedback(overall_adherence, csat, benchmarks["Overall"], benchmarks["CSAT"])
    }
    section_feedback: Dict[str, str] = {}
    for section, adherence in section_scores.items():
        if section in benchmarks:
            section_feedback[section] = generate_section_feedback(section, adherence, benchmarks[section])
        else:
            section_feedback[section] = generate_section_feedback(section, adherence, 75.0)
    feedback["Sections"] = section_feedback
    return feedback


def get_gpt4_coaching(
    transcript: str,
    section_scores: Dict[str, float],
    csat: float,
    golden_script_md: str,
    resolution_results: List[ResolutionResult],
    openai_api_key: str | None,
    model: str = "gpt-4",
    extra_kpis: Dict[str, object] | None = None,
) -> str:
    if OpenAI is None:
        return "OpenAI SDK is not installed. Install `openai` to enable coaching feedback."
    if not openai_api_key:
        return "Missing OPENAI_API_KEY. Set it in your environment to enable coaching feedback."

    client = OpenAI(api_key=openai_api_key)
    resolution_feedback = "\\n".join(
        [f'Promise: "{r.promise}" -> Action: {r.action}, Status: {r.status}' for r in resolution_results]
    )

    extra_kpis = extra_kpis or {}
    prompt = f"""
You are an expert call center coach. Analyze the following full call transcript and the computed script adherence scores.
Below is the golden script for each section:
{golden_script_md}

The transcript is:
<<<TRANSCRIPT>>>
{transcript}
<<<END TRANSCRIPT>>>

The section adherence scores (in %) are:
{json.dumps(section_scores, indent=2)}

The customer satisfaction (CSAT) score for this call is {csat} out of 10.

Additionally, here are the resolution verification results from the call:
{resolution_feedback}

Extra KPIs to consider (may be useful for prioritization):
{json.dumps(extra_kpis, indent=2)}

Provide a detailed coaching summary with:
  - A customer sentiment analysis that describes sentiment at the start, middle, and end (or key moments) and notes shifts with brief transcript quotes.
  - What the agent did well in both communication and fulfilling operational commitments.
  - Areas for improvement, especially if any resolution promises (e.g., opening a support ticket or sending a confirmation email) were not validated.
  - Specific suggestions on how to improve low-scoring sections and ensure that promised actions are confirmed.
  - Highlight sections with largest coverage gaps and unique-match efficiency problems; address promise fulfillment lag and topic drift if relevant.
For each suggestion, include example utterances from the golden script that the agent should consider using, and explain how to check and log operational actions.
Please use bullet points and indentation in your response.
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful call center coaching assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=700,
        )
        return response.choices[0].message.content
    except Exception as exc:  # pragma: no cover - runtime errors
        return f"Error calling OpenAI: {exc}"


def analyze_transcript(
    transcript: str,
    csat_score: float,
    weights: Dict[str, float] | None = None,
    benchmarks: Dict[str, float] | None = None,
    golden_scripts: Dict[str, List[str]] | None = None,
    segmentation_mode: str = "semantic",
    similarity_threshold: float = 0.4,
    enforce_unique: bool = True,
) -> Dict[str, object]:
    benchmarks = benchmarks or BENCHMARKS
    golden_scripts = golden_scripts or GOLDEN_SCRIPTS
    weights = weights or DEFAULT_SECTION_WEIGHTS

    model = get_embedding_model()
    sections = segment_transcript(
        transcript,
        golden_scripts=golden_scripts,
        section_keywords=SECTION_KEYWORDS,
        segmentation_mode=segmentation_mode,
        model=model,
    )
    path = sections.get("_path", [])
    section_scores: Dict[str, float] = {}
    detailed_comparisons: Dict[str, List[Dict[str, str | float]]] = {}

    for section_name, golden_lines in golden_scripts.items():
        if section_name in sections:
            section_content = sections[section_name]
            agent_utterances = extract_agent_utterances(section_content)
            agent_name = extract_agent_name(section_content)
            adherence, _, comparisons = compute_section_adherence(
                golden_lines,
                agent_sentences=agent_utterances,
                model=model,
                similarity_threshold=similarity_threshold,
                enforce_unique=enforce_unique,
                agent_name=agent_name,
            )
            section_scores[section_name] = adherence
            detailed_comparisons[section_name] = comparisons
        else:
            section_scores[section_name] = 0.0
            detailed_comparisons[section_name] = []

    overall_adherence = compute_overall_adherence(section_scores, weights)

    resolution_promises = extract_resolution_promises(transcript)
    resolution_results = [
        check_resolution_action(promise, turn=turn) for promise, turn in resolution_promises
    ]

    detailed_feedback = generate_detailed_feedback(section_scores, overall_adherence, csat_score, benchmarks)

    sentiment_curve = sentiment_score_lines(transcript)

    # Section-level KPIs
    section_kpis: Dict[str, Dict[str, float]] = {}
    for section, comparisons in detailed_comparisons.items():
        if not comparisons:
            section_kpis[section] = {
                "coverage_gap": 1.0,
                "matched_fraction": 0.0,
            }
            continue
        below = sum(1 for c in comparisons if c["similarity_score"] < similarity_threshold)
        matched = sum(1 for c in comparisons if c["similarity_score"] >= similarity_threshold)
        coverage_gap = below / max(len(comparisons), 1)
        # approximate agent utterances used = len(comparisons) when enforce_unique; otherwise min
        agent_count = max(1, matched)
        matched_fraction = matched / agent_count
        section_kpis[section] = {
            "coverage_gap": coverage_gap,
            "matched_fraction": matched_fraction,
        }

    # Promise density (promises per 100 agent words)
    agent_words = sum(
        len(line.split())
        for line in transcript.splitlines()
        if line.lower().startswith("agent:")
    )
    promise_density = (len(resolution_results) / max(agent_words, 1)) * 100

    # Sentiment KPIs
    if sentiment_curve:
        min_sent = min(item["sentiment_smooth"] for item in sentiment_curve)
        end_sent = sentiment_curve[-1]["sentiment_smooth"]
        sentiment_recovery = end_sent - min_sent
    else:
        min_sent = 0.0
        end_sent = 0.0
        sentiment_recovery = 0.0

    # Topic drift (backtracks and big jumps)
    topic_drift = 0.0
    if path:
        order = [sec for sec, _ in SECTION_KEYWORDS]
        idx_map = {sec: i for i, sec in enumerate(order)}
        backtracks = 0
        jumps = 0
        for prev, curr in zip(path, path[1:]):
            if idx_map.get(curr, 0) < idx_map.get(prev, 0):
                backtracks += 1
            if abs(idx_map.get(curr, 0) - idx_map.get(prev, 0)) > 1:
                jumps += 1
        topic_drift = (backtracks + jumps) / max(len(path) - 1, 1)

    # Promise fulfillment lag (simulated)
    lag_scores = []
    for r in resolution_results:
        if r.status == "validated":
            lag_scores.append(0.1)
        elif r.status == "pending":
            lag_scores.append(1.0)
        else:
            lag_scores.append(0.5)
    promise_fulfillment_lag = float(np.mean(lag_scores)) if lag_scores else 0.0

    # Escalation risk: weighted combo
    low_adherence = max(0.0, (BENCHMARKS["Overall"] - overall_adherence) / 100)
    pending_ratio = sum(1 for r in resolution_results if r.status == "pending") / max(len(resolution_results), 1) if resolution_results else 0.0
    negative_end = max(0.0, (0 - end_sent) / 1.0) if sentiment_curve else 0.0
    escalation_risk = min(1.0, 0.4 * low_adherence + 0.4 * pending_ratio + 0.2 * negative_end)

    extra_kpis = {
        "promise_density_per_100_agent_words": promise_density,
        "sentiment_recovery": sentiment_recovery,
        "topic_drift": topic_drift,
        "promise_fulfillment_lag": promise_fulfillment_lag,
        "escalation_risk": escalation_risk,
        "coverage_gaps": {k: v.get("coverage_gap", 0.0) for k, v in section_kpis.items()},
        "unique_match_efficiency": {k: v.get("matched_fraction", 0.0) for k, v in section_kpis.items()},
    }

    return {
        "sections": sections,
        "section_scores": section_scores,
        "comparisons": detailed_comparisons,
        "overall_adherence": overall_adherence,
        "csat_score": csat_score,
        "resolution_promises": resolution_promises,
        "resolution_results": resolution_results,
        "detailed_feedback": detailed_feedback,
        "sentiment_curve": sentiment_curve,
        "section_kpis": section_kpis,
        "promise_density": promise_density,
        "sentiment_kpis": {
            "min": min_sent,
            "end": end_sent,
            "recovery": sentiment_recovery,
        },
        "topic_drift": topic_drift,
        "promise_fulfillment_lag": promise_fulfillment_lag,
        "escalation_risk": escalation_risk,
        "section_path": path,
        "extra_kpis": extra_kpis,
    }


def safe_html(text: str) -> str:
    return html_lib.escape(text)


def resolution_integrity(resolution_results: List[ResolutionResult]) -> Tuple[float, int]:
    if not resolution_results:
        return 100.0, 0
    validated = sum(1 for result in resolution_results if result.status == "validated")
    pending = sum(1 for result in resolution_results if result.status == "pending")
    return (validated / len(resolution_results)) * 100, pending


def export_dashboard_json(
    results: Dict,
    csat_score: Optional[float] = None,
    path: str = "dashboard_data.json",
    benchmarks: Optional[Dict[str, float]] = None,
) -> None:
    """
    Serialize key metrics for the static HTML dashboard (coach-dashboard.html).
    Writes a compact JSON file that the dashboard JS can load and render.
    """
    benchmarks = benchmarks or BENCHMARKS
    section_scores = results.get("section_scores", {})
    integrity_score, pending_count = resolution_integrity(results.get("resolution_results", []))

    payload = {
        "metrics": {
            "overall_adherence": results.get("overall_adherence", 0.0),
            "overall_delta": results.get("overall_adherence", 0.0) - benchmarks.get("Overall", 0.0),
            "csat": csat_score,
            "csat_delta": (csat_score - benchmarks.get("CSAT", 0.0)) if csat_score is not None else None,
            "resolution_integrity": integrity_score,
            "pending_promises": pending_count,
            "sentiment_summary": results.get("sentiment_kpis", {}).get("summary", ""),
        },
        "section_scores": section_scores,
        "resolution_promises": [
            {
                "title": r.promise,
                "detail": f"Turn {r.turn}" if r.turn is not None else "",
                "status": r.status,
            }
            for r in results.get("resolution_results", [])
        ],
        "coach_notes": results.get("detailed_feedback", {})
        .get("Overall", {})
        .get("summary", "")
        .split("\n"),
        "suggested_phrase": results.get("detailed_feedback", {})
        .get("Overall", {})
        .get("summary", ""),
        "sentiment_arc": results.get("sentiment_curve", []),
        "flow_path": results.get("section_path", results.get("sections", {}).get("_path", [])),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
