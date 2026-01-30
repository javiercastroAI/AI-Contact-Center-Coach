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
from typing import Dict, List, Tuple

import numpy as np

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


@dataclass
class ResolutionResult:
    promise: str
    action: str
    status: str
    details: str


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


def auto_segment_transcript(
    transcript: str, section_keywords: List[Tuple[str, List[str]]] | None = None
) -> Dict[str, List[str]]:
    section_keywords = section_keywords or SECTION_KEYWORDS
    sections = {sec: [] for sec, _ in section_keywords}
    current_section = "Greeting Section"
    lines = transcript.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("agent:"):
            for sec, keywords in section_keywords:
                for keyword in keywords:
                    if keyword in line.lower():
                        current_section = sec
                        break
        sections[current_section].append(line)
    return sections


def extract_agent_utterances(section_lines: List[str]) -> List[str]:
    agent_lines = []
    for line in section_lines:
        if line.startswith("Agent:"):
            agent_lines.append(line[len("Agent:") :].strip())
    return agent_lines


def extract_resolution_promises(transcript: str) -> List[str]:
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
    for line in lines:
        if line.lower().startswith("agent:"):
            if re.search(pattern, line, re.IGNORECASE):
                promises.append(line.strip())
    return promises


def check_resolution_action(promise: str) -> ResolutionResult:
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
        )
    if "sent" in promise_lower and "address" in promise_lower:
        return ResolutionResult(
            promise=promise,
            action="email confirmation",
            status="pending",
            details="Latest email was not sent.",
        )
    if "checking our systems" in promise_lower:
        return ResolutionResult(
            promise=promise,
            action="system check",
            status="validated",
            details="Instability in the area confirmed in system logs.",
        )
    return ResolutionResult(
        promise=promise,
        action="other",
        status="unknown",
        details="No matching log found.",
    )


def compute_section_adherence(
    golden_sentences: List[str],
    agent_sentences: List[str],
    model: SentenceTransformer,
) -> Tuple[float, List[float], List[Dict[str, str | float]]]:
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

    agent_embeddings = model.encode(agent_sentences, convert_to_tensor=True)
    best_match_scores: List[float] = []
    comparisons: List[Dict[str, str | float]] = []
    for golden_sentence in golden_sentences:
        if "[agent name]" in golden_sentence:
            best_similarity = -1.0
            best_idx = None
            for j, candidate in enumerate(agent_sentences):
                processed_golden = replace_agent_name(golden_sentence, candidate)
                embedding_processed = model.encode([processed_golden], convert_to_tensor=True)
                similarity = util.cos_sim(embedding_processed, agent_embeddings[j : j + 1])
                similarity_value = float(similarity.item())
                if similarity_value > best_similarity:
                    best_similarity = similarity_value
                    best_idx = j
            best_match_scores.append(best_similarity)
            comparisons.append(
                {
                    "golden_sentence": golden_sentence,
                    "agent_sentence": agent_sentences[best_idx] if best_idx is not None else "No matching sentence found",
                    "similarity_score": best_similarity,
                }
            )
        else:
            embedding_golden = model.encode([golden_sentence], convert_to_tensor=True)
            similarities = util.cos_sim(embedding_golden, agent_embeddings)
            best_similarity = float(similarities.max().item())
            best_idx = int(similarities.argmax())
            best_match_scores.append(best_similarity)
            comparisons.append(
                {
                    "golden_sentence": golden_sentence,
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
) -> str:
    if OpenAI is None:
        return "OpenAI SDK is not installed. Install `openai` to enable coaching feedback."
    if not openai_api_key:
        return "Missing OPENAI_API_KEY. Set it in your environment to enable coaching feedback."

    client = OpenAI(api_key=openai_api_key)
    resolution_feedback = "\\n".join(
        [f'Promise: "{r.promise}" -> Action: {r.action}, Status: {r.status}' for r in resolution_results]
    )

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

Provide a detailed coaching summary with:
  - A customer sentiment analysis that describes sentiment at the start, middle, and end (or key moments) and notes shifts with brief transcript quotes.
  - What the agent did well in both communication and fulfilling operational commitments.
  - Areas for improvement, especially if any resolution promises (e.g., opening a support ticket or sending a confirmation email) were not validated.
  - Specific suggestions on how to improve low-scoring sections and ensure that promised actions are confirmed.
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
) -> Dict[str, object]:
    benchmarks = benchmarks or BENCHMARKS
    golden_scripts = golden_scripts or GOLDEN_SCRIPTS
    weights = weights or DEFAULT_SECTION_WEIGHTS

    sections = auto_segment_transcript(transcript)
    model = get_embedding_model()
    section_scores: Dict[str, float] = {}
    detailed_comparisons: Dict[str, List[Dict[str, str | float]]] = {}

    for section_name, golden_lines in golden_scripts.items():
        if section_name in sections:
            section_content = sections[section_name]
            agent_utterances = extract_agent_utterances(section_content)
            adherence, _, comparisons = compute_section_adherence(
                golden_lines, agent_sentences=agent_utterances, model=model
            )
            section_scores[section_name] = adherence
            detailed_comparisons[section_name] = comparisons
        else:
            section_scores[section_name] = 0.0
            detailed_comparisons[section_name] = []

    overall_adherence = compute_overall_adherence(section_scores, weights)

    resolution_promises = extract_resolution_promises(transcript)
    resolution_results = [check_resolution_action(promise) for promise in resolution_promises]

    detailed_feedback = generate_detailed_feedback(section_scores, overall_adherence, csat_score, benchmarks)

    return {
        "sections": sections,
        "section_scores": section_scores,
        "comparisons": detailed_comparisons,
        "overall_adherence": overall_adherence,
        "csat_score": csat_score,
        "resolution_promises": resolution_promises,
        "resolution_results": resolution_results,
        "detailed_feedback": detailed_feedback,
    }


def safe_html(text: str) -> str:
    return html_lib.escape(text)


def resolution_integrity(resolution_results: List[ResolutionResult]) -> Tuple[float, int]:
    if not resolution_results:
        return 100.0, 0
    validated = sum(1 for result in resolution_results if result.status == "validated")
    pending = sum(1 for result in resolution_results if result.status == "pending")
    return (validated / len(resolution_results)) * 100, pending
