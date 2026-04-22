"""PDF/TXT text extraction + LLM-based field extraction."""
import os
import io
import json
import re
import uuid
import logging
from typing import Tuple

import fitz  # PyMuPDF
from emergentintegrations.llm.chat import LlmChat, UserMessage

logger = logging.getLogger(__name__)


def _get_llm_key() -> str:
    return os.environ.get("EMERGENT_LLM_KEY", "")

EXTRACTION_SYSTEM_PROMPT = """You are an expert insurance claims data extraction engine.
You will receive the raw text of a First Notice of Loss (FNOL) document and must return a STRICT JSON object
with the structure below. If a field is not found in the document, use null (not empty string).

Return ONLY valid JSON — no markdown fences, no explanation, no preamble.

Schema:
{
  "policyNumber": string|null,
  "policyholderName": string|null,
  "effectiveDates": { "from": string|null, "to": string|null },
  "incident": {
    "date": string|null,
    "time": string|null,
    "location": string|null,
    "description": string|null
  },
  "involvedParties": {
    "claimant": { "name": string|null, "contact": string|null },
    "thirdParties": [ { "name": string|null, "contact": string|null } ]
  },
  "asset": {
    "type": string|null,
    "id": string|null,
    "estimatedDamage": number|null
  },
  "claimType": string|null,
  "attachments": [ string ],
  "initialEstimate": number|null
}

Rules:
- "claimType" must be lowercase and one of: "injury", "property", "auto", "theft", "fire", "flood", "vandalism", "liability", "other".
- For "incident.date" and "incident.time" ALWAYS use the DATE OF LOSS and TIME OF LOSS fields from the document (e.g. "Date of Loss", "Loss Date", "Date/Time of Loss" in ACORD forms) — not the policy effective dates or the report submission date.
- "effectiveDates" should be the policy's own effective coverage window (from / to) when the document clearly shows it; otherwise leave both null.
- Dates in ISO-like format YYYY-MM-DD when possible. Times in HH:MM (24-hour).
- Money values as numbers (no currency symbols, no commas).
- "attachments" is a list of strings — filenames or descriptions mentioned in document.
- If document is unclear or empty, return the schema with all null values.
"""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from PDF bytes using PyMuPDF.

    Also harvests AcroForm widget field-name + field-value pairs so that filled
    fillable forms (e.g. ACORD FNOL PDFs) yield useful content.
    """
    text_parts = []
    form_pairs = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text("text"))
            try:
                widgets = page.widgets() or []
            except Exception:
                widgets = []
            for w in widgets:
                name = (getattr(w, "field_name", None) or "").strip()
                value = getattr(w, "field_value", None)
                if value in (None, "", False):
                    continue
                if isinstance(value, bool):
                    value = "Yes" if value else "No"
                form_pairs.append(f"{name}: {value}")

    body = "\n".join(text_parts).strip()
    if form_pairs:
        body += "\n\n--- FORM FIELD VALUES ---\n" + "\n".join(form_pairs)
    return body


def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8").strip()
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1", errors="ignore").strip()


def extract_raw_text(filename: str, file_bytes: bytes) -> str:
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    if ext == ".txt":
        return extract_text_from_txt(file_bytes)
    raise ValueError(f"Unsupported file type: {ext}. Only .pdf and .txt are supported.")


def _clean_json_response(text: str) -> str:
    """Strip markdown fences if the LLM added them."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


async def extract_fields_llm(raw_text: str) -> Tuple[dict, float]:
    """Use Claude Sonnet 4.5 via Emergent LLM key to extract fields. Returns (fields_dict, confidence)."""
    key = _get_llm_key()
    if not key:
        raise RuntimeError("EMERGENT_LLM_KEY not configured")

    session_id = f"claim-extract-{uuid.uuid4()}"
    chat = LlmChat(
        api_key=key,
        session_id=session_id,
        system_message=EXTRACTION_SYSTEM_PROMPT,
    ).with_model("anthropic", "claude-sonnet-4-5-20250929")

    prompt = f"Extract all fields from this FNOL document and return STRICT JSON:\n\n---DOCUMENT START---\n{raw_text[:12000]}\n---DOCUMENT END---"
    response = await chat.send_message(UserMessage(text=prompt))
    cleaned = _clean_json_response(response)

    try:
        fields = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("LLM JSON parse failed: %s -- response: %s", e, response[:400])
        # Fallback: return empty skeleton
        fields = _empty_fields()
        return fields, 0.3

    confidence = _compute_confidence(fields, raw_text)
    return fields, confidence


def _empty_fields() -> dict:
    return {
        "policyNumber": None,
        "policyholderName": None,
        "effectiveDates": {"from": None, "to": None},
        "incident": {"date": None, "time": None, "location": None, "description": None},
        "involvedParties": {"claimant": {"name": None, "contact": None}, "thirdParties": []},
        "asset": {"type": None, "id": None, "estimatedDamage": None},
        "claimType": None,
        "attachments": [],
        "initialEstimate": None,
    }


def _compute_confidence(fields: dict, raw_text: str) -> float:
    """Simple heuristic: ratio of populated critical fields."""
    critical = [
        fields.get("policyNumber"),
        fields.get("policyholderName"),
        (fields.get("incident") or {}).get("date"),
        (fields.get("incident") or {}).get("location"),
        (fields.get("incident") or {}).get("description"),
        (fields.get("asset") or {}).get("type"),
        fields.get("claimType"),
        fields.get("initialEstimate"),
    ]
    populated = sum(1 for v in critical if v not in (None, "", [], {}))
    base = populated / len(critical)
    # Bonus for text quality
    if len(raw_text) > 200:
        base = min(1.0, base + 0.05)
    return round(base, 2)


async def summarize_claim(fields: dict) -> str:
    """Generate a short AI summary of the claim."""
    key = _get_llm_key()
    if not key:
        return ""
    session_id = f"claim-summary-{uuid.uuid4()}"
    chat = LlmChat(
        api_key=key,
        session_id=session_id,
        system_message="You are an insurance analyst. Given structured FNOL data, write a single concise sentence (max 25 words) describing the incident, severity, and any red flags. No preamble.",
    ).with_model("anthropic", "claude-sonnet-4-5-20250929")

    try:
        response = await chat.send_message(
            UserMessage(text=f"Summarize this claim:\n{json.dumps(fields, default=str)}")
        )
        return response.strip().strip('"')
    except Exception as e:
        logger.warning("Summary generation failed: %s", e)
        return ""
