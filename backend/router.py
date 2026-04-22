"""Routing engine: decides claim workflow + computes fraud risk."""
import re
from typing import List, Tuple

FRAUD_KEYWORDS = [
    "fraud", "fraudulent", "inconsistent", "staged", "fake",
    "suspicious", "forged", "exaggerated", "misrepresent",
]

FAST_TRACK_THRESHOLD = 25000


def detect_fraud_keywords(text: str) -> List[str]:
    if not text:
        return []
    lower = text.lower()
    return [kw for kw in FRAUD_KEYWORDS if re.search(rf"\b{re.escape(kw)}\w*\b", lower)]


def compute_fraud_risk(fraud_hits: List[str], fields: dict) -> Tuple[int, str]:
    """Return (risk_percent 0-100, risk_level 'Low'|'Medium'|'High')."""
    base = min(len(fraud_hits) * 30, 90)

    # Bonus risk: very high damage claims
    est = (fields.get("asset") or {}).get("estimatedDamage") or fields.get("initialEstimate") or 0
    try:
        est = float(est)
    except (TypeError, ValueError):
        est = 0
    if est >= 100000:
        base += 10

    base = max(0, min(100, base))
    if base >= 60:
        level = "High"
    elif base >= 30:
        level = "Medium"
    else:
        level = "Low"
    return base, level


def route_claim(fields: dict, missing_fields: List[str]) -> Tuple[str, str, List[str]]:
    """
    Apply routing rules in priority order.
    Returns (recommendedRoute, reasoning, fraudHits).
    """
    description = ((fields.get("incident") or {}).get("description") or "")
    fraud_hits = detect_fraud_keywords(description)

    # Rule 1: Missing mandatory fields
    if missing_fields:
        reasoning = (
            f"Routed to Manual Review because {len(missing_fields)} mandatory "
            f"field(s) are missing: {', '.join(missing_fields[:4])}"
            f"{'...' if len(missing_fields) > 4 else ''}."
        )
        return "Manual Review", reasoning, fraud_hits

    # Rule 2: Fraud keywords
    if fraud_hits:
        reasoning = (
            f"Flagged for Investigation — suspicious language detected in the "
            f"incident description: {', '.join(sorted(set(fraud_hits)))}."
        )
        return "Investigation Flag", reasoning, fraud_hits

    # Rule 3: Injury claims
    claim_type = (fields.get("claimType") or "").lower()
    if claim_type == "injury":
        reasoning = "Routed to Specialist Queue because the claim type is 'injury' and requires specialist review."
        return "Specialist Queue", reasoning, fraud_hits

    # Rule 4: Damage threshold
    est = (fields.get("asset") or {}).get("estimatedDamage")
    if est is None:
        est = fields.get("initialEstimate")
    try:
        est_val = float(est) if est is not None else None
    except (TypeError, ValueError):
        est_val = None

    if est_val is not None and est_val < FAST_TRACK_THRESHOLD:
        reasoning = (
            f"Fast-tracked because the estimated damage of ₹{est_val:,.0f} is below "
            f"the ₹{FAST_TRACK_THRESHOLD:,} fast-track threshold and all mandatory fields are present."
        )
        return "Fast-track", reasoning, fraud_hits

    # Rule 5: Default
    if est_val is not None:
        reasoning = (
            f"Routed to Standard Processing. All mandatory fields present; "
            f"estimated damage of ₹{est_val:,.0f} exceeds the fast-track threshold."
        )
    else:
        reasoning = "Routed to Standard Processing with all mandatory fields present."
    return "Standard Processing", reasoning, fraud_hits
