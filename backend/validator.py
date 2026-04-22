"""Mandatory field validator for FNOL claims."""
from typing import List

# Dotted-path → Human label. Missing if value is None or empty.
MANDATORY_FIELDS = {
    "policyNumber": "Policy Number",
    "policyholderName": "Policyholder Name",
    "incident.date": "Date of Loss",
    "incident.time": "Time of Loss",
    "incident.location": "Incident Location",
    "incident.description": "Incident Description",
    "involvedParties.claimant.name": "Claimant Name",
    "involvedParties.claimant.contact": "Claimant Contact",
    "asset.type": "Asset Type",
    "asset.estimatedDamage": "Estimated Damage",
    "claimType": "Claim Type",
    "initialEstimate": "Initial Estimate",
}


def _resolve(data: dict, path: str):
    cur = data
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict)) and not value:
        return True
    return False


def find_missing_fields(fields: dict) -> List[str]:
    """Return human-readable labels of missing mandatory fields."""
    missing = []
    for path, label in MANDATORY_FIELDS.items():
        if _is_missing(_resolve(fields, path)):
            missing.append(label)
    return missing
