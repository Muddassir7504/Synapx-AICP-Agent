"""Backend tests for Synapx Autonomous Insurance Claims Processing Agent.

Covers:
- Health root
- POST /api/claims/process (TXT, PDF, invalid types, empty)
- GET /api/samples
- POST /api/samples/{id}/process for all 5 sample ids (routing validation)
- GET /api/claims, GET /api/claims/{id}, GET /api/claims/stats
- DELETE /api/claims/{id}
- Routing rule priorities (missing > fraud > injury > fast-track > standard)
- MongoDB persistence (no _id leakage)
"""
import io
import os
import time
from pathlib import Path

import pytest
import requests

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "").rstrip("/")
if not BASE_URL:
    # Fallback – read frontend env
    from dotenv import dotenv_values
    vals = dotenv_values("/app/frontend/.env")
    BASE_URL = (vals.get("REACT_APP_BACKEND_URL") or "").rstrip("/")

API = f"{BASE_URL}/api"
SAMPLES_DIR = Path("/app/backend/samples")
TIMEOUT = 120  # LLM calls can be slow


# ---------- Fixtures ----------
@pytest.fixture(scope="session")
def session():
    s = requests.Session()
    return s


@pytest.fixture(scope="session")
def created_ids():
    """Track ids created during tests so we can clean them up at the end."""
    ids = []
    yield ids
    # teardown — attempt to delete any leftover test claims
    for cid in ids:
        try:
            requests.delete(f"{API}/claims/{cid}", timeout=15)
        except Exception:
            pass


# ---------- Health ----------
def test_health_root(session):
    r = session.get(f"{API}/", timeout=15)
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"
    assert "Synapx" in data.get("message", "")


# ---------- Samples list ----------
def test_list_samples(session):
    r = session.get(f"{API}/samples", timeout=15)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    ids = {s["id"] for s in data}
    assert ids == {"fast-track", "standard", "specialist", "investigation", "manual"}
    for s in data:
        assert s["title"] and s["description"] and s["expectedRoute"] and s["filename"]


# ---------- Sample processing (all 5 routes) ----------
EXPECTED = {
    "fast-track": "Fast-track",
    "standard": "Standard Processing",
    "specialist": "Specialist Queue",
    "investigation": "Investigation Flag",
    "manual": "Manual Review",
}


def _assert_full_schema(doc):
    required = [
        "id", "filename", "createdAt", "extractedFields", "missingFields",
        "recommendedRoute", "reasoning", "confidenceScore", "fraudRisk",
        "fraudRiskScore", "fraudKeywordHits", "aiSummary", "rawTextPreview",
    ]
    for k in required:
        assert k in doc, f"Missing key in response: {k}"
    # Shouldn't leak mongo _id
    assert "_id" not in doc


@pytest.mark.parametrize("sample_id", list(EXPECTED.keys()))
def test_sample_process_routing(session, sample_id, created_ids):
    r = session.post(f"{API}/samples/{sample_id}/process", timeout=TIMEOUT)
    assert r.status_code == 200, f"{sample_id} -> {r.status_code} {r.text[:400]}"
    doc = r.json()
    _assert_full_schema(doc)
    created_ids.append(doc["id"])
    assert doc["recommendedRoute"] == EXPECTED[sample_id], (
        f"{sample_id}: expected {EXPECTED[sample_id]}, got {doc['recommendedRoute']}. "
        f"missing={doc['missingFields']} fraudHits={doc['fraudKeywordHits']}"
    )

    # route-specific deeper assertions
    if sample_id == "investigation":
        assert doc["fraudKeywordHits"], "Investigation sample should have fraud keyword hits"
        assert doc["fraudRisk"] == "High", f"expected High fraud risk, got {doc['fraudRisk']}"
    if sample_id == "manual":
        assert len(doc["missingFields"]) >= 8, (
            f"Manual sample should have 8+ missing fields, got {len(doc['missingFields'])}: {doc['missingFields']}"
        )
    if sample_id == "fast-track":
        est = (doc["extractedFields"].get("asset") or {}).get("estimatedDamage") or doc["extractedFields"].get("initialEstimate")
        assert est is not None and float(est) < 25000


def test_sample_invalid_id(session):
    r = session.post(f"{API}/samples/does-not-exist/process", timeout=30)
    assert r.status_code == 404


# ---------- File upload processing ----------
def test_process_txt_upload(session, created_ids):
    path = SAMPLES_DIR / "fast_track_auto.txt"
    with open(path, "rb") as f:
        files = {"file": ("fast_track_auto.txt", f, "text/plain")}
        r = session.post(f"{API}/claims/process", files=files, timeout=TIMEOUT)
    assert r.status_code == 200, r.text[:400]
    doc = r.json()
    _assert_full_schema(doc)
    created_ids.append(doc["id"])
    assert doc["filename"] == "fast_track_auto.txt"
    assert doc["rawTextPreview"]
    assert doc["recommendedRoute"] == "Fast-track"


def test_process_pdf_upload(session, created_ids):
    """Build a simple PDF in-memory from txt content and upload."""
    import fitz
    txt = (SAMPLES_DIR / "standard_property.txt").read_text()
    pdf = fitz.open()
    # PyMuPDF: need to insert text into a page
    page = pdf.new_page(width=612, height=792)
    # insert_textbox wraps long text
    page.insert_textbox(fitz.Rect(36, 36, 576, 756), txt, fontsize=9)
    buf = io.BytesIO(pdf.tobytes())
    pdf.close()
    buf.seek(0)
    files = {"file": ("standard_property.pdf", buf, "application/pdf")}
    r = session.post(f"{API}/claims/process", files=files, timeout=TIMEOUT)
    assert r.status_code == 200, r.text[:400]
    doc = r.json()
    _assert_full_schema(doc)
    created_ids.append(doc["id"])
    assert doc["filename"].endswith(".pdf")
    assert doc["rawTextPreview"], "PDF text should be extracted"


def test_process_unsupported_file_type(session):
    files = {"file": ("image.jpg", io.BytesIO(b"\xff\xd8\xff\xe0fakejpeg"), "image/jpeg")}
    r = session.post(f"{API}/claims/process", files=files, timeout=30)
    assert r.status_code == 400
    assert "supported" in r.text.lower() or "pdf" in r.text.lower()


def test_process_empty_file(session):
    files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
    r = session.post(f"{API}/claims/process", files=files, timeout=30)
    assert r.status_code == 400


# ---------- Listing / Stats / Persistence ----------
def test_list_claims_sorted(session, created_ids):
    # Ensure there is at least 2 claims; we should already have some from prior tests
    r = session.get(f"{API}/claims", timeout=15)
    assert r.status_code == 200
    items = r.json()
    assert isinstance(items, list)
    assert len(items) >= 1
    # Check sort DESC by createdAt
    timestamps = [it["createdAt"] for it in items]
    assert timestamps == sorted(timestamps, reverse=True)
    # _id should not leak
    for it in items:
        assert "_id" not in it


def test_get_claim_full_persistence(session, created_ids):
    assert created_ids, "Expected prior tests to have created claims"
    cid = created_ids[0]
    r = session.get(f"{API}/claims/{cid}", timeout=15)
    assert r.status_code == 200
    doc = r.json()
    _assert_full_schema(doc)
    assert doc["id"] == cid


def test_get_claim_not_found(session):
    r = session.get(f"{API}/claims/nonexistent-xyz-1234", timeout=15)
    assert r.status_code == 404


def test_stats_endpoint(session):
    r = session.get(f"{API}/claims/stats", timeout=15)
    assert r.status_code == 200
    data = r.json()
    for k in ["totalClaims", "fastTrackPercent", "fraudFlags",
              "manualReviews", "specialistQueue", "standardProcessing"]:
        assert k in data
    assert data["totalClaims"] >= 1
    assert 0.0 <= data["fastTrackPercent"] <= 100.0


# ---------- Routing priority: missing field + fraud keyword -> Manual Review ----------
def test_routing_priority_missing_over_fraud(session, created_ids):
    """
    A claim that has BOTH missing mandatory fields AND a fraud keyword should
    be routed to Manual Review (missing wins over fraud).
    We use the 'manual' sample which is partial/incomplete and inject a fraud
    keyword into its incident description via a custom TXT upload.
    """
    base = (SAMPLES_DIR / "manual_review_incomplete.txt").read_text()
    # Append a clearly suspicious phrase so LLM sees it and sets description
    doctored = base + "\n\nNote from adjuster: story appears inconsistent and possibly staged."
    files = {"file": ("manual_with_fraud.txt", io.BytesIO(doctored.encode("utf-8")), "text/plain")}
    r = session.post(f"{API}/claims/process", files=files, timeout=TIMEOUT)
    assert r.status_code == 200, r.text[:400]
    doc = r.json()
    created_ids.append(doc["id"])
    # Must route to Manual Review regardless of fraud hits
    assert doc["recommendedRoute"] == "Manual Review", (
        f"Expected Manual Review (missing > fraud), got {doc['recommendedRoute']}. "
        f"missing={doc['missingFields']} fraudHits={doc['fraudKeywordHits']}"
    )


# ---------- Delete flow ----------
def test_delete_claim(session):
    # Create a fresh claim to delete
    path = SAMPLES_DIR / "fast_track_auto.txt"
    with open(path, "rb") as f:
        files = {"file": ("fast_track_for_delete.txt", f, "text/plain")}
        r = session.post(f"{API}/claims/process", files=files, timeout=TIMEOUT)
    assert r.status_code == 200
    cid = r.json()["id"]

    r2 = session.delete(f"{API}/claims/{cid}", timeout=15)
    assert r2.status_code == 200
    assert r2.json().get("deleted") is True

    r3 = session.get(f"{API}/claims/{cid}", timeout=15)
    assert r3.status_code == 404

    # Deleting again should 404
    r4 = session.delete(f"{API}/claims/{cid}", timeout=15)
    assert r4.status_code == 404
