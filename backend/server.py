"""FastAPI server for Autonomous Insurance Claims Processing Agent."""
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from pathlib import Path
from datetime import datetime, timezone
import os
import io
import csv
import uuid
import logging

from extractor import extract_raw_text, extract_fields_llm, summarize_claim
from validator import find_missing_fields
from router import route_claim, compute_fraud_risk

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")
SAMPLES_DIR = ROOT_DIR / "samples"

mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]
claims_col = db["claims"]

app = FastAPI(title="Synapx Claims Agent")
api_router = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------- Models ----------
class ClaimSummary(BaseModel):
    id: str
    filename: str
    createdAt: str
    recommendedRoute: str
    confidenceScore: float
    fraudRisk: str
    fraudRiskScore: int
    policyholderName: Optional[str] = None
    claimType: Optional[str] = None
    initialEstimate: Optional[float] = None


class ClaimFull(BaseModel):
    id: str
    filename: str
    createdAt: str
    extractedFields: Dict[str, Any]
    missingFields: List[str]
    recommendedRoute: str
    reasoning: str
    confidenceScore: float
    fraudRisk: str
    fraudRiskScore: int
    fraudKeywordHits: List[str]
    aiSummary: str
    rawTextPreview: str


class StatsResponse(BaseModel):
    totalClaims: int
    fastTrackPercent: float
    fraudFlags: int
    manualReviews: int
    specialistQueue: int
    standardProcessing: int


class SampleInfo(BaseModel):
    id: str
    title: str
    description: str
    expectedRoute: str
    filename: str


SAMPLES: List[Dict[str, str]] = [
    {
        "id": "fast-track",
        "title": "Fast-Track Auto",
        "description": "Minor rear-end collision, damage $3,200, fully documented.",
        "expectedRoute": "Fast-track",
        "filename": "fast_track_auto.txt",
    },
    {
        "id": "standard",
        "title": "Standard Property",
        "description": "Kitchen fire with $78k damage, complete documentation.",
        "expectedRoute": "Standard Processing",
        "filename": "standard_property.txt",
    },
    {
        "id": "specialist",
        "title": "Specialist — Injury",
        "description": "Multi-vehicle collision with whiplash and concussion.",
        "expectedRoute": "Specialist Queue",
        "filename": "specialist_injury.txt",
    },
    {
        "id": "investigation",
        "title": "Investigation — Suspicious",
        "description": "Garage fire with inconsistent, possibly staged circumstances.",
        "expectedRoute": "Investigation Flag",
        "filename": "investigation_fraud.txt",
    },
    {
        "id": "manual",
        "title": "Manual Review — Incomplete",
        "description": "Partial report missing policy, dates, and damage estimate.",
        "expectedRoute": "Manual Review",
        "filename": "manual_review_incomplete.txt",
    },
]


# ---------- Helpers ----------
async def _run_pipeline(filename: str, raw_bytes: bytes) -> Dict[str, Any]:
    """Core processing pipeline: text -> LLM -> validate -> route -> save."""
    try:
        raw_text = extract_raw_text(filename, raw_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse document: {e}")

    if not raw_text:
        raise HTTPException(status_code=400, detail="Document appears to be empty or unreadable.")

    try:
        fields, confidence = await extract_fields_llm(raw_text)
    except Exception as e:
        logger.exception("LLM extraction failed")
        raise HTTPException(status_code=500, detail=f"AI extraction failed: {e}")

    missing = find_missing_fields(fields)
    recommended_route, reasoning, fraud_hits = route_claim(fields, missing)
    fraud_score, fraud_level = compute_fraud_risk(fraud_hits, fields)

    try:
        ai_summary = await summarize_claim(fields)
    except Exception:
        ai_summary = ""

    claim_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    doc = {
        "id": claim_id,
        "filename": filename,
        "createdAt": created_at,
        "extractedFields": fields,
        "missingFields": missing,
        "recommendedRoute": recommended_route,
        "reasoning": reasoning,
        "confidenceScore": confidence,
        "fraudRisk": fraud_level,
        "fraudRiskScore": fraud_score,
        "fraudKeywordHits": fraud_hits,
        "aiSummary": ai_summary,
        "rawTextPreview": raw_text[:1500],
    }
    await claims_col.insert_one(doc.copy())
    return doc


def _to_summary(doc: dict) -> dict:
    fields = doc.get("extractedFields") or {}
    return {
        "id": doc["id"],
        "filename": doc["filename"],
        "createdAt": doc["createdAt"],
        "recommendedRoute": doc["recommendedRoute"],
        "confidenceScore": doc["confidenceScore"],
        "fraudRisk": doc["fraudRisk"],
        "fraudRiskScore": doc["fraudRiskScore"],
        "policyholderName": fields.get("policyholderName"),
        "claimType": fields.get("claimType"),
        "initialEstimate": fields.get("initialEstimate"),
    }


# ---------- Routes ----------
@api_router.get("/")
async def root():
    return {"message": "Synapx Claims Agent API", "status": "ok"}


@api_router.post("/claims/process", response_model=ClaimFull)
async def process_claim(file: UploadFile = File(...)):
    """Upload a PDF or TXT FNOL document and run the full pipeline."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in (".pdf", ".txt"):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    doc = await _run_pipeline(file.filename, raw_bytes)
    return ClaimFull(**doc)


@api_router.get("/samples", response_model=List[SampleInfo])
async def list_samples():
    return [SampleInfo(**s) for s in SAMPLES]


@api_router.post("/samples/{sample_id}/process", response_model=ClaimFull)
async def process_sample(sample_id: str):
    match = next((s for s in SAMPLES if s["id"] == sample_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Sample not found")
    path = SAMPLES_DIR / match["filename"]
    if not path.exists():
        raise HTTPException(status_code=500, detail="Sample file missing on server")
    raw_bytes = path.read_bytes()
    doc = await _run_pipeline(match["filename"], raw_bytes)
    return ClaimFull(**doc)


@api_router.get("/claims", response_model=List[ClaimSummary])
async def list_claims(limit: int = 50):
    cursor = claims_col.find({}, {"_id": 0}).sort("createdAt", -1).limit(limit)
    items = await cursor.to_list(length=limit)
    return [ClaimSummary(**_to_summary(d)) for d in items]


@api_router.get("/claims/stats", response_model=StatsResponse)
async def claim_stats():
    total = await claims_col.count_documents({})
    if total == 0:
        return StatsResponse(
            totalClaims=0, fastTrackPercent=0.0, fraudFlags=0,
            manualReviews=0, specialistQueue=0, standardProcessing=0,
        )
    fast = await claims_col.count_documents({"recommendedRoute": "Fast-track"})
    fraud = await claims_col.count_documents({"recommendedRoute": "Investigation Flag"})
    manual = await claims_col.count_documents({"recommendedRoute": "Manual Review"})
    specialist = await claims_col.count_documents({"recommendedRoute": "Specialist Queue"})
    standard = await claims_col.count_documents({"recommendedRoute": "Standard Processing"})

    return StatsResponse(
        totalClaims=total,
        fastTrackPercent=round((fast / total) * 100, 1),
        fraudFlags=fraud,
        manualReviews=manual,
        specialistQueue=specialist,
        standardProcessing=standard,
    )


def _inr(value) -> str:
    """Format a number as Indian rupee string with Indian lakh/crore grouping."""
    if value is None or value == "":
        return ""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return str(value)
    if n == int(n):
        n = int(n)
    # Indian number format — e.g. 1,23,45,678
    s = f"{n:,.0f}" if isinstance(n, int) else f"{n:,.2f}"
    # convert western grouping to Indian grouping
    if "." in s:
        intpart, dec = s.split(".")
    else:
        intpart, dec = s, None
    intpart = intpart.replace(",", "")
    neg = intpart.startswith("-")
    if neg:
        intpart = intpart[1:]
    if len(intpart) > 3:
        last3 = intpart[-3:]
        rest = intpart[:-3]
        rest_grouped = ",".join([rest[max(i - 2, 0):i] for i in range(len(rest), 0, -2)][::-1])
        intpart = f"{rest_grouped},{last3}"
    if neg:
        intpart = "-" + intpart
    out = f"₹{intpart}" + (f".{dec}" if dec else "")
    return out


@api_router.get("/claims/export.csv")
async def export_claims_csv():
    """Download all processed claims as a CSV file with INR-formatted monetary values."""
    cursor = claims_col.find({}, {"_id": 0}).sort("createdAt", -1)
    rows = await cursor.to_list(length=10000)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "Claim ID", "Filename", "Processed At",
        "Policy Number", "Policyholder Name",
        "Effective Date (Loss Date)", "Effective Time (Loss Time)",
        "Incident Location", "Incident Description",
        "Claimant", "Claimant Contact", "Third Parties",
        "Asset Type", "Asset ID", "Estimated Damage (INR)",
        "Claim Type", "Initial Estimate (INR)", "Attachments",
        "Recommended Route", "Reasoning",
        "Confidence Score", "Fraud Risk Level", "Fraud Risk Score",
        "Missing Fields", "Fraud Keyword Hits", "AI Summary",
    ])
    for d in rows:
        f = d.get("extractedFields") or {}
        incident = f.get("incident") or {}
        ip = f.get("involvedParties") or {}
        claimant = ip.get("claimant") or {}
        third = "; ".join(
            f"{t.get('name') or ''} ({t.get('contact') or ''})".strip()
            for t in (ip.get("thirdParties") or [])
        )
        asset = f.get("asset") or {}

        writer.writerow([
            d.get("id", ""),
            d.get("filename", ""),
            d.get("createdAt", ""),
            f.get("policyNumber") or "",
            f.get("policyholderName") or "",
            incident.get("date") or "",
            incident.get("time") or "",
            incident.get("location") or "",
            (incident.get("description") or "").replace("\n", " "),
            claimant.get("name") or "",
            claimant.get("contact") or "",
            third,
            asset.get("type") or "",
            asset.get("id") or "",
            _inr(asset.get("estimatedDamage")),
            f.get("claimType") or "",
            _inr(f.get("initialEstimate")),
            "; ".join(f.get("attachments") or []),
            d.get("recommendedRoute", ""),
            d.get("reasoning", ""),
            f"{int(round((d.get('confidenceScore') or 0) * 100))}%",
            d.get("fraudRisk", ""),
            d.get("fraudRiskScore", 0),
            "; ".join(d.get("missingFields") or []),
            "; ".join(d.get("fraudKeywordHits") or []),
            d.get("aiSummary", ""),
        ])

    buf.seek(0)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="synapx-claims-{stamp}.csv"'},
    )


@api_router.get("/claims/{claim_id}", response_model=ClaimFull)
async def get_claim(claim_id: str):
    doc = await claims_col.find_one({"id": claim_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Claim not found")
    return ClaimFull(**doc)


@api_router.delete("/claims/{claim_id}")
async def delete_claim(claim_id: str):
    result = await claims_col.delete_one({"id": claim_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Claim not found")
    return {"deleted": True, "id": claim_id}


app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
