# Synapx-AICP-Agent
AI-powered insurance claims processing app that reads FNOL PDF/TXT files, extracts claim details, detects missing fields and fraud indicators, routes claims automatically (Fast-track/Review/Investigation), and returns explainable JSON results through a modern dashboard using React + FastAPI.

 1 · End-to-end workflow (A → Z)
┌────────────┐   upload PDF/TXT    ┌──────────────────────┐
│  Browser   │ ──────────────────▶ │  FastAPI /claims/    │
│ (React UI) │                     │      process         │
└────────────┘                     └──────────┬───────────┘
                                              │
                 ┌────────────────────────────┼───────────────────────────────┐
                 ▼                            ▼                               ▼
        ┌────────────────┐         ┌────────────────────┐           ┌────────────────┐
        │ extractor.py   │         │    validator.py    │           │   router.py    │
        │                │         │                    │           │                │
        │ ① PyMuPDF      │         │ ② find missing     │           │ ③ priority     │
        │   extract text │         │   mandatory fields │           │   routing      │
        │   +AcroForm    │         └────────────────────┘           │   rules +      │
        │                │                                          │   fraud score  │
        │ ② LLM (Claude  │                                          └────────────────┘
        │   Sonnet 4.5)  │                                                │
        │   structured   │                                                │
        │   JSON         │         ┌────────────────────┐                 │
        │                │ ◀────── │  summarize_claim   │                 │
        │ ③ summarizer   │         │  (LLM 1-liner)     │                 │
        └────────────────┘         └────────────────────┘                 │
                 │                                                        │
                 ▼                                                        ▼
        ┌──────────────────────────────────────────────────────────────────────┐
        │   server.py  →  assemble ClaimFull  →  insert into MongoDB           │
        │   response:   extractedFields, missingFields, recommendedRoute,     │
        │               reasoning, confidenceScore, fraudRisk, aiSummary       │
        └──────────────────────────────────────────────────────────────────────┘
                 │
                 ▼
        ┌──────────────────────────────────────────────────────────────────────┐
        │   React Results page                                                 │
        │   • field cards (missing ones pulse red)                             │
        │   • route badge  • confidence bar  • fraud radar (Recharts)          │
        │   • "Download JSON" button                                           │
        └──────────────────────────────────────────────────────────────────────┘
## System Workflow & Architecture

The **Autonomous Insurance Claims Processing Agent** is designed to automate the complete insurance claim intake process from document upload to final routing decision. Users upload a **PDF** or **TXT** FNOL (First Notice of Loss) document through the React frontend dashboard. The uploaded file is sent to the FastAPI backend endpoint (`/api/claims/process`) for processing. The backend then extracts raw text from the file using a dedicated extraction layer. For TXT files, plain text is decoded using UTF-8 / Latin-1 formats. For PDF files, **PyMuPDF (fitz)** is used to extract visible page text as well as filled AcroForm fields, ensuring that structured insurance forms are also processed accurately.

Once the text is extracted, the content is passed to **Claude Sonnet 4.5** through Emergent AI integration. The AI model converts messy unstructured insurance text into a clean structured JSON response containing important fields such as policy number, claimant name, claim type, asset details, damage estimate, date of incident, and contact information. This removes the need for manual data entry and speeds up processing significantly.

After extraction, a validation engine checks whether all mandatory business fields are available. Important fields such as policy number, claimant contact, date of loss, asset type, claim type, and initial estimate are verified. If any information is missing, those fields are added to the `missingFields` list for manual attention.

Next, the routing engine applies predefined business rules to determine the best workflow queue for the claim. Claims with damage below ₹25,000 are sent to **Fast Track** processing. Claims with missing mandatory data are moved to **Manual Review**. Claims containing suspicious words such as fraud, staged, fake, forged, or inconsistent are routed to **Investigation**. Claims involving injuries are assigned to the **Specialist Queue**. Along with routing, the system also generates a fraud risk score categorized as Low, Medium, or High.

To improve usability, a second AI step generates a one-line plain-English claim summary. For example: *Vehicle accident claim with rear bumper damage estimated at ₹45,000.* This allows claim handlers to quickly understand the case without reading the full document.

All processed claims are then stored in **MongoDB** with a unique claim ID, timestamp, extracted fields, missing fields, route decision, confidence score, fraud score, and AI summary. Finally, the React frontend displays the results in a clean dashboard containing claim cards, route badges, missing field alerts, confidence progress bars, fraud charts, downloadable JSON reports, and CSV export functionality.

This architecture combines AI intelligence for document understanding with rule-based deterministic business logic, making it scalable, explainable, and production-ready.


## Workflow Summary Table

| Step | Component | Purpose |
|------|-----------|---------|
| 1 | React Frontend | Upload PDF/TXT claim documents |
| 2 | FastAPI Backend | Receives files and triggers processing |
| 3 | extractor.py | Extracts text from PDF/TXT documents |
| 4 | Claude Sonnet 4.5 | Converts unstructured text into structured JSON |
| 5 | validator.py | Detects missing mandatory fields |
| 6 | router.py | Applies routing rules and fraud detection |
| 7 | MongoDB | Stores processed claim records |
| 8 | React Dashboard | Displays final claim results |

---

## Routing Rules Table

| Condition | Recommended Route |
|----------|------------------|
| Estimated Damage < ₹25,000 | Fast Track |
| Missing Mandatory Fields | Manual Review |
| Fraud Keywords Found | Investigation |
| Claim Type = Injury | Specialist Queue |

---

## Key Technologies Used

| Category | Tools / Technologies |
|---------|----------------------|
| Frontend | React.js, Tailwind CSS |
| Backend | FastAPI, Python |
| AI Model | Claude Sonnet 4.5 |
| PDF Parsing | PyMuPDF |
| Database | MongoDB |
| Charts | Recharts |
| APIs | REST APIs |

---


## 3 · Code structure
/app
├── backend/
│   ├── server.py          # FastAPI app: /api routes, MongoDB wiring, CSV export
│   ├── extractor.py       # PDF/TXT text extraction + LLM field extraction + AI summary
│   ├── validator.py       # Mandatory-field audit
│   ├── router.py          # Priority-ordered routing + fraud risk
│   ├── samples/           # 5 demo FNOL .txt files (one per routing path)
│   ├── requirements.txt
│   └── .env               # MONGO_URL, DB_NAME, CORS_ORIGINS, EMERGENT_LLM_KEY
│
└── frontend/
    ├── package.json
    └── src/
        ├── App.js         # Router + ThemeProvider + Toaster
        ├── index.css      # Tailwind layers + CSS vars (light/dark) + animations
        ├── lib/
        │   ├── api.js     # Axios client + typed endpoint fns
        │   └── theme.jsx  # Dark/light context, persisted in localStorage
        ├── components/
        │   ├── Header.jsx
        │   ├── UploadDropzone.jsx   # drag-drop, queue, progress beam
        │   ├── SamplePicker.jsx     # 5 one-click demo samples
        │   ├── StatsCard.jsx
        │   ├── RouteBadge.jsx       # pill with per-route colour + dot
        │   └── Skeleton.jsx
        └── pages/
            ├── Landing.jsx    # Hero + animated backdrop + ticker + features
            ├── Dashboard.jsx  # Stats + picker + upload + claim table + CSV
            └── Results.jsx    # Bento grid: fields + routing + confidence + radar


## Final Outcome

The system acts like an **AI-powered insurance operations assistant** that automatically reads claims, extracts data, validates missing fields, detects fraud risk, routes claims to the correct team, and reduces manual workload significantly.
