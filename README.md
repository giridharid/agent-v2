# Smaartbrand Hotels Intelligence Dashboard - Deployment

## Files Included
- `main.py` - FastAPI backend (14 endpoints + caching)
- `index.html` - Frontend (drill-down, emotions, R&D signals, chat)
- `agent_prompts.py` - Multi-vertical SmaartAnalyst prompts
- `acquink_logo.png` - Logo (favicon, header, footer)
- `requirements.txt` - Python dependencies
- `Procfile` - Railway startup command

## Railway Deployment

### 1. Create New Project
```bash
railway login
railway init
```

### 2. Set Environment Variable
```bash
# Base64 encode your service account JSON
cat gen-lang-client-0143536012-cfdb14673cb4.json | base64 | pbcopy

# Set in Railway
railway variables set GCP_CREDENTIALS_JSON='<paste base64 string>'
```

That's it! No separate GEMINI_API_KEY needed — uses service account via Vertex AI.

### 3. Deploy
```bash
railway up
```

### 4. Get Public URL
```bash
railway domain
```

## BigQuery Tables Required
Project: `gen-lang-client-0143536012`
Dataset: `smaartanalyst`

1. brand_master (~120 rows)
2. hotel_master (~3,300 rows)
3. product_aspect_summary (~25,000 rows)
4. product_demographics (~28,000 rows)
5. product_segment_aspect (~73,000 rows)
6. product_phrases (~420,000 rows)
7. product_pain_delights (~285,000 rows)
8. product_emotions (~20,000 rows)
9. product_rd_signals (~17,000 rows)
10. review_drilldown (5,337,005 rows) ← NEW

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/brands | GET | List brands |
| /api/hotels | GET | List hotels with filters |
| /api/search | GET | Wildcard hotel search |
| /api/brand/{id}/summary | GET | Brand-level summary |
| /api/hotel/{id}/summary | GET | Hotel-level summary |
| /api/drilldown | POST | Reviews for phrase/signal |
| /api/compare/hotels | POST | Compare 2-3 hotels |
| /api/chat | POST | SmaartAnalyst chat |
| /health | GET | Health check |

## Features
✅ Clickable pain points → drill-down to reviews (5.3M reviews)
✅ Clickable delights → drill-down to reviews
✅ Clickable R&D signals → drill-down to reviews
✅ Sentiment text highlighting in yellow
✅ Top emotions bar (Plutchik 8)
✅ Guest demographics (traveler type, gender, stay purpose)
✅ SmaartAnalyst chat (Gemini 2.0 Flash via Vertex AI)
✅ Hotel comparison (2-3 hotels)
✅ Brand comparison (2-3 brands)
✅ In-memory caching (24h refresh)
✅ Multi-vertical support (hotels, bikes, autos, smartphones)
✅ Acquink branding (favicon, header, footer)

## Local Testing
```bash
pip install -r requirements.txt
export GCP_CREDENTIALS_JSON=$(cat /path/to/service-account.json | base64)
uvicorn main:app --reload --port 8080
```

Open http://localhost:8080
