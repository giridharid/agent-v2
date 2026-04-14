"""
Smaartbrand Intelligence Dashboard - FastAPI Backend v2
Multi-vertical support with BigQuery flat tables, caching, and drill-down
"""

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import bigquery
from google.oauth2 import service_account
import google.generativeai as genai
import json
import os
import base64
import traceback
import threading
import time
import asyncio
import math
import uuid
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from functools import lru_cache

# Import agent prompts
from agent_prompts import get_agent_prompt, DATA_CONTEXT_TEMPLATE, DRILLDOWN_CONTEXT_TEMPLATE


def clean_val(v):
    """Convert BigQuery/numpy types to JSON-serializable Python types"""
    import math as _math
    if v is None:
        return None
    if hasattr(v, 'isoformat'):
        return v.isoformat()
    if hasattr(v, 'item'):
        v = v.item()
    if isinstance(v, float) and (_math.isnan(v) or _math.isinf(v)):
        return None
    return v

def clean_row(row_dict):
    return {k: clean_val(v) for k, v in row_dict.items()}

app = FastAPI(title="Smaartbrand Intelligence API", version="2.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
PROJECT = "gen-lang-client-0143536012"
DATASET = "smaartanalyst"
AGENT_ID = os.environ.get("GEMINI_AGENT_ID", "agent_024eedb9-0b86-4101-82a5-f4b3d72c5ee3")
AGENT_LOCATION = "global"  # Data Analytics Agent uses global, not regional

ASPECT_MAP = {
    1: "Dining", 2: "Cleanliness", 3: "Amenities", 4: "Staff",
    5: "Room", 6: "Location", 7: "Value for Money", 8: "General"
}

ASPECT_ICONS = {
    "Dining": "🍽️", "Cleanliness": "🧹", "Amenities": "🏊", "Staff": "👨‍💼",
    "Room": "🛏️", "Location": "📍", "Value for Money": "💰", "General": "⭐"
}

EMOTION_COLORS = {
    "joy": "#22c55e", "trust": "#3b82f6", "fear": "#6366f1", "surprise": "#a855f7",
    "sadness": "#6b7280", "disgust": "#84cc16", "anger": "#ef4444", "anticipation": "#f59e0b"
}

# ─────────────────────────────────────────
# CACHING
# ─────────────────────────────────────────
CACHE = {
    "brands":           {"data": None, "timestamp": 0},
    "hotels":           {"data": None, "timestamp": 0},
    "cities":           {"data": None, "timestamp": 0},
    "aspects_by_pid":   {"data": None, "timestamp": 0},  # dict: product_id -> [aspects]
    "aspects_by_brand": {"data": None, "timestamp": 0},  # dict: brand_name -> [aspects aggregated]
    "emotions_by_pid":  {"data": None, "timestamp": 0},  # dict: product_id -> [emotions]
    "pain_by_pid":      {"data": None, "timestamp": 0},  # dict: product_id -> [pain_points]
    "delight_by_pid":   {"data": None, "timestamp": 0},  # dict: product_id -> [delights]
    "demo_by_pid":      {"data": None, "timestamp": 0},  # dict: product_id -> {gender,traveler_type,stay_purpose}
}
CACHE_TTL = 86400  # 24 hours

def is_cache_valid(key: str) -> bool:
    if key not in CACHE or CACHE[key]["data"] is None:
        return False
    return (time.time() - CACHE[key]["timestamp"]) < CACHE_TTL

def get_cache(key: str):
    if is_cache_valid(key):
        return CACHE[key]["data"]
    return None

def set_cache(key: str, data: Any):
    CACHE[key] = {"data": data, "timestamp": time.time()}

# ─────────────────────────────────────────
# CREDENTIALS & CLIENTS
# ─────────────────────────────────────────
bq_client = None
gemini_model = None

def init_bq_client():
    global bq_client
    if bq_client is not None:
        return bq_client
    
    gcp_creds = os.environ.get("GCP_CREDENTIALS_JSON", "")
    if not gcp_creds:
        print("[ERROR] GCP_CREDENTIALS_JSON not set")
        return None
    
    gcp_creds = gcp_creds.strip().strip('"').strip("'")
    
    try:
        if gcp_creds.startswith("{"):
            creds_dict = json.loads(gcp_creds)
        else:
            padding = 4 - len(gcp_creds) % 4
            if padding != 4:
                gcp_creds += "=" * padding
            creds_dict = json.loads(base64.b64decode(gcp_creds).decode('utf-8'))
        
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        bq_client = bigquery.Client(credentials=credentials, project=credentials.project_id)
        print(f"[SUCCESS] BigQuery client initialized for project: {credentials.project_id}")
        return bq_client
    except Exception as e:
        print(f"[ERROR] BQ credential error: {e}")
        traceback.print_exc()
        return None

def get_gcp_credentials():
    """Parse GCP service-account credentials from env. Returns (creds, project_id) or (None, None)."""
    raw = os.environ.get("GCP_CREDENTIALS_JSON", "").strip().strip('"').strip("'")
    if not raw:
        return None, None
    try:
        if raw.startswith("{"):
            d = json.loads(raw)
        else:
            padding = 4 - len(raw) % 4
            if padding != 4:
                raw += "=" * padding
            d = json.loads(base64.b64decode(raw).decode("utf-8"))
        creds = service_account.Credentials.from_service_account_info(d)
        return creds, d.get("project_id", PROJECT)
    except Exception as e:
        print(f"[ERROR] GCP creds parse: {e}")
        return None, None


def get_data_chat_client():
    """Return a geminidataanalytics DataChatServiceClient using service-account creds."""
    try:
        from google.cloud import geminidataanalytics_v1alpha as gda
        creds, _ = get_gcp_credentials()
        if creds:
            return gda.DataChatServiceClient(credentials=creds)
        return gda.DataChatServiceClient()   # ADC fallback
    except Exception as e:
        print(f"[CHAT CLIENT] geminidataanalytics unavailable: {e}")
        return None


def init_gemini():
    """
    Gemini fallback model for when the Data Analytics Agent is unavailable.
    The main chat path uses get_data_chat_client() + geminidataanalytics_v1alpha.
    This model is only used if that fails.
    """
    global gemini_model
    creds, project_id = get_gcp_credentials()

    # Vertex AI GenerativeModel (2.5-flash → 2.0-flash → 1.5-flash)
    if creds:
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
            vertexai.init(project=project_id, location="us-central1", credentials=creds)
            for name in ["gemini-2.5-flash-preview-04-17", "gemini-2.0-flash", "gemini-1.5-flash"]:
                try:
                    gemini_model = GenerativeModel(name)
                    print(f"[GEMINI FALLBACK] Vertex AI model: {name}")
                    return gemini_model
                except Exception:
                    pass
        except Exception as e:
            print(f"[GEMINI FALLBACK] Vertex AI failed: {e}")

    # API-key fallback
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        print("[GEMINI FALLBACK] API key model")
        return gemini_model

    print("[WARN] No Gemini fallback available")
    return None


def get_bq():
    global bq_client
    if bq_client is None:
        init_bq_client()
    return bq_client

def get_gemini():
    global gemini_model
    if gemini_model is None:
        init_gemini()
    return gemini_model

# ─────────────────────────────────────────
# STARTUP & CACHE LOADING
# ─────────────────────────────────────────
@app.on_event("startup")
async def startup():
    init_bq_client()
    init_gemini()
    # Load caches in background
    threading.Thread(target=load_master_caches, daemon=True).start()

def load_master_caches():
    """Load brand_master and hotel_master into cache at startup"""
    client = get_bq()
    if not client:
        return
    
    try:
        # Load brands
        brands_query = f"""
        SELECT brand_id, brand_name, categories
        FROM `{PROJECT}.{DATASET}.brand_master`
        ORDER BY brand_name
        """
        brands_df = client.query(brands_query).to_dataframe()
        set_cache("brands", brands_df.to_dict(orient='records'))
        print(f"[CACHE] Loaded {len(brands_df)} brands")
        
        # Load hotels
        hotels_query = f"""
        SELECT product_id, hotel_name, brand_id, brand_name, city, state, country,
               star_category, review_count, overall_satisfaction
        FROM `{PROJECT}.{DATASET}.hotel_master`
        ORDER BY hotel_name
        """
        hotels_df = client.query(hotels_query).to_dataframe()
        set_cache("hotels", hotels_df.to_dict(orient='records'))
        print(f"[CACHE] Loaded {len(hotels_df)} hotels")
        
        # Extract unique cities
        cities = sorted(hotels_df['city'].dropna().unique().tolist())
        set_cache("cities", cities)
        print(f"[CACHE] Loaded {len(cities)} cities")

        # Build hotel_name → brand_name lookup
        hotel_to_brand = dict(zip(
            hotels_df['product_id'].astype(str),
            hotels_df['brand_name'].fillna('')
        ))

        # ── Load product_aspect_summary (all products + brands) ──
        asp_df = client.query(f"""
            SELECT product_id, aspect_id, aspect_name,
                   positive_count, negative_count, total_mentions,
                   satisfaction_pct, share_of_voice_pct
            FROM `{PROJECT}.{DATASET}.product_aspect_summary`
        """).to_dataframe()

        asp_by_pid = {}
        asp_brand_raw = {}  # brand_name -> {aspect_name -> {pos,neg,total}}
        for _, row in asp_df.iterrows():
            pid = str(int(row['product_id']))
            brand = hotel_to_brand.get(pid, '')
            r = {
                'aspect_id': int(row['aspect_id'] or 0),
                'aspect_name': str(row['aspect_name'] or ''),
                'positive_count': int(row['positive_count'] or 0),
                'negative_count': int(row['negative_count'] or 0),
                'total_mentions': int(row['total_mentions'] or 0),
                'satisfaction_pct': int(row['satisfaction_pct'] or 0),
                'share_of_voice_pct': int(row['share_of_voice_pct'] or 0),
            }
            asp_by_pid.setdefault(pid, []).append(r)
            if brand:
                ba = asp_brand_raw.setdefault(brand, {})
                ak = r['aspect_name']
                if ak not in ba:
                    ba[ak] = {'aspect_id': r['aspect_id'], 'pos': 0, 'neg': 0, 'total': 0}
                ba[ak]['pos'] += r['positive_count']
                ba[ak]['neg'] += r['negative_count']
                ba[ak]['total'] += r['total_mentions']
        set_cache("aspects_by_pid", asp_by_pid)

        # Compute brand-level aggregated aspects
        asp_by_brand = {}
        for brand, aspects in asp_brand_raw.items():
            total_all = sum(v['total'] for v in aspects.values()) or 1
            rows = []
            for asp_name, v in aspects.items():
                sat = round(v['pos'] * 100 / (v['pos'] + v['neg'])) if (v['pos'] + v['neg']) > 0 else 0
                sov = round(v['total'] * 100 / total_all)
                rows.append({
                    'aspect_id': v['aspect_id'],
                    'aspect_name': asp_name,
                    'positive_count': v['pos'],
                    'negative_count': v['neg'],
                    'total_mentions': v['total'],
                    'satisfaction_pct': sat,
                    'share_of_voice_pct': sov,
                    'satisfaction': sat,
                    'share_of_voice': sov,
                })
            asp_by_brand[brand] = sorted(rows, key=lambda x: -x['total_mentions'])
        set_cache("aspects_by_brand", asp_by_brand)
        print(f"[CACHE] Loaded aspects: {len(asp_by_pid)} products, {len(asp_by_brand)} brands")

        # ── Load product_emotions ──
        emo_df = client.query(f"""
            SELECT product_id, emotion, mention_count, pct_of_total as percentage
            FROM `{PROJECT}.{DATASET}.product_emotions`
        """).to_dataframe()

        emo_by_pid = {}
        for _, row in emo_df.iterrows():
            pid = str(int(row['product_id']))
            emo_by_pid.setdefault(pid, []).append({
                'emotion': str(row['emotion'] or ''),
                'count': int(row['mention_count'] or 0),
                'percentage': int(row['percentage'] or 0),
            })
        set_cache("emotions_by_pid", emo_by_pid)
        print(f"[CACHE] Loaded emotions: {len(emo_by_pid)} products")

        # ── Load product_pain_delights ──
        pd_df = client.query(f"""
            SELECT product_id, phrase, aspect_name, signal_type, mention_count, severity_rank
            FROM `{PROJECT}.{DATASET}.product_pain_delights`
            WHERE phrase IS NOT NULL AND TRIM(phrase) != ''
            ORDER BY product_id, signal_type, severity_rank
        """).to_dataframe()

        pain_by_pid = {}
        delight_by_pid = {}
        for _, row in pd_df.iterrows():
            pid = str(int(row['product_id']))
            phrase = str(row['phrase'] or '').strip()
            if not phrase or phrase.lower() in ('null','none','nan'): continue
            r = {
                'phrase': phrase,
                'aspect_name': str(row['aspect_name'] or ''),
                'mention_count': int(row['mention_count'] or 0),
            }
            sig = str(row['signal_type'] or '')
            if sig == 'pain_point':
                pain_by_pid.setdefault(pid, []).append(r)
            elif sig == 'delight':
                delight_by_pid.setdefault(pid, []).append(r)
        set_cache("pain_by_pid", pain_by_pid)
        set_cache("delight_by_pid", delight_by_pid)
        print(f"[CACHE] Loaded pain: {len(pain_by_pid)} products, delights: {len(delight_by_pid)} products")

        # ── Load product_demographics ──
        demo_df = client.query(f"""
            SELECT product_id, dimension, dimension_value, review_count, pct_of_total
            FROM `{PROJECT}.{DATASET}.product_demographics`
            WHERE dimension_value IS NOT NULL
        """).to_dataframe()

        demo_by_pid = {}
        for _, row in demo_df.iterrows():
            pid = str(int(row['product_id']))
            dim = str(row['dimension'] or '')
            val = str(row['dimension_value'] or '').strip()
            if not val or val.lower() in ('unknown','none','null',''): continue
            entry = demo_by_pid.setdefault(pid, {'gender':[], 'traveler_type':[], 'stay_purpose':[]})
            if dim in entry:
                entry[dim].append({
                    "dimension_value": val,           # consistent with BQ response
                    "count": int(row['review_count'] or 0),
                    "pct_of_total": int(row['pct_of_total'] or 0),
                })
        set_cache("demo_by_pid", demo_by_pid)
        print(f"[CACHE] Loaded demographics: {len(demo_by_pid)} products")

        # ── Load product_phrases treemap cache ──
        VALID_ASPECTS = {1:"Dining",2:"Cleanliness",3:"Amenities",4:"Staff",5:"Room",6:"Location",7:"Value for Money"}
        phrases_df = client.query(f"""
            SELECT p.product_id, p.aspect_id, p.treemap_name, SUM(p.mention_count) as mention_count,
                   h.brand_name
            FROM `{PROJECT}.{DATASET}.product_phrases` p
            JOIN `{PROJECT}.{DATASET}.hotel_master` h ON p.product_id = h.product_id
            WHERE p.treemap_name IS NOT NULL AND TRIM(p.treemap_name) != ''
              AND p.aspect_id IN ({','.join(str(k) for k in VALID_ASPECTS)})
            GROUP BY p.product_id, p.aspect_id, p.treemap_name, h.brand_name
            ORDER BY p.product_id, p.aspect_id, mention_count DESC
        """).to_dataframe()

        treemap_by_pid = {}   # str(product_id) -> {aspect_name: [{treemap_name, mention_count}]}
        treemap_by_brand = {} # brand_name -> {aspect_name: [{treemap_name, mention_count}]}
        brand_asp_accum = {}  # brand -> aspect_id -> {treemap_name: total_count}

        for _, row in phrases_df.iterrows():
            pid = str(int(row['product_id']))
            asp_id = int(row['aspect_id'])
            asp_name = VALID_ASPECTS.get(asp_id)
            if not asp_name: continue
            tn = str(row['treemap_name'])
            mc = int(row['mention_count'])
            brand = str(row['brand_name'] or '')

            # Per product
            pid_asp = treemap_by_pid.setdefault(pid, {}).setdefault(asp_name, [])
            if len(pid_asp) < 5:
                pid_asp.append({"treemap_name": tn, "mention_count": mc})

            # Accumulate per brand
            brand_asp_accum.setdefault(brand, {}).setdefault(asp_id, {})[tn] = \
                brand_asp_accum.get(brand, {}).get(asp_id, {}).get(tn, 0) + mc

        # Build top-5 per brand per aspect
        for brand, asp_map in brand_asp_accum.items():
            treemap_by_brand[brand] = {}
            for asp_id, tn_map in asp_map.items():
                asp_name = VALID_ASPECTS.get(asp_id)
                if not asp_name: continue
                top5 = sorted(tn_map.items(), key=lambda x: -x[1])[:5]
                treemap_by_brand[brand][asp_name] = [{"treemap_name": t, "mention_count": c} for t,c in top5]

        set_cache("treemap_by_pid", treemap_by_pid)
        set_cache("treemap_by_brand", treemap_by_brand)
        print(f"[CACHE] Loaded treemap phrases: {len(treemap_by_pid)} products, {len(treemap_by_brand)} brands")

    except Exception as e:
        print(f"[CACHE ERROR] {e}")
        traceback.print_exc()

# Background cache refresh
def refresh_caches():
    while True:
        time.sleep(CACHE_TTL)
        print("[CACHE] Refreshing caches...")
        load_master_caches()

threading.Thread(target=refresh_caches, daemon=True).start()

# ─────────────────────────────────────────
# STATIC FILES
# ─────────────────────────────────────────
@app.get("/")
async def root():
    try:
        with open("index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error loading page: {e}</h1>")

@app.get("/acquink_logo.png")
async def get_logo():
    return FileResponse("acquink_logo.png", media_type="image/png")

@app.get("/health")
async def health():
    return {
        "status": "healthy" if get_bq() else "degraded",
        "database": "connected" if get_bq() else "disconnected",
        "gemini": "connected" if get_gemini() else "disconnected",
        "cache": {k: is_cache_valid(k) for k in CACHE.keys()}
    }

# ─────────────────────────────────────────
# MASTER DATA APIs
# ─────────────────────────────────────────
@app.get("/api/brands")
async def get_brands(category: str = "hotels"):
    """Get all brands for a category"""
    cached = get_cache("brands")
    if cached:
        # Filter by category
        filtered = [b for b in cached if category in (b.get('categories') or '')]
        return {"brands": filtered}
    
    client = get_bq()
    if not client:
        raise HTTPException(status_code=500, detail="Database unavailable")
    
    query = f"""
    SELECT brand_id, brand_name, categories
    FROM `{PROJECT}.{DATASET}.brand_master`
    WHERE categories LIKE '%{category}%'
    ORDER BY brand_name
    """
    df = client.query(query).to_dataframe()
    return {"brands": df.to_dict(orient='records')}

@app.get("/api/hotels")
async def get_hotels(
    brand: Optional[str] = None,
    city: Optional[str] = None,
    star_category: Optional[str] = None,
    category: str = "hotels"
):
    """Get hotels with optional filters"""
    cached = get_cache("hotels")
    if cached:
        filtered = cached
        if brand:
            filtered = [h for h in filtered if h.get('brand_name','').lower() == brand.lower()]
        if city and city != "All Cities":
            filtered = [h for h in filtered if h.get('city') == city]
        if star_category and star_category != "All Stars":
            try:
                star_val = int(float(star_category))
                filtered = [h for h in filtered if int(h.get('star_category') or 0) == star_val]
            except: pass
        return {"hotels": filtered}
    
    client = get_bq()
    if not client:
        raise HTTPException(status_code=500, detail="Database unavailable")
    
    conditions = ["1=1"]
    if brand:
        conditions.append(f"brand_name = '{brand}'")
    if city and city != "All Cities":
        conditions.append(f"city = '{city}'")
    if star_category and star_category != "All Stars":
        try:
            star_val = int(float(star_category))
            conditions.append(f"star_category = {star_val}")
        except: pass
    
    query = f"""
    SELECT product_id, hotel_name, brand_id, brand_name, city, state, country,
           star_category, review_count, overall_satisfaction
    FROM `{PROJECT}.{DATASET}.hotel_master`
    WHERE {' AND '.join(conditions)}
    ORDER BY hotel_name
    """
    df = client.query(query).to_dataframe()
    return {"hotels": df.to_dict(orient='records')}

@app.get("/api/cities")
async def get_cities(brand: Optional[str] = None):
    """Get unique cities, optionally filtered by brand name"""
    cached = get_cache("hotels")
    if cached:
        filtered = cached
        if brand:
            filtered = [h for h in filtered if h.get('brand_name','').lower() == brand.lower()]
        cities = sorted(set(h.get('city') for h in filtered if h.get('city')))
        return {"cities": cities}

    # Hotels cache not ready — query BQ directly (never fall back to all-cities when brand is set)
    client = get_bq()
    if not client:
        return {"cities": []}

    conditions = ["city IS NOT NULL"]
    if brand:
        safe_brand = brand.replace("'", "''")
        conditions.append(f"LOWER(brand_name) = LOWER('{safe_brand}')")

    query = f"""
    SELECT DISTINCT city
    FROM `{PROJECT}.{DATASET}.hotel_master`
    WHERE {' AND '.join(conditions)}
    ORDER BY city
    """
    try:
        df = client.query(query).to_dataframe()
        return {"cities": df['city'].tolist()}
    except Exception as e:
        print(f"[get_cities BQ error] {e}")
        return {"cities": []}

@app.get("/api/search")
async def search_hotels(q: str = Query(..., min_length=2)):
    """Wildcard search for hotels"""
    cached = get_cache("hotels")
    if cached:
        q_lower = q.lower()
        results = [
            h for h in cached
            if q_lower in (h.get('hotel_name') or '').lower()
            or q_lower in (h.get('city') or '').lower()
            or q_lower in (h.get('brand_name') or '').lower()
        ][:20]
        return {"results": results}
    
    client = get_bq()
    if not client:
        return {"results": []}
    
    query = f"""
    SELECT product_id, hotel_name, brand_name, city, star_category
    FROM `{PROJECT}.{DATASET}.hotel_master`
    WHERE LOWER(hotel_name) LIKE '%{q.lower()}%'
       OR LOWER(city) LIKE '%{q.lower()}%'
       OR LOWER(brand_name) LIKE '%{q.lower()}%'
    LIMIT 20
    """
    df = client.query(query).to_dataframe()
    return {"results": df.to_dict(orient='records')}

# ─────────────────────────────────────────
# BRAND-LEVEL APIs
# ─────────────────────────────────────────
@app.get("/api/brand/{brand_id}/summary")
async def get_brand_summary(brand_id: str, city: Optional[str] = None, star: Optional[str] = None):
    """Get aggregated brand-level summary, optionally filtered by city and star category"""
    client = get_bq()
    if not client:
        raise HTTPException(status_code=500, detail="Database unavailable")

    safe_brand = brand_id.replace("'", "''")
    safe_city = city.replace("'", "''") if city else None
    safe_star = star.replace("'", "''") if star else None

    # Build product_id subquery with optional city/star filter
    pid_filter = f"brand_name = '{safe_brand}'"
    if safe_city:
        pid_filter += f" AND city = '{safe_city}'"
    if safe_star:
        pid_filter += f" AND star_category = '{safe_star}'"
    pid_subquery = f"SELECT product_id FROM `{PROJECT}.{DATASET}.hotel_master` WHERE {pid_filter}"

    # Brand info
    brand_query = f"""
    SELECT brand_id, brand_name
    FROM `{PROJECT}.{DATASET}.brand_master`
    WHERE brand_name = '{safe_brand}' OR CAST(brand_id AS STRING) = '{safe_brand}'
    LIMIT 1
    """
    brand_df = client.query(brand_query).to_dataframe()
    if brand_df.empty:
        brand_info = {"brand_id": brand_id, "brand_name": brand_id}
    else:
        brand_info = clean_row(brand_df.iloc[0].to_dict())

    # Hotel count and review count — filtered
    hotels_query = f"""
    SELECT COUNT(*) as hotel_count, SUM(review_count) as total_reviews,
           ROUND(AVG(overall_satisfaction)) as overall_satisfaction,
           ROUND(AVG(google_rating),1) as avg_rating
    FROM `{PROJECT}.{DATASET}.hotel_master`
    WHERE {pid_filter}
    """
    hotels_df = client.query(hotels_query).to_dataframe()
    stats = hotels_df.iloc[0].to_dict() if not hotels_df.empty else {'hotel_count':0,'total_reviews':0,'overall_satisfaction':0,'avg_rating':0}

    # Aspects aggregated — filtered by product_ids
    aspects_query = f"""
    SELECT aspect_id, aspect_name,
           SUM(positive_count) as positive_count,
           SUM(negative_count) as negative_count,
           SUM(total_mentions) as total_mentions
    FROM `{PROJECT}.{DATASET}.product_aspect_summary`
    WHERE product_id IN ({pid_subquery})
    GROUP BY aspect_id, aspect_name
    ORDER BY total_mentions DESC
    """
    aspects_df = client.query(aspects_query).to_dataframe()

    aspects = []
    total_mentions = aspects_df['total_mentions'].sum()
    for _, row in aspects_df.iterrows():
        pos = row['positive_count'] or 0
        neg = row['negative_count'] or 0
        total = row['total_mentions'] or 0
        sat = round(pos * 100 / (pos + neg)) if (pos + neg) > 0 else 0
        sov = round(total * 100 / total_mentions) if total_mentions > 0 else 0
        aspects.append({
            "aspect_id": int(row['aspect_id']),
            "aspect_name": row['aspect_name'],
            "satisfaction_pct": sat,
            "share_of_voice_pct": sov,
            "total_mentions": int(total)
        })

    # Emotions aggregated — filtered
    emotions_query = f"""
    SELECT emotion, SUM(mention_count) as count
    FROM `{PROJECT}.{DATASET}.product_emotions`
    WHERE product_id IN ({pid_subquery})
    GROUP BY emotion ORDER BY count DESC
    """
    emotions_df = client.query(emotions_query).to_dataframe()
    total_emotion = emotions_df['count'].sum()
    emotions = [
        {"emotion": row['emotion'], "count": int(row['count']),
         "percentage": round(row['count'] * 100 / total_emotion) if total_emotion > 0 else 0}
        for _, row in emotions_df.iterrows()
    ]

    # Demographics aggregated — filtered
    demo_query = f"""
    SELECT dimension, dimension_value, SUM(review_count) as count
    FROM `{PROJECT}.{DATASET}.product_demographics`
    WHERE product_id IN ({pid_subquery})
    GROUP BY dimension, dimension_value ORDER BY dimension, count DESC
    """
    demo_df = client.query(demo_query).to_dataframe()
    demographics = {"traveler_type": [], "gender": [], "stay_purpose": []}
    for dim in ["traveler_type", "gender", "stay_purpose"]:
        dim_data = demo_df[demo_df['dimension'] == dim]
        total_dim = dim_data['count'].sum()
        for _, row in dim_data.iterrows():
            if row['dimension_value'] and str(row['dimension_value']).strip():
                demographics[dim].append({
                    "dimension_value": row['dimension_value'],
                    "count": int(row['count']),
                    "pct_of_total": round(row['count'] * 100 / total_dim) if total_dim > 0 else 0
                })

    # Pain points & delights — filtered
    pain_brand_query = f"""
    SELECT p.phrase, p.aspect_name, SUM(p.mention_count) as mention_count, p.signal_type
    FROM `{PROJECT}.{DATASET}.product_pain_delights` p
    WHERE p.product_id IN ({pid_subquery})
      AND p.phrase IS NOT NULL AND TRIM(p.phrase) != ''
    GROUP BY p.phrase, p.aspect_name, p.signal_type
    ORDER BY mention_count DESC
    """
    try:
        pain_brand_df = client.query(pain_brand_query).to_dataframe()
        brand_pain = [{"phrase":r["phrase"],"aspect_name":r["aspect_name"],"mention_count":int(r["mention_count"])} for _,r in pain_brand_df[pain_brand_df["signal_type"]=="pain_point"].head(30).iterrows()]
        brand_delights = [{"phrase":r["phrase"],"aspect_name":r["aspect_name"],"mention_count":int(r["mention_count"])} for _,r in pain_brand_df[pain_brand_df["signal_type"]=="delight"].head(30).iterrows()]
    except:
        brand_pain, brand_delights = [], []

    # RD signals — filtered
    rd_brand_query = f"""
    SELECT r.signal_type as rd_signal, r.phrase, SUM(r.mention_count) as mention_count
    FROM `{PROJECT}.{DATASET}.product_rd_signals` r
    WHERE r.product_id IN ({pid_subquery})
    GROUP BY r.signal_type, r.phrase ORDER BY mention_count DESC
    """
    try:
        rd_brand_df = client.query(rd_brand_query).to_dataframe()
        brand_rd = {"feature_request":[],"price_feedback":[],"expectation_gap":[]}
        for _,row in rd_brand_df.iterrows():
            sig = str(row["rd_signal"] or "")
            if sig in brand_rd:
                brand_rd[sig].append({"phrase":str(row["phrase"]),"mention_count":int(row["mention_count"]),"aspect_name":str(row.get("aspect_name") or "")})
    except:
        brand_rd = {"feature_request":[],"price_feedback":[],"expectation_gap":[]}

    return {
        **brand_info,
        "hotel_count": int(stats['hotel_count'] or 0),
        "review_count": int(stats['total_reviews'] or 0),
        "overall_satisfaction": int(stats['overall_satisfaction'] or 0),
        "avg_rating": float(stats.get('avg_rating') or 0),
        "aspects": aspects,
        "emotions": emotions,
        "demographics": demographics,
        "pain_points": brand_pain,
        "delights": brand_delights,
        "rd_signals": brand_rd
    }

# ─────────────────────────────────────────
# HOTEL-LEVEL APIs
# ─────────────────────────────────────────
@app.get("/api/hotel/{product_id}/summary")
async def get_hotel_summary(product_id: int):
    """Get full hotel summary with all data - parallel BQ queries"""
    client = get_bq()
    if not client:
        raise HTTPException(status_code=500, detail="Database unavailable")

    pid_str = str(product_id)
    loop = asyncio.get_event_loop()

    def run_query(sql):
        try:
            return client.query(sql).to_dataframe()
        except Exception as e:
            print(f"[BQ ERROR] {e}")
            import pandas as pd
            return pd.DataFrame()

    # ── Serve from cache where possible ──────────────────────────────────
    asp_cache   = get_cache("aspects_by_pid")
    emo_cache   = get_cache("emotions_by_pid")
    pain_cache  = get_cache("pain_by_pid")
    del_cache   = get_cache("delight_by_pid")
    demo_cache  = get_cache("demo_by_pid")

    aspects_cached   = asp_cache.get(pid_str)  if asp_cache  else None
    emotions_cached  = emo_cache.get(pid_str)  if emo_cache  else None
    pain_cached      = pain_cache.get(pid_str) if pain_cache else None
    delights_cached  = del_cache.get(pid_str)  if del_cache  else None
    demo_cached      = demo_cache.get(pid_str) if demo_cache else None

    # ── Always need hotel_master row + rd_signals from BQ ────────────────
    bq_queries = {
        "hotel": f"SELECT * FROM `{PROJECT}.{DATASET}.hotel_master` WHERE product_id = {product_id}",
        "rd":    f"""SELECT signal_type as rd_signal, phrase, treemap_name, aspect_name, mention_count
                     FROM `{PROJECT}.{DATASET}.product_rd_signals`
                     WHERE product_id = {product_id} ORDER BY rd_signal, mention_count DESC"""
    }

    # Add BQ fallbacks for anything not yet cached
    if aspects_cached  is None: bq_queries["aspects"]  = f"""SELECT aspect_id, aspect_name, positive_count, negative_count, total_mentions, satisfaction_pct, share_of_voice_pct FROM `{PROJECT}.{DATASET}.product_aspect_summary` WHERE product_id = {product_id} ORDER BY total_mentions DESC"""
    if emotions_cached is None: bq_queries["emotions"] = f"""SELECT emotion, mention_count as count, pct_of_total as percentage FROM `{PROJECT}.{DATASET}.product_emotions` WHERE product_id = {product_id} ORDER BY mention_count DESC"""
    if pain_cached     is None: bq_queries["pain"]     = f"""SELECT phrase, aspect_name, mention_count FROM `{PROJECT}.{DATASET}.product_pain_delights` WHERE product_id = {product_id} AND signal_type = 'pain_point' AND phrase IS NOT NULL AND TRIM(phrase) != '' ORDER BY mention_count DESC LIMIT 20"""
    if delights_cached is None: bq_queries["delight"]  = f"""SELECT phrase, aspect_name, mention_count FROM `{PROJECT}.{DATASET}.product_pain_delights` WHERE product_id = {product_id} AND signal_type = 'delight' AND phrase IS NOT NULL AND TRIM(phrase) != '' ORDER BY mention_count DESC LIMIT 20"""
    if demo_cached     is None: bq_queries["demo"]     = f"""SELECT dimension, dimension_value, review_count as count, pct_of_total FROM `{PROJECT}.{DATASET}.product_demographics` WHERE product_id = {product_id} ORDER BY dimension, review_count DESC"""

    results = await asyncio.gather(*[loop.run_in_executor(None, run_query, sql) for sql in bq_queries.values()])
    dfs = dict(zip(bq_queries.keys(), results))

    if dfs["hotel"].empty:
        raise HTTPException(status_code=404, detail="Hotel not found")

    hotel_info = clean_row(dfs["hotel"].iloc[0].to_dict())

    # ── Aspects ───────────────────────────────────────────────────────────
    if aspects_cached is not None:
        aspects = aspects_cached
        print(f"[CACHE HIT] aspects for {pid_str}")
    else:
        aspects = [clean_row(r) for r in dfs.get("aspects", __import__('pandas').DataFrame()).to_dict(orient='records')]

    # ── Emotions ──────────────────────────────────────────────────────────
    if emotions_cached is not None:
        emotions = [{"emotion": e["emotion"], "count": e.get("count",0), "percentage": e.get("percentage",0)} for e in emotions_cached]
        print(f"[CACHE HIT] emotions for {pid_str}")
    else:
        emotions = [clean_row(r) for r in dfs.get("emotions", __import__('pandas').DataFrame()).to_dict(orient='records')]

    # ── Pain points ───────────────────────────────────────────────────────
    if pain_cached is not None:
        pain_points = sorted(pain_cached, key=lambda x: -(x.get("mention_count") or 0))
        print(f"[CACHE HIT] pain for {pid_str}: {len(pain_points)} items")
    else:
        raw = dfs.get("pain", __import__('pandas').DataFrame()).to_dict(orient='records')
        pain_points = [clean_row(r) for r in raw if r.get("phrase") and str(r.get("phrase")).lower() not in ("nan","none","null","")]

    # ── Delights ──────────────────────────────────────────────────────────
    if delights_cached is not None:
        delights = sorted(delights_cached, key=lambda x: -(x.get("mention_count") or 0))
        print(f"[CACHE HIT] delights for {pid_str}: {len(delights)} items")
    else:
        raw = dfs.get("delight", __import__('pandas').DataFrame()).to_dict(orient='records')
        delights = [clean_row(r) for r in raw if r.get("phrase") and str(r.get("phrase")).lower() not in ("nan","none","null","")]

    # ── Demographics ──────────────────────────────────────────────────────
    demographics = {"traveler_type": [], "gender": [], "stay_purpose": []}
    if demo_cached is not None:
        demographics = demo_cached
        print(f"[CACHE HIT] demo for {pid_str}")
    else:
        for _, row in dfs.get("demo", __import__('pandas').DataFrame()).iterrows():
            dim = str(row.get('dimension', ''))
            val = row.get('dimension_value')
            if dim in demographics and val and str(val).lower() not in ('unknown','none','null',''):
                demographics[dim].append({
                    "dimension_value": str(val),
                    "count": int(row['count'] or 0),
                    "pct_of_total": int(row['pct_of_total'] or 0)
                })

    # ── RD Signals (always from BQ — not cached) ──────────────────────────
    rd_signals = {"feature_request": [], "price_feedback": [], "expectation_gap": []}
    for _, row in dfs.get("rd", __import__('pandas').DataFrame()).iterrows():
        sig = str(row.get('rd_signal', ''))
        if sig in rd_signals:
            rd_signals[sig].append({
                "phrase": str(row['phrase'] or ''),
                "treemap_name": str(row['treemap_name'] or ''),
                "mention_count": int(row['mention_count'] or 0)
            })

    return {
        **hotel_info,
        "aspects": aspects,
        "emotions": emotions,
        "demographics": demographics,
        "pain_points": pain_points,
        "delights": delights,
        "rd_signals": rd_signals
    }

@app.get("/api/hotel/{product_id}/aspects")
async def get_hotel_aspects(product_id: int):
    """Get aspect breakdown with top phrases"""
    client = get_bq()
    if not client:
        raise HTTPException(status_code=500, detail="Database unavailable")
    
    # Aspects
    aspects_query = f"""
    SELECT aspect_id, aspect_name, positive_count, negative_count,
           total_mentions, satisfaction_pct, share_of_voice_pct
    FROM `{PROJECT}.{DATASET}.product_aspect_summary`
    WHERE product_id = {product_id}
    ORDER BY total_mentions DESC
    """
    aspects_df = client.query(aspects_query).to_dataframe()
    
    # Top phrases per aspect
    phrases_query = f"""
    SELECT aspect_id, phrase, mention_count, sentiment_type
    FROM `{PROJECT}.{DATASET}.product_phrases`
    WHERE product_id = {product_id}
    ORDER BY aspect_id, mention_count DESC
    """
    phrases_df = client.query(phrases_query).to_dataframe()
    
    aspects = []
    for _, row in aspects_df.iterrows():
        aspect_id = row['aspect_id']
        aspect_phrases = phrases_df[phrases_df['aspect_id'] == aspect_id].head(5)
        aspects.append({
            **row.to_dict(),
            "top_phrases": aspect_phrases.to_dict(orient='records')
        })
    
    return {"aspects": aspects}

@app.get("/api/hotel/{product_id}/pain_delights")
async def get_hotel_pain_delights(product_id: int):
    """Get pain points and delights"""
    client = get_bq()
    if not client:
        raise HTTPException(status_code=500, detail="Database unavailable")
    
    query = f"""
    SELECT phrase, treemap_name, aspect_id, aspect_name, signal_type,
           mention_count, severity_rank
    FROM `{PROJECT}.{DATASET}.product_pain_delights`
    WHERE product_id = {product_id}
    AND phrase IS NOT NULL AND TRIM(phrase) != ''
    ORDER BY signal_type, severity_rank
    """
    df = client.query(query).to_dataframe()
    
    pain_points = df[df['signal_type'] == 'pain_point'].to_dict(orient='records')
    delights = df[df['signal_type'] == 'delight'].to_dict(orient='records')
    
    return {"pain_points": pain_points, "delights": delights}

@app.get("/api/hotel/{product_id}/emotions")
async def get_hotel_emotions(product_id: int):
    """Get emotion distribution"""
    client = get_bq()
    if not client:
        raise HTTPException(status_code=500, detail="Database unavailable")
    
    query = f"""
    SELECT emotion, mention_count as count, pct_of_total as percentage
    FROM `{PROJECT}.{DATASET}.product_emotions`
    WHERE product_id = {product_id}
    ORDER BY mention_count DESC
    """
    df = client.query(query).to_dataframe()
    return {"emotions": df.to_dict(orient='records')}

@app.get("/api/hotel/{product_id}/rd_signals")
async def get_hotel_rd_signals(product_id: int):
    """Get R&D signals"""
    client = get_bq()
    if not client:
        raise HTTPException(status_code=500, detail="Database unavailable")
    
    query = f"""
    SELECT signal_type as rd_signal, phrase, treemap_name, mention_count
    FROM `{PROJECT}.{DATASET}.product_rd_signals`
    WHERE product_id = {product_id}
    ORDER BY rd_signal, mention_count DESC
    """
    df = client.query(query).to_dataframe()
    
    signals = {"feature_request": [], "price_feedback": [], "expectation_gap": []}
    for _, row in df.iterrows():
        signal_type = row['rd_signal']
        if signal_type in signals:
            signals[signal_type].append({
                "phrase": row['phrase'],
                "treemap_name": row['treemap_name'],
                "mention_count": int(row['mention_count'])
            })
    
    return {"rd_signals": signals}

# ─────────────────────────────────────────
# DRILL-DOWN API
# ─────────────────────────────────────────
class DrilldownRequest(BaseModel):
    product_id: int
    phrase: str
    signal_type: str  # "pain_point", "delight", "rd_signal"
    limit: int = 10

@app.post("/api/drilldown")
async def drilldown(request: DrilldownRequest):
    """Get actual reviews for a specific phrase/signal"""
    client = get_bq()
    if not client:
        raise HTTPException(status_code=500, detail="Database unavailable")

    safe_phrase = request.phrase.replace("'", "''")

    if request.signal_type == "pain_point":
        sentiment_filter = "AND sentiment_type = 'negative'"
    elif request.signal_type == "delight":
        sentiment_filter = "AND sentiment_type = 'positive'"
    else:
        sentiment_filter = ""

    try:
        # Try 1: phrase column match (fast, exact)
        query = f"""
        SELECT review_text, sentiment_text, star_rating, reviewer_name,
               review_date, traveler_type, stay_purpose, emotion
        FROM `{PROJECT}.{DATASET}.review_drilldown`
        WHERE CAST(product_id AS STRING) = '{request.product_id}'
          AND LOWER(phrase) = LOWER('{safe_phrase}')
          {sentiment_filter}
        LIMIT {request.limit}
        """
        df = client.query(query).to_dataframe()

        # Try 2: review_text LIKE fallback (phrase column may not match)
        if df.empty:
            query2 = f"""
            SELECT review_text, sentiment_text, star_rating, reviewer_name,
                   review_date, traveler_type, stay_purpose, emotion
            FROM `{PROJECT}.{DATASET}.review_drilldown`
            WHERE CAST(product_id AS STRING) = '{request.product_id}'
              AND LOWER(review_text) LIKE LOWER('%{safe_phrase}%')
              {sentiment_filter}
            LIMIT {request.limit}
            """
            df = client.query(query2).to_dataframe()
            print(f"[DRILLDOWN] LIKE fallback: {len(df)} rows for pid={request.product_id} phrase='{request.phrase}'")

        reviews = df.to_dict(orient='records')
        return {"reviews": reviews, "total": len(reviews), "total_count": len(reviews),
                "phrase": request.phrase, "signal_type": request.signal_type}
    except Exception as e:
        print(f"[DRILLDOWN ERROR] {e}")
        return {"reviews": [], "total_count": 0, "error": str(e)}


@app.post("/api/brand_drilldown")
async def brand_drilldown(request: Request):
    """Get reviews mentioning a phrase across all hotels of a brand.
    Uses review_text LIKE search — avoids phrase column mismatch issues."""
    client = get_bq()
    if not client:
        raise HTTPException(status_code=500, detail="Database unavailable")
    body = await request.json()
    brand = body.get("brand", "")
    phrase = body.get("phrase", "")
    signal_type = body.get("signal_type", "pain_point")
    limit = body.get("limit", 20)
    safe_brand = brand.replace("'", "''")
    safe_phrase = phrase.replace("'", "''")
    print(f"[BRAND_DRILLDOWN] brand='{brand}', phrase='{phrase}'")

    try:
        # Match by phrase column (same as hotel drilldown) — no sentiment_type filter
        # sentiment_type is unreliable across rows; phrase column is the ground truth
        query = f"""
        SELECT r.review_text, r.sentiment_text, r.star_rating, r.reviewer_name,
               r.review_date, r.traveler_type, h.hotel_name
        FROM `{PROJECT}.{DATASET}.review_drilldown` r
        JOIN `{PROJECT}.{DATASET}.hotel_master` h
          ON CAST(r.product_id AS STRING) = CAST(h.product_id AS STRING)
        WHERE LOWER(h.brand_name) = LOWER('{safe_brand}')
          AND LOWER(r.phrase) = LOWER('{safe_phrase}')
        ORDER BY r.star_rating ASC
        LIMIT {limit}
        """
        df = client.query(query).to_dataframe()
        print(f"[BRAND_DRILLDOWN] phrase match rows={len(df)}")
        if df.empty:
            # Fallback: review_text LIKE search
            query = f"""
            SELECT r.review_text, r.sentiment_text, r.star_rating, r.reviewer_name,
                   r.review_date, r.traveler_type, h.hotel_name
            FROM `{PROJECT}.{DATASET}.review_drilldown` r
            JOIN `{PROJECT}.{DATASET}.hotel_master` h
              ON CAST(r.product_id AS STRING) = CAST(h.product_id AS STRING)
            WHERE LOWER(h.brand_name) = LOWER('{safe_brand}')
              AND LOWER(r.review_text) LIKE LOWER('%{safe_phrase}%')
            ORDER BY r.star_rating ASC
            LIMIT {limit}
            """
            df = client.query(query).to_dataframe()
            print(f"[BRAND_DRILLDOWN] text fallback rows={len(df)}")
        return {"reviews": df.to_dict(orient='records'), "total": len(df), "phrase": phrase}

    except Exception as e:
        print(f"[BRAND_DRILLDOWN ERROR] {e}")
        return {"reviews": [], "total": 0, "error": str(e)}




@app.get("/api/debug_drilldown")
async def debug_drilldown(brand: str = "", phrase: str = ""):
    """Debug endpoint — check what's in review_drilldown for a brand+phrase"""
    client = get_bq()
    if not client: return {"error": "no db"}
    safe_brand = brand.replace("'","''")
    safe_phrase = phrase.replace("'","''")
    try:
        # Get product_ids for brand
        pids_df = client.query(f"SELECT CAST(product_id AS STRING) as product_id FROM `{PROJECT}.{DATASET}.hotel_master` WHERE LOWER(brand_name)=LOWER('{safe_brand}')").to_dataframe()
        pids = pids_df['product_id'].tolist()
        if not pids: return {"error": f"No products for brand '{brand}'", "brand": brand}
        pid_list = ','.join(f"'{p}'" for p in pids[:5])  # sample first 5
        all_pid_list = ','.join(f"'{p}'" for p in pids)
        # Check phrases in review_drilldown for these products
        sample = client.query(f"SELECT DISTINCT phrase FROM `{PROJECT}.{DATASET}.review_drilldown` WHERE product_id IN ({pid_list}) AND phrase IS NOT NULL LIMIT 20").to_dataframe()
        match = client.query(f"SELECT COUNT(*) as cnt FROM `{PROJECT}.{DATASET}.review_drilldown` WHERE product_id IN ({all_pid_list}) AND LOWER(phrase)=LOWER('{safe_phrase}')").to_dataframe()
        return {
            "brand": brand, "phrase": phrase,
            "product_count": len(pids),
            "sample_phrases_in_drilldown": sample['phrase'].tolist(),
            "exact_phrase_match_count": int(match.iloc[0]['cnt'])
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/hotel/{product_id}/paradox")
async def get_paradox_reviews(product_id: int, limit: int = 50):
    """5-star reviews with negative sentiment — the paradox reviews"""
    client = get_bq()
    if not client:
        raise HTTPException(status_code=500, detail="Database unavailable")
    try:
        query = f"""
        SELECT DISTINCT review_id, review_text, sentiment_text, star_rating,
               reviewer_name, review_date, traveler_type, stay_purpose,
               emotion, phrase, treemap_name
        FROM `{PROJECT}.{DATASET}.review_drilldown`
        WHERE CAST(product_id AS STRING) = CAST({product_id} AS STRING)
          AND star_rating = 5
          AND sentiment_type = 'negative'
          AND pain_point = 1
        ORDER BY review_date DESC
        LIMIT {limit}
        """
        df = client.query(query).to_dataframe()
        count_query = f"""
        SELECT COUNT(DISTINCT review_id) as total
        FROM `{PROJECT}.{DATASET}.review_drilldown`
        WHERE CAST(product_id AS STRING) = CAST({product_id} AS STRING)
          AND star_rating = 5
          AND sentiment_type = 'negative'
          AND pain_point = 1
        """
        count_df = client.query(count_query).to_dataframe()
        total = int(count_df.iloc[0]['total']) if not count_df.empty else 0
        reviews = [clean_row(r) for r in df.to_dict(orient='records')]
        return {"reviews": reviews, "total_count": total}
    except Exception as e:
        print(f"[PARADOX ERROR] {e}")
        return {"reviews": [], "total_count": 0, "error": str(e)}

# ─────────────────────────────────────────
# COMPARE APIs
# ─────────────────────────────────────────
class CompareHotelsRequest(BaseModel):
    product_ids: List[int]

@app.post("/api/compare/hotels")
async def compare_hotels(request: CompareHotelsRequest):
    """Compare 2-3 hotels side by side"""
    client = get_bq()
    if not client:
        raise HTTPException(status_code=500, detail="Database unavailable")
    
    if len(request.product_ids) < 2 or len(request.product_ids) > 3:
        raise HTTPException(status_code=400, detail="Provide 2-3 hotel IDs")
    
    ids_str = ",".join(str(id) for id in request.product_ids)
    
    # Get hotel info
    hotels_query = f"""
    SELECT product_id, hotel_name, brand_name, city, star_category,
           review_count, overall_satisfaction
    FROM `{PROJECT}.{DATASET}.hotel_master`
    WHERE product_id IN ({ids_str})
    """
    hotels_df = client.query(hotels_query).to_dataframe()
    
    # Get aspects for each
    aspects_query = f"""
    SELECT product_id, aspect_id, aspect_name, satisfaction_pct, share_of_voice_pct
    FROM `{PROJECT}.{DATASET}.product_aspect_summary`
    WHERE product_id IN ({ids_str})
    ORDER BY product_id, total_mentions DESC
    """
    aspects_df = client.query(aspects_query).to_dataframe()
    
    # Get top pain point and delight for each
    signals_query = f"""
    SELECT product_id, signal_type, phrase, severity_rank
    FROM `{PROJECT}.{DATASET}.product_pain_delights`
    WHERE product_id IN ({ids_str}) AND severity_rank = 1
    """
    signals_df = client.query(signals_query).to_dataframe()
    
    hotels = []
    for _, hotel in hotels_df.iterrows():
        pid = hotel['product_id']
        hotel_aspects = aspects_df[aspects_df['product_id'] == pid].to_dict(orient='records')
        hotel_signals = signals_df[signals_df['product_id'] == pid]
        
        top_pain = hotel_signals[hotel_signals['signal_type'] == 'pain_point']
        top_delight = hotel_signals[hotel_signals['signal_type'] == 'delight']
        
        hotels.append({
            **hotel.to_dict(),
            "aspects": hotel_aspects,
            "top_pain": top_pain.iloc[0]['phrase'] if not top_pain.empty else None,
            "top_delight": top_delight.iloc[0]['phrase'] if not top_delight.empty else None
        })
    
    return {"hotels": hotels}

class CompareBrandsRequest(BaseModel):
    brand_ids: List[str]

@app.post("/api/compare/brands")
async def compare_brands(request: CompareBrandsRequest):
    """Compare 2-3 brands side by side"""
    client = get_bq()
    if not client:
        raise HTTPException(status_code=500, detail="Database unavailable")
    
    if len(request.brand_ids) < 2 or len(request.brand_ids) > 3:
        raise HTTPException(status_code=400, detail="Provide 2-3 brand IDs")
    
    # Get brand summaries
    brands = []
    for brand_id in request.brand_ids:
        try:
            summary = await get_brand_summary(brand_id)
            brands.append(summary)
        except:
            pass
    
    return {"brands": brands}

# ─────────────────────────────────────────
# CHAT API
# ─────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    product_id: Optional[int] = None
    brand_id: Optional[str] = None
    category: str = "hotels"
    conversation_id: Optional[str] = None

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat with SmaartAnalyst — uses geminidataanalytics agent, falls back to GenerativeModel."""
    if not request.product_id and not request.brand_id:
        return {"response": "Please select a hotel or brand first, then ask me about it.", "conversation_id": None}

    try:
        entity_name = ""
        entity_type = ""
        data_context = ""
        if not request.product_id and not request.brand_id:
            pass  # already checked above
        
        if request.product_id:
            # ── Hotel context — pull from cache where possible ──
            pid_str = str(request.product_id)
            asp_cache  = get_cache("aspects_by_pid")
            emo_cache  = get_cache("emotions_by_pid")
            pain_cache = get_cache("pain_by_pid")
            del_cache  = get_cache("delight_by_pid")

            aspects   = asp_cache.get(pid_str, []) if asp_cache else []
            emotions  = emo_cache.get(pid_str, []) if emo_cache else []
            pains     = pain_cache.get(pid_str, []) if pain_cache else []
            delights_ = del_cache.get(pid_str, []) if del_cache else []

            # Still need hotel info from BQ (name, city, rating)
            hotel_info = {}
            try:
                client2 = get_bq()
                if client2:
                    h_df = client2.query(f"SELECT * FROM `{PROJECT}.{DATASET}.hotel_master` WHERE product_id = {request.product_id}").to_dataframe()
                    if not h_df.empty:
                        hotel_info = clean_row(h_df.iloc[0].to_dict())
            except: pass

            entity_name = hotel_info.get('hotel_name', f'Hotel {request.product_id}')
            entity_type = "Hotel"
            overall_sat = hotel_info.get('overall_satisfaction', 0)
            total_reviews = hotel_info.get('review_count', 0)

            aspect_lines = "\n".join([
                f"  {a['aspect_name']}: {a.get('satisfaction_pct',0)}% satisfaction, {a.get('share_of_voice_pct',0)}% share of voice, {a.get('positive_count',0)} positive / {a.get('negative_count',0)} negative mentions"
                for a in sorted(aspects, key=lambda x: -x.get('total_mentions',0))
            ])
            emotion_lines = ", ".join([
                f"{e['emotion']} ({e.get('percentage',0)}%, {e.get('count',0)} mentions)"
                for e in sorted(emotions, key=lambda x: -x.get('count',0))[:6]
            ])
            pain_lines = "\n".join([
                f"  * {p['phrase']} — {p.get('aspect_name','')} — {p.get('mention_count',0)} mentions"
                for p in pains[:8]
            ])
            delight_lines = "\n".join([
                f"  * {d['phrase']} — {d.get('aspect_name','')} — {d.get('mention_count',0)} mentions"
                for d in delights_[:8]
            ])

            data_context = f"""
=== HOTEL INTELLIGENCE CONTEXT ===
Hotel: {entity_name}
City: {hotel_info.get('city','Unknown')} | Stars: {hotel_info.get('star_category','N/A')} | Google Rating: {hotel_info.get('google_rating','N/A')}
Total Reviews: {total_reviews:,} | Overall Satisfaction: {overall_sat}%

=== ASPECT PERFORMANCE (sorted by volume) ===
{aspect_lines if aspect_lines else 'No aspect data'}

=== GUEST EMOTIONS ===
{emotion_lines if emotion_lines else 'No emotion data'}

=== TOP PAIN POINTS (most mentioned) ===
{pain_lines if pain_lines else 'No pain point data'}

=== TOP DELIGHTS (most mentioned) ===
{delight_lines if delight_lines else 'No delight data'}

INSTRUCTION: Use ONLY the exact numbers above. Do not estimate or invent any figure.
"""

        elif request.brand_id:
            # ── Brand context — pull aspects from cache ──
            brand = request.brand_id
            asp_cache = get_cache("aspects_by_brand")
            aspects   = asp_cache.get(brand, []) if asp_cache else []

            # Get brand stats from hotel_master
            brand_stats = {"hotel_count": 0, "total_reviews": 0, "overall_satisfaction": 0, "avg_rating": 0}
            try:
                client2 = get_bq()
                if client2:
                    b_df = client2.query(f"""
                        SELECT COUNT(*) as hotel_count, SUM(review_count) as total_reviews,
                               ROUND(AVG(overall_satisfaction)) as overall_satisfaction,
                               ROUND(AVG(google_rating),1) as avg_rating
                        FROM `{PROJECT}.{DATASET}.hotel_master`
                        WHERE brand_name = '{brand}'
                    """).to_dataframe()
                    if not b_df.empty:
                        brand_stats = {k: (int(v) if k!='avg_rating' else float(v or 0)) for k,v in clean_row(b_df.iloc[0].to_dict()).items()}
            except: pass

            entity_name = brand
            entity_type = "Brand"

            aspect_lines = "\n".join([
                f"  {a['aspect_name']}: {a.get('satisfaction_pct',0)}% satisfaction, {a.get('share_of_voice_pct',0)}% share of voice, {a.get('positive_count',0)} positive / {a.get('negative_count',0)} negative mentions"
                for a in aspects
            ])

            # Get brand pain/delights from brand summary
            brand_pain, brand_delights = [], []
            try:
                bs = await get_brand_summary(brand)
                brand_pain = bs.get('pain_points', [])[:8]
                brand_delights = bs.get('delights', [])[:8]
            except: pass

            pain_lines = "\n".join([f"  * {p['phrase']} — {p.get('aspect_name','')} — {p.get('mention_count',0)} mentions" for p in brand_pain])
            delight_lines = "\n".join([f"  * {d['phrase']} — {d.get('aspect_name','')} — {d.get('mention_count',0)} mentions" for d in brand_delights])

            data_context = f"""
=== BRAND INTELLIGENCE CONTEXT ===
Brand: {entity_name}
Portfolio: {brand_stats['hotel_count']} hotels | Total Reviews: {brand_stats['total_reviews']:,}
Overall Satisfaction: {brand_stats['overall_satisfaction']}% | Avg Google Rating: {brand_stats['avg_rating']}

=== ASPECT PERFORMANCE (aggregated across all hotels) ===
{aspect_lines if aspect_lines else 'No aspect data'}

=== TOP PAIN POINTS (brand-wide) ===
{pain_lines if pain_lines else 'No pain point data'}

=== TOP DELIGHTS (brand-wide) ===
{delight_lines if delight_lines else 'No delight data'}

INSTRUCTION: Use ONLY the exact numbers above. Do not estimate or invent any figure.
"""
        
        # Build full prompt with system instructions + data context + user query
        system_prompt = get_agent_prompt(request.category)
        full_prompt = f"""{system_prompt}

{data_context}

=== USER QUERY ===
{request.message}

Remember: Use the EXACT numbers from the data above. Do NOT invent any numbers. If the user asks to see reviews, tell them to click the relevant pain point or delight on the dashboard."""

        conv_id = request.conversation_id or f"smaart-{uuid.uuid4().hex[:8]}"
        response_text = ""

        # ── Path 1: geminidataanalytics agent ─────────────────────────────
        try:
            from google.cloud import geminidataanalytics_v1alpha as gda
            cc = get_data_chat_client()
            if cc:
                parent = f"projects/{PROJECT}/locations/{AGENT_LOCATION}"
                agent_path = f"{parent}/dataAgents/{AGENT_ID}"
                conv_path = cc.conversation_path(PROJECT, AGENT_LOCATION, conv_id)

                # Create conversation if it doesn't exist yet
                try:
                    cc.get_conversation(name=conv_path)
                except Exception:
                    cc.create_conversation(request=gda.CreateConversationRequest(
                        parent=parent,
                        conversation_id=conv_id,
                        conversation=gda.Conversation(agents=[agent_path])
                    ))

                stream = cc.chat(request={
                    "parent": parent,
                    "conversation_reference": {
                        "conversation": conv_path,
                        "data_agent_context": {"data_agent": agent_path}
                    },
                    "messages": [{"user_message": {"text": full_prompt}}]
                })

                for chunk in stream:
                    if hasattr(chunk, 'system_message') and hasattr(chunk.system_message, 'text'):
                        for p in chunk.system_message.text.parts:
                            part = str(p)
                            if any(part.startswith(c) for c in ['📊','🎯','👔','📢','🛏','🛎','🍽','⚙','👥','♂','🔑','⚠','✓','✗']) or '**' in part:
                                response_text += part + "\n"
                    if hasattr(chunk, 'agent_message') and hasattr(chunk.agent_message, 'text'):
                        for p in chunk.agent_message.text.parts:
                            response_text += str(p)

                response_text = response_text.replace('💭 ', '').strip()
                print(f"[CHAT] Agent response length: {len(response_text)}")
        except Exception as agent_err:
            print(f"[CHAT] Agent path failed: {agent_err} — falling back to GenerativeModel")

        # ── Path 2: GenerativeModel fallback ──────────────────────────────
        if not response_text:
            model = get_gemini()
            if model:
                resp = model.generate_content(full_prompt)
                response_text = resp.text if resp.text else "No response generated."
            else:
                response_text = "Chat service unavailable. Please check your credentials."

        return {
            "response": response_text,
            "conversation_id": conv_id,
            "entity": entity_name,
            "entity_type": entity_type
        }
        
    except Exception as e:
        print(f"[CHAT ERROR] {e}")
        traceback.print_exc()
        return {"response": f"Error: {str(e)}", "conversation_id": None}

@app.get("/api/treemap_phrases")
async def get_treemap_phrases(
    product_id: Optional[int] = None,
    brand: Optional[str] = None,
    limit: int = 5
):
    """Top treemap phrases per aspect — served from startup cache."""
    if product_id:
        cache = get_cache("treemap_by_pid")
        result = (cache or {}).get(str(product_id), {})
        if result:
            print(f"[CACHE HIT] treemap_phrases for product {product_id}")
            return result
    elif brand:
        cache = get_cache("treemap_by_brand")
        result = (cache or {}).get(brand, {})
        if result:
            print(f"[CACHE HIT] treemap_phrases for brand {brand}")
            return result

    # Fallback to live BQ if cache miss
    client = get_bq()
    if not client:
        return {}
    VALID_ASPECTS = {1:"Dining",2:"Cleanliness",3:"Amenities",4:"Staff",5:"Room",6:"Location",7:"Value for Money"}
    try:
        if product_id:
            query = f"""
            SELECT aspect_id, treemap_name, SUM(mention_count) as mention_count
            FROM `{PROJECT}.{DATASET}.product_phrases`
            WHERE product_id = {product_id}
              AND treemap_name IS NOT NULL AND TRIM(treemap_name) != ''
              AND aspect_id IN ({','.join(str(k) for k in VALID_ASPECTS)})
            GROUP BY aspect_id, treemap_name ORDER BY aspect_id, mention_count DESC
            """
        else:
            safe = brand.replace("'","''")
            query = f"""
            SELECT p.aspect_id, p.treemap_name, SUM(p.mention_count) as mention_count
            FROM `{PROJECT}.{DATASET}.product_phrases` p
            JOIN `{PROJECT}.{DATASET}.hotel_master` h ON p.product_id = h.product_id
            WHERE h.brand_name = '{safe}'
              AND p.treemap_name IS NOT NULL AND TRIM(p.treemap_name) != ''
              AND p.aspect_id IN ({','.join(str(k) for k in VALID_ASPECTS)})
            GROUP BY p.aspect_id, p.treemap_name ORDER BY p.aspect_id, mention_count DESC
            """
        df = client.query(query).to_dataframe()
        results = {}
        for asp_id, asp_name in VALID_ASPECTS.items():
            rows = df[df['aspect_id']==asp_id].head(limit)
            if not rows.empty:
                results[asp_name] = [{"treemap_name": str(r['treemap_name']), "mention_count": int(r['mention_count'])} for _,r in rows.iterrows()]
        return results
    except Exception as e:
        print(f"[TREEMAP_PHRASES FALLBACK ERROR] {e}")
        return {}


@app.get("/api/hotel/{product_id}/segment_aspect")
async def get_segment_aspect(product_id: int):
    """Get segment x aspect matrix for heatmaps"""
    client = get_bq()
    if not client:
        raise HTTPException(status_code=500, detail="Database unavailable")
    
    query = f"""
    SELECT segment_type, segment_value, aspect_id, aspect_name,
           positive_count, negative_count, total_mentions, satisfaction_pct
    FROM `{PROJECT}.{DATASET}.product_segment_aspect`
    WHERE product_id = {product_id}
    ORDER BY segment_type, segment_value, aspect_id
    """
    df = client.query(query).to_dataframe()
    return {"data": df.to_dict(orient='records')}

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

# ─────────────────────────────────────────
# FRONTEND COMPATIBILITY ALIASES
# ─────────────────────────────────────────


@app.get("/api/star_categories")
async def get_star_categories(brand: Optional[str] = None, city: Optional[str] = None):
    """Get distinct star categories for filter dropdown"""
    client = get_bq()
    if not client:
        return []
    conditions = ["star_category IS NOT NULL"]
    if brand:
        conditions.append(f"brand_name = '{brand}'")
    if city:
        conditions.append(f"city = '{city}'")
    query = f"""
    SELECT DISTINCT star_category
    FROM `{PROJECT}.{DATASET}.hotel_master`
    WHERE {" AND ".join(conditions)}
    ORDER BY star_category
    """
    df = client.query(query).to_dataframe()
    return [int(r['star_category']) for _, r in df.iterrows() if r['star_category']]

@app.get("/api/hotel_details")
async def hotel_details_alias(product_id: Optional[int] = None, brand: Optional[str] = None,
                               city: Optional[str] = None, star: Optional[str] = None):
    client = get_bq()
    if not client:
        raise HTTPException(status_code=500, detail="Database unavailable")
    if product_id:
        return await get_hotel_summary(product_id)
    elif brand:
        try:
            safe_brand = brand.replace("'","''")
            where = f"brand_name = '{safe_brand}'"
            if city: where += f" AND city = '{city.replace(chr(39),chr(39)*2)}'"
            if star: where += f" AND star_category = '{star.replace(chr(39),chr(39)*2)}'"
            query = f"""
            SELECT COUNT(*) as hotel_count, SUM(review_count) as total_reviews,
                   ROUND(AVG(overall_satisfaction)) as overall_satisfaction,
                   ROUND(AVG(google_rating),1) as avg_rating,
                   '{safe_brand}' as brand_name
            FROM `{PROJECT}.{DATASET}.hotel_master`
            WHERE {where}
            """
            df = client.query(query).to_dataframe()
            if df.empty:
                return {"brand_name": brand, "hotel_count": 0, "review_count": 0}
            row = df.iloc[0]
            return {
                "brand_name": brand,
                "hotel_count": int(row['hotel_count'] or 0),
                "review_count": int(row['total_reviews'] or 0),
                "overall_satisfaction": int(row['overall_satisfaction'] or 0),
                "avg_rating": float(row['avg_rating'] or 0)
            }
        except Exception as e:
            print(f"[hotel_details ERROR] {e}")
            return {"brand_name": brand, "hotel_count": 0, "review_count": 0}
    return {}

@app.get("/api/drivers")
async def drivers_alias(product_id: Optional[int] = None, brand: Optional[str] = None,
                         city: Optional[str] = None, star: Optional[str] = None):
    ASPECT_ICONS = {1:"🍽️",2:"🧹",3:"🏊",4:"👨‍💼",5:"🛏️",6:"📍",7:"💰",8:"⭐"}

    if product_id:
        cached = get_cache("aspects_by_pid")
        if cached:
            rows = cached.get(str(product_id), [])
            if rows:
                total = sum(r['total_mentions'] for r in rows) or 1
                result = []
                for r in sorted(rows, key=lambda x: -x['total_mentions']):
                    pos, neg = r['positive_count'], r['negative_count']
                    sat = r['satisfaction_pct'] or (round(pos*100/(pos+neg)) if (pos+neg)>0 else 0)
                    sov = r['share_of_voice_pct'] or round(r['total_mentions']*100/total)
                    result.append({**r, 'satisfaction': sat, 'share_of_voice': sov,
                                   'icon': ASPECT_ICONS.get(r['aspect_id'], '⭐')})
                return result

    elif brand and not city and not star:
        # Only use cache when no city/star filter
        cached = get_cache("aspects_by_brand")
        if cached and brand in cached:
            rows = cached[brand]
            return [{**r, 'icon': ASPECT_ICONS.get(r['aspect_id'], '⭐')} for r in rows]

    # Fallback to BQ
    client = get_bq()
    if not client:
        raise HTTPException(status_code=500, detail="Database unavailable")
    if product_id:
        query = f"""
        SELECT aspect_id, aspect_name, positive_count, negative_count,
               total_mentions, satisfaction_pct as satisfaction, share_of_voice_pct as share_of_voice
        FROM `{PROJECT}.{DATASET}.product_aspect_summary`
        WHERE product_id = {product_id} ORDER BY total_mentions DESC
        """
    elif brand:
        safe_brand = brand.replace("'","''")
        where = f"h.brand_name = '{safe_brand}'"
        if city: where += f" AND h.city = '{city.replace(chr(39),chr(39)*2)}'"
        if star: where += f" AND h.star_category = '{star.replace(chr(39),chr(39)*2)}'"
        query = f"""
        SELECT a.aspect_id, a.aspect_name,
               SUM(a.positive_count) as positive_count, SUM(a.negative_count) as negative_count,
               SUM(a.total_mentions) as total_mentions,
               ROUND(SUM(a.positive_count)*100/NULLIF(SUM(a.positive_count)+SUM(a.negative_count),0)) as satisfaction,
               ROUND(SUM(a.total_mentions)*100/NULLIF(SUM(SUM(a.total_mentions)) OVER(),0)) as share_of_voice
        FROM `{PROJECT}.{DATASET}.product_aspect_summary` a
        JOIN `{PROJECT}.{DATASET}.hotel_master` h ON a.product_id = h.product_id
        WHERE {where} GROUP BY a.aspect_id, a.aspect_name ORDER BY total_mentions DESC
        """
    else:
        return []
    df = client.query(query).to_dataframe()
    results = []
    total_mentions = df['total_mentions'].sum() if not df.empty else 1
    for _, row in df.iterrows():
        pos = int(row.get('positive_count') or 0)
        neg = int(row.get('negative_count') or 0)
        total = int(row.get('total_mentions') or 0)
        sat = int(row.get('satisfaction') or 0) or (round(pos*100/(pos+neg)) if (pos+neg)>0 else 0)
        sov = int(row.get('share_of_voice') or 0) or (round(total*100/total_mentions) if total_mentions>0 else 0)
        results.append({
            'aspect_id': int(row.get('aspect_id') or 0),
            'aspect_name': str(row.get('aspect_name') or ''),
            'satisfaction': sat, 'share_of_voice': sov,
            'positive_count': pos, 'negative_count': neg, 'total_mentions': total,
            'icon': ASPECT_ICONS.get(int(row.get('aspect_id') or 0), '⭐')
        })
    return results

@app.get("/api/satisfaction")
async def satisfaction_alias(product_id: Optional[int] = None, brand: Optional[str] = None):
    return await drivers_alias(product_id=product_id, brand=brand)

@app.get("/api/demographics")
async def demographics_alias(product_id: Optional[int] = None, brand: Optional[str] = None):
    client = get_bq()
    if not client:
        raise HTTPException(status_code=500, detail="Database unavailable")
    if product_id:
        query = f"""
        SELECT dimension, dimension_value, review_count, pct_of_total
        FROM `{PROJECT}.{DATASET}.product_demographics`
        WHERE product_id = {product_id}
        ORDER BY dimension, review_count DESC
        """
        df = client.query(query).to_dataframe()
    elif brand:
        query = f"""
        SELECT d.dimension, d.dimension_value, SUM(d.review_count) as review_count, 0 as pct_of_total
        FROM `{PROJECT}.{DATASET}.product_demographics` d
        JOIN `{PROJECT}.{DATASET}.hotel_master` h ON d.product_id = h.product_id
        WHERE h.brand_name = '{brand}'
        GROUP BY d.dimension, d.dimension_value
        ORDER BY d.dimension, review_count DESC
        """
        df = client.query(query).to_dataframe()
    else:
        return {"gender": [], "traveler_type": [], "stay_purpose": []}
    result = {"gender": [], "traveler_type": [], "stay_purpose": []}
    for _, row in df.iterrows():
        dim = row['dimension']
        val = row['dimension_value']
        if dim in result and val and str(val).lower() not in ('unknown', 'none', 'null', ''):
            result[dim].append({
                "traveler_type": val,
                "gender": val,
                "stay_purpose": val,
                "review_count": int(row['review_count']),
                "pct_of_total": int(row.get('pct_of_total') or 0)
            })
    return result

@app.get("/api/stay_purpose_preferences")
async def stay_purpose_preferences(product_id: Optional[int] = None, brand: Optional[str] = None):
    client = get_bq()
    if not client:
        raise HTTPException(status_code=500, detail="Database unavailable")
    if product_id:
        where = f"s.product_id = {product_id}"
        join_clause = ""
    elif brand:
        where = f"h.brand_name = '{brand}'"
        join_clause = f"JOIN `{PROJECT}.{DATASET}.hotel_master` h ON s.product_id = h.product_id"
    else:
        return []
    query = f"""
    SELECT s.segment_value as stay_purpose, s.aspect_name,
           SUM(s.total_mentions) as mentions,
           ROUND(SUM(s.positive_count)*100/NULLIF(SUM(s.positive_count)+SUM(s.negative_count),0)) as satisfaction
    FROM `{PROJECT}.{DATASET}.product_segment_aspect` s
    {join_clause}
    WHERE {where} AND s.segment_type = 'stay_purpose' AND s.segment_value IS NOT NULL
    GROUP BY s.segment_value, s.aspect_name
    ORDER BY s.segment_value, mentions DESC
    """
    df = client.query(query).to_dataframe()
    if df.empty:
        return []
    result = {}
    for _, row in df.iterrows():
        sp = row['stay_purpose']
        if sp not in result:
            result[sp] = {"stay_purpose": sp, "aspects": {}}
        result[sp]["aspects"][row['aspect_name']] = {
            "satisfaction": int(row['satisfaction'] or 0),
            "mentions": int(row['mentions'])
        }
    return list(result.values())

@app.get("/api/traveler_preferences")
async def traveler_preferences(product_id: Optional[int] = None, brand: Optional[str] = None):
    client = get_bq()
    if not client:
        raise HTTPException(status_code=500, detail="Database unavailable")
    if product_id:
        where = f"s.product_id = {product_id}"
        join_clause = ""
    elif brand:
        where = f"h.brand_name = '{brand}'"
        join_clause = f"JOIN `{PROJECT}.{DATASET}.hotel_master` h ON s.product_id = h.product_id"
    else:
        return []
    query = f"""
    SELECT s.segment_value as traveler_type, s.aspect_name,
           SUM(s.total_mentions) as mentions,
           ROUND(SUM(s.positive_count)*100/NULLIF(SUM(s.positive_count)+SUM(s.negative_count),0)) as satisfaction
    FROM `{PROJECT}.{DATASET}.product_segment_aspect` s
    {join_clause}
    WHERE {where} AND s.segment_type = 'traveler_type' AND s.segment_value IS NOT NULL
    GROUP BY s.segment_value, s.aspect_name
    ORDER BY s.segment_value, mentions DESC
    """
    df = client.query(query).to_dataframe()
    if df.empty:
        return []
    result = {}
    for _, row in df.iterrows():
        tt = row['traveler_type']
        if tt not in result:
            result[tt] = {"traveler_type": tt, "aspects": {}}
        result[tt]["aspects"][row['aspect_name']] = {
            "satisfaction": int(row['satisfaction'] or 0),
            "mentions": int(row['mentions'])
        }
    return list(result.values())

@app.get("/api/comparison")
async def comparison_alias(items: str = "", compare_by: str = "hotel"):
    client = get_bq()
    if not client:
        raise HTTPException(status_code=500, detail="Database unavailable")
    item_list = [i.strip() for i in items.split("|||") if i.strip()]
    if len(item_list) < 2:
        return {}
    result = {}
    if compare_by == "hotel":
        for pid_str in item_list:
            try:
                pid = int(pid_str)
                query = f"""
                SELECT h.product_id, h.hotel_name as display_name,
                       a.aspect_name, a.satisfaction_pct, a.share_of_voice_pct, a.total_mentions,
                       a.positive_count, a.negative_count
                FROM `{PROJECT}.{DATASET}.hotel_master` h
                JOIN `{PROJECT}.{DATASET}.product_aspect_summary` a ON h.product_id = a.product_id
                WHERE h.product_id = {pid}
                """
                df = client.query(query).to_dataframe()
                if df.empty: continue
                aspects = df[["aspect_name","satisfaction_pct","share_of_voice_pct","total_mentions"]].to_dict(orient='records')
                total_pos = int(df["positive_count"].sum())
                total_neg = int(df["negative_count"].sum())
                result[pid_str] = {
                    "display_name": str(df.iloc[0]["display_name"]),
                    "aspects": aspects,
                    "overall": {
                        "total_mentions": int(df["total_mentions"].sum()),
                        "positive": total_pos, "negative": total_neg,
                        "satisfaction": round(total_pos*100/max(total_pos+total_neg,1))
                    }
                }
            except Exception:
                result[pid_str] = {"display_name": pid_str, "aspects": [], "overall": {}}
    else:
        for brand_name in item_list:
            try:
                query = f"""
                SELECT h.brand_name as display_name, a.aspect_name,
                       ROUND(SUM(a.satisfaction_pct*a.total_mentions)/NULLIF(SUM(a.total_mentions),0)) as satisfaction_pct,
                       SUM(a.total_mentions) as total_mentions,
                       SUM(a.positive_count) as positive_count, SUM(a.negative_count) as negative_count
                FROM `{PROJECT}.{DATASET}.hotel_master` h
                JOIN `{PROJECT}.{DATASET}.product_aspect_summary` a ON h.product_id = a.product_id
                WHERE h.brand_name = '{brand_name}'
                GROUP BY h.brand_name, a.aspect_name
                """
                df = client.query(query).to_dataframe()
                if df.empty: continue
                aspects = [{"aspect_name": r["aspect_name"], "satisfaction_pct": int(r["satisfaction_pct"] or 0),
                            "total_mentions": int(r["total_mentions"])} for _, r in df.iterrows()]
                total_pos = int(df["positive_count"].sum())
                total_neg = int(df["negative_count"].sum())
                result[brand_name] = {
                    "display_name": brand_name, "aspects": aspects,
                    "overall": {"total_mentions": int(df["total_mentions"].sum()),
                                "positive": total_pos, "negative": total_neg,
                                "satisfaction": round(total_pos*100/max(total_pos+total_neg,1))}
                }
            except Exception:
                result[brand_name] = {"display_name": brand_name, "aspects": [], "overall": {}}
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
