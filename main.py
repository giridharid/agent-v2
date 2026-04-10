"""
Smaartbrand Intelligence Dashboard - FastAPI Backend v2
Multi-vertical support with BigQuery flat tables, caching, and drill-down
"""

from fastapi import FastAPI, HTTPException, Query
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
    "brands": {"data": None, "timestamp": 0},
    "hotels": {"data": None, "timestamp": 0},
    "cities": {"data": None, "timestamp": 0},
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

def init_gemini():
    """
    Initialize Gemini Data Analytics Agent.
    Uses service account credentials (same as BigQuery).
    Agent ID: agent_024eedb9-0b86-4101-82a5-f4b3d72c5ee3
    """
    global gemini_model
    
    # Get agent ID from env or use default
    agent_id = os.environ.get("GEMINI_AGENT_ID", "agent_024eedb9-0b86-4101-82a5-f4b3d72c5ee3")
    
    try:
        # Try to use service account credentials for Gemini
        gcp_creds = os.environ.get("GCP_CREDENTIALS_JSON", "")
        if gcp_creds:
            gcp_creds = gcp_creds.strip().strip('"').strip("'")
            if gcp_creds.startswith("{"):
                creds_dict = json.loads(gcp_creds)
            else:
                padding = 4 - len(gcp_creds) % 4
                if padding != 4:
                    gcp_creds += "=" * padding
                creds_dict = json.loads(base64.b64decode(gcp_creds).decode('utf-8'))
            
            # Initialize Vertex AI with project from credentials
            import vertexai
            from vertexai.generative_models import GenerativeModel
            
            project_id = creds_dict.get('project_id', PROJECT)
            credentials = service_account.Credentials.from_service_account_info(creds_dict)
            
            vertexai.init(project=project_id, location="us-central1", credentials=credentials)
            
            # Use gemini-2.0-flash model (agent context will be added via system prompt)
            gemini_model = GenerativeModel("gemini-2.0-flash-001")
            print(f"[SUCCESS] Gemini model initialized via Vertex AI (project: {project_id})")
            return gemini_model
        else:
            # Fallback: try API key if available
            api_key = os.environ.get("GEMINI_API_KEY", "")
            if api_key:
                genai.configure(api_key=api_key)
                gemini_model = genai.GenerativeModel('gemini-2.0-flash')
                print("[SUCCESS] Gemini model initialized via API key")
                return gemini_model
            else:
                print("[WARNING] No Gemini credentials available")
                return None
                
    except ImportError:
        # vertexai not installed, try genai
        print("[INFO] vertexai not installed, trying google-generativeai")
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            print("[SUCCESS] Gemini model initialized via API key")
            return gemini_model
        return None
    except Exception as e:
        print(f"[ERROR] Gemini init error: {e}")
        traceback.print_exc()
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
    brand_id: Optional[str] = None,
    city: Optional[str] = None,
    stars: Optional[str] = None,
    category: str = "hotels"
):
    """Get hotels with optional filters"""
    cached = get_cache("hotels")
    if cached:
        filtered = cached
        if brand_id:
            filtered = [h for h in filtered if str(h.get('brand_id')) == str(brand_id)]
        if city and city != "All Cities":
            filtered = [h for h in filtered if h.get('city') == city]
        if stars and stars != "All Stars":
            star_val = float(stars.replace('+', ''))
            filtered = [h for h in filtered if (h.get('star_category') or 0) >= star_val]
        return {"hotels": filtered}
    
    client = get_bq()
    if not client:
        raise HTTPException(status_code=500, detail="Database unavailable")
    
    conditions = ["1=1"]
    if brand_id:
        conditions.append(f"brand_id = '{brand_id}'")
    if city and city != "All Cities":
        conditions.append(f"city = '{city}'")
    if stars and stars != "All Stars":
        star_val = float(stars.replace('+', ''))
        conditions.append(f"star_category >= {star_val}")
    
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
async def get_cities(brand_id: Optional[str] = None):
    """Get unique cities, optionally filtered by brand"""
    cached = get_cache("hotels")
    if cached:
        filtered = cached
        if brand_id:
            filtered = [h for h in filtered if str(h.get('brand_id')) == str(brand_id)]
        cities = sorted(set(h.get('city') for h in filtered if h.get('city')))
        return {"cities": cities}
    
    cached_cities = get_cache("cities")
    if cached_cities:
        return {"cities": cached_cities}
    
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
async def get_brand_summary(brand_id: str):
    """Get aggregated brand-level summary"""
    client = get_bq()
    if not client:
        raise HTTPException(status_code=500, detail="Database unavailable")
    
    # Brand info
    brand_query = f"""
    SELECT brand_id, brand_name
    FROM `{PROJECT}.{DATASET}.brand_master`
    WHERE brand_id = '{brand_id}'
    """
    brand_df = client.query(brand_query).to_dataframe()
    if brand_df.empty:
        raise HTTPException(status_code=404, detail="Brand not found")
    
    brand_info = brand_df.iloc[0].to_dict()
    
    # Hotel count and review count
    hotels_query = f"""
    SELECT COUNT(*) as hotel_count, SUM(review_count) as total_reviews,
           ROUND(AVG(overall_satisfaction)) as overall_satisfaction,
           ROUND(AVG(google_rating),1) as avg_rating
    FROM `{PROJECT}.{DATASET}.hotel_master`
    WHERE brand_name = '{brand_id}'
    """
    hotels_df = client.query(hotels_query).to_dataframe()
    stats = hotels_df.iloc[0].to_dict() if not hotels_df.empty else {'hotel_count':0,'total_reviews':0,'overall_satisfaction':0,'avg_rating':0}
    
    # Aspects aggregated
    aspects_query = f"""
    SELECT aspect_id, aspect_name,
           SUM(positive_count) as positive_count,
           SUM(negative_count) as negative_count,
           SUM(total_mentions) as total_mentions
    FROM `{PROJECT}.{DATASET}.product_aspect_summary`
    WHERE brand_id IN (SELECT DISTINCT brand_id FROM `{PROJECT}.{DATASET}.hotel_master` WHERE brand_name = '{brand_id}')
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
    
    # Emotions aggregated
    emotions_query = f"""
    SELECT emotion, SUM(mention_count) as count
    FROM `{PROJECT}.{DATASET}.product_emotions`
    WHERE brand_id IN (SELECT DISTINCT brand_id FROM `{PROJECT}.{DATASET}.hotel_master` WHERE brand_name = '{brand_id}')
    GROUP BY emotion
    ORDER BY count DESC
    """
    emotions_df = client.query(emotions_query).to_dataframe()
    total_emotion = emotions_df['count'].sum()
    emotions = [
        {"emotion": row['emotion'], "count": int(row['count']),
         "percentage": round(row['count'] * 100 / total_emotion) if total_emotion > 0 else 0}
        for _, row in emotions_df.iterrows()
    ]
    
    # Demographics aggregated
    demo_query = f"""
    SELECT dimension, dimension_value, SUM(review_count) as count
    FROM `{PROJECT}.{DATASET}.product_demographics`
    WHERE brand_id IN (SELECT DISTINCT brand_id FROM `{PROJECT}.{DATASET}.hotel_master` WHERE brand_name = '{brand_id}')
    GROUP BY dimension, dimension_value
    ORDER BY dimension, count DESC
    """
    demo_df = client.query(demo_query).to_dataframe()
    
    demographics = {"traveler_type": [], "gender": [], "stay_purpose": []}
    for dim in ["traveler_type", "gender", "stay_purpose"]:
        dim_data = demo_df[demo_df['dimension'] == dim]
        total_dim = dim_data['count'].sum()
        for _, row in dim_data.iterrows():
            if row['dimension_value'] and row['dimension_value'].strip():
                demographics[dim].append({
                    "dimension_value": row['dimension_value'],
                    "count": int(row['count']),
                    "pct_of_total": round(row['count'] * 100 / total_dim) if total_dim > 0 else 0
                })
    
    # Pain points aggregated
    pain_brand_query = f"""
    SELECT p.phrase, p.aspect_name, SUM(p.mention_count) as mention_count, p.signal_type
    FROM `{PROJECT}.{DATASET}.product_pain_delights` p
    JOIN `{PROJECT}.{DATASET}.hotel_master` h ON p.product_id = h.product_id
    WHERE h.brand_name = '{brand_id}'
    GROUP BY p.phrase, p.aspect_name, p.signal_type
    ORDER BY mention_count DESC
    """
    try:
        pain_brand_df = client.query(pain_brand_query).to_dataframe()
        brand_pain = [{"phrase":r["phrase"],"aspect_name":r["aspect_name"],"mention_count":int(r["mention_count"])} for _,r in pain_brand_df[pain_brand_df["signal_type"]=="pain_point"].head(10).iterrows()]
        brand_delights = [{"phrase":r["phrase"],"aspect_name":r["aspect_name"],"mention_count":int(r["mention_count"])} for _,r in pain_brand_df[pain_brand_df["signal_type"]=="delight"].head(10).iterrows()]
    except:
        brand_pain, brand_delights = [], []

    # RD signals aggregated
    rd_brand_query = f"""
    SELECT r.signal_type as rd_signal, r.phrase, SUM(r.mention_count) as mention_count
    FROM `{PROJECT}.{DATASET}.product_rd_signals` r
    JOIN `{PROJECT}.{DATASET}.hotel_master` h ON r.product_id = h.product_id
    WHERE h.brand_name = '{brand_id}'
    GROUP BY r.signal_type, r.phrase
    ORDER BY mention_count DESC
    """
    try:
        rd_brand_df = client.query(rd_brand_query).to_dataframe()
        brand_rd = {"feature_request":[],"price_feedback":[],"expectation_gap":[]}
        for _,row in rd_brand_df.iterrows():
            sig = str(row["rd_signal"] or "")
            if sig in brand_rd:
                brand_rd[sig].append({"phrase":str(row["phrase"]),"mention_count":int(row["mention_count"])})
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

    loop = asyncio.get_event_loop()

    def run_query(sql):
        try:
            return client.query(sql).to_dataframe()
        except Exception as e:
            print(f"[BQ ERROR] {e}")
            import pandas as pd
            return pd.DataFrame()

    # Run all queries in parallel using thread pool
    queries = {
        "hotel": f"SELECT * FROM `{PROJECT}.{DATASET}.hotel_master` WHERE product_id = {product_id}",
        "aspects": f"""SELECT aspect_id, aspect_name, positive_count, negative_count,
               total_mentions, satisfaction_pct, share_of_voice_pct
               FROM `{PROJECT}.{DATASET}.product_aspect_summary`
               WHERE product_id = {product_id} ORDER BY total_mentions DESC""",
        "emotions": f"""SELECT emotion, mention_count as count, pct_of_total as percentage
               FROM `{PROJECT}.{DATASET}.product_emotions`
               WHERE product_id = {product_id} ORDER BY mention_count DESC""",
        "demo": f"""SELECT dimension, dimension_value, review_count as count, pct_of_total
               FROM `{PROJECT}.{DATASET}.product_demographics`
               WHERE product_id = {product_id} ORDER BY dimension, review_count DESC""",
        "pain": f"""SELECT phrase, treemap_name, aspect_id, aspect_name, mention_count, severity_rank
               FROM `{PROJECT}.{DATASET}.product_pain_delights`
               WHERE product_id = {product_id} AND signal_type = 'pain_point'
               ORDER BY severity_rank LIMIT 10""",
        "delight": f"""SELECT phrase, treemap_name, aspect_id, aspect_name, mention_count, severity_rank
               FROM `{PROJECT}.{DATASET}.product_pain_delights`
               WHERE product_id = {product_id} AND signal_type = 'delight'
               ORDER BY severity_rank LIMIT 10""",
        "rd": f"""SELECT signal_type as rd_signal, phrase, treemap_name, mention_count
               FROM `{PROJECT}.{DATASET}.product_rd_signals`
               WHERE product_id = {product_id} ORDER BY rd_signal, mention_count DESC"""
    }

    results = await asyncio.gather(*[
        loop.run_in_executor(None, run_query, sql)
        for sql in queries.values()
    ])
    dfs = dict(zip(queries.keys(), results))

    if dfs["hotel"].empty:
        raise HTTPException(status_code=404, detail="Hotel not found")

    hotel_info = clean_row(dfs["hotel"].iloc[0].to_dict())
    aspects = [clean_row(r) for r in dfs["aspects"].to_dict(orient='records')]
    emotions = [clean_row(r) for r in dfs["emotions"].to_dict(orient='records')]

    demographics = {"traveler_type": [], "gender": [], "stay_purpose": []}
    for _, row in dfs["demo"].iterrows():
        dim = str(row.get('dimension', ''))
        val = row.get('dimension_value')
        if dim in demographics and val and str(val).lower() not in ('unknown', 'none', 'null', ''):
            demographics[dim].append({
                "dimension_value": str(val),
                "count": int(row['count'] or 0),
                "pct_of_total": int(row['pct_of_total'] or 0)
            })

    pain_points = [clean_row(r) for r in dfs["pain"].to_dict(orient='records')]
    delights = [clean_row(r) for r in dfs["delight"].to_dict(orient='records')]

    rd_signals = {"feature_request": [], "price_feedback": [], "expectation_gap": []}
    for _, row in dfs["rd"].iterrows():
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
    
    # Build filter based on signal type
    if request.signal_type == "pain_point":
        filter_clause = "pain_point = 1"
    elif request.signal_type == "delight":
        filter_clause = "delight = 1"
    elif request.signal_type == "rd_signal":
        filter_clause = "rd_signal IS NOT NULL"
    else:
        filter_clause = "1=1"
    
    query = f"""
    SELECT review_text, sentiment_text, star_rating, reviewer_name,
           review_date, traveler_type, stay_purpose, emotion
    FROM `{PROJECT}.{DATASET}.review_drilldown`
    WHERE CAST(product_id AS STRING) = CAST({request.product_id} AS STRING)
      AND LOWER(phrase) = LOWER('{request.phrase.replace("'", "''")}') 
      AND {filter_clause}
    LIMIT {request.limit}
    """
    
    try:
        df = client.query(query).to_dataframe()
        reviews = df.to_dict(orient='records')
        
        # Get total count
        count_query = f"""
        SELECT COUNT(*) as total
        FROM `{PROJECT}.{DATASET}.review_drilldown`
        WHERE CAST(product_id AS STRING) = CAST({request.product_id} AS STRING)
          AND LOWER(phrase) = LOWER('{request.phrase.replace("'", "''")}')
          AND {filter_clause}
        """
        count_df = client.query(count_query).to_dataframe()
        total_count = int(count_df.iloc[0]['total'])
        
        return {
            "reviews": reviews,
            "total_count": total_count,
            "phrase": request.phrase,
            "signal_type": request.signal_type
        }
    except Exception as e:
        print(f"[DRILLDOWN ERROR] {e}")
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
    """Chat with SmaartAnalyst agent"""
    model = get_gemini()
    client = get_bq()
    
    if not model:
        return {"response": "Chat service unavailable. Please set GEMINI_API_KEY.", "conversation_id": None}
    
    if not client:
        return {"response": "Database unavailable.", "conversation_id": None}
    
    try:
        # Get context data
        entity_name = ""
        entity_type = ""
        data_context = ""
        
        if request.product_id:
            # Get hotel data
            summary = await get_hotel_summary(request.product_id)
            entity_name = summary.get('hotel_name', 'Unknown Hotel')
            entity_type = "Hotel"
            
            # Format data context
            aspect_lines = "\n".join([
                f"  {a['aspect_name']}: {a.get('satisfaction_pct', 0)}% satisfaction, {a.get('share_of_voice_pct', 0)}% SOV"
                for a in summary.get('aspects', [])
            ])
            
            emotion_lines = ", ".join([
                f"{e['emotion']} {e.get('percentage', 0)}%"
                for e in summary.get('emotions', [])[:5]
            ])
            
            pain_lines = "\n".join([
                f"  * {p['phrase']} ({p['aspect_name']}, {p['mention_count']} mentions)"
                for p in summary.get('pain_points', [])[:5]
            ])
            
            delight_lines = "\n".join([
                f"  * {d['phrase']} ({d['aspect_name']}, {d['mention_count']} mentions)"
                for d in summary.get('delights', [])[:5]
            ])
            
            data_context = f"""
=== CURRENT CONTEXT ===
Category: {request.category}
Hotel: {entity_name} (ID: {request.product_id})
City: {summary.get('city', 'Unknown')}
Star Rating: {summary.get('star_category', 'N/A')}
Overall Satisfaction: {summary.get('overall_satisfaction', 0)}%

=== ASPECT SATISFACTION ===
{aspect_lines}

=== TOP EMOTIONS ===
{emotion_lines}

=== PAIN POINTS (Top 5) ===
{pain_lines}

=== DELIGHTS (Top 5) ===
{delight_lines}
"""
        elif request.brand_id:
            # Get brand data
            summary = await get_brand_summary(request.brand_id)
            entity_name = summary.get('brand_name', 'Unknown Brand')
            entity_type = "Brand"
            
            aspect_lines = "\n".join([
                f"  {a['aspect_name']}: {a.get('satisfaction_pct', 0)}% satisfaction"
                for a in summary.get('aspects', [])
            ])
            
            data_context = f"""
=== CURRENT CONTEXT ===
Category: {request.category}
Brand: {entity_name} (ID: {request.brand_id})
Hotels: {summary.get('hotel_count', 0)}
Total Reviews: {summary.get('total_reviews', 0)}
Overall Satisfaction: {summary.get('overall_satisfaction', 0)}%

=== ASPECT SATISFACTION ===
{aspect_lines}
"""
        
        # Check for drill-down request
        drill_down_keywords = ["show me reviews", "see reviews", "drill down", "actual reviews", 
                              "what guests said", "guest feedback about", "reviews about"]
        is_drilldown = any(kw in request.message.lower() for kw in drill_down_keywords)
        
        if is_drilldown and request.product_id:
            # Try to extract phrase from message
            drilldown_context = "\n\n[User is requesting to see actual reviews. If you can identify a specific phrase or topic, query the review_drilldown table and show 3-5 sample reviews with the sentiment_text highlighted using ==text==.]"
            data_context += drilldown_context
        
        # Get agent prompt based on category
        system_prompt = get_agent_prompt(request.category)
        
        # Build full prompt
        full_prompt = f"""{system_prompt}

{data_context}

=== USER QUERY ===
{request.message}

Remember: Use the EXACT numbers from the data above. Do NOT invent any numbers. If user asks for reviews, indicate that they can click on specific pain points or delights to see actual guest reviews."""
        
        # Call Gemini
        response = model.generate_content(full_prompt)
        response_text = response.text if response.text else "No response generated."
        
        return {
            "response": response_text,
            "conversation_id": request.conversation_id,
            "entity": entity_name,
            "entity_type": entity_type
        }
        
    except Exception as e:
        print(f"[CHAT ERROR] {e}")
        traceback.print_exc()
        return {"response": f"Error: {str(e)}", "conversation_id": None}

# ─────────────────────────────────────────
# SEGMENT APIs (for heatmaps)
# ─────────────────────────────────────────
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
async def hotel_details_alias(product_id: Optional[int] = None, brand: Optional[str] = None):
    client = get_bq()
    if not client:
        raise HTTPException(status_code=500, detail="Database unavailable")
    if product_id:
        return await get_hotel_summary(product_id)
    elif brand:
        try:
            query = f"""
            SELECT COUNT(*) as hotel_count, SUM(review_count) as total_reviews,
                   ROUND(AVG(overall_satisfaction)) as overall_satisfaction,
                   ROUND(AVG(google_rating),1) as avg_rating,
                   '{brand}' as brand_name
            FROM `{PROJECT}.{DATASET}.hotel_master`
            WHERE brand_name = '{brand}'
            """
            df = client.query(query).to_dataframe()
            if df.empty:
                return {{"brand_name": brand, "hotel_count": 0, "review_count": 0}}
            row = df.iloc[0]
            return {{
                "brand_name": brand,
                "hotel_count": int(row['hotel_count'] or 0),
                "review_count": int(row['total_reviews'] or 0),
                "overall_satisfaction": int(row['overall_satisfaction'] or 0),
                "avg_rating": float(row['avg_rating'] or 0)
            }}
        except Exception as e:
            return {{"brand_name": brand, "hotel_count": 0, "review_count": 0}}
    return {}

@app.get("/api/drivers")
async def drivers_alias(product_id: Optional[int] = None, brand: Optional[str] = None):
    client = get_bq()
    if not client:
        raise HTTPException(status_code=500, detail="Database unavailable")
    ASPECT_ICONS = {1:"🍽️",2:"🧹",3:"🏊",4:"👨‍💼",5:"🛏️",6:"📍",7:"💰",8:"⭐"}
    if product_id:
        query = f"""
        SELECT aspect_id, aspect_name, positive_count, negative_count,
               total_mentions, satisfaction_pct as satisfaction, share_of_voice_pct as share_of_voice
        FROM `{PROJECT}.{DATASET}.product_aspect_summary`
        WHERE product_id = {product_id}
        ORDER BY total_mentions DESC
        """
    elif brand:
        query = f"""
        SELECT a.aspect_id, a.aspect_name,
               SUM(a.positive_count) as positive_count, SUM(a.negative_count) as negative_count,
               SUM(a.total_mentions) as total_mentions,
               ROUND(SUM(a.positive_count)*100/NULLIF(SUM(a.positive_count)+SUM(a.negative_count),0)) as satisfaction,
               ROUND(SUM(a.total_mentions)*100/NULLIF(SUM(SUM(a.total_mentions)) OVER(),0)) as share_of_voice
        FROM `{PROJECT}.{DATASET}.product_aspect_summary` a
        JOIN `{PROJECT}.{DATASET}.hotel_master` h ON a.product_id = h.product_id
        WHERE h.brand_name = '{brand}'
        GROUP BY a.aspect_id, a.aspect_name
        ORDER BY total_mentions DESC
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
        sat = int(row.get('satisfaction') or 0)
        sov = int(row.get('share_of_voice') or 0)
        if sat == 0 and (pos + neg) > 0:
            sat = round(pos * 100 / (pos + neg))
        if sov == 0 and total > 0 and total_mentions > 0:
            sov = round(total * 100 / total_mentions)
        results.append({
            'aspect_id': int(row.get('aspect_id') or 0),
            'aspect_name': str(row.get('aspect_name') or ''),
            'satisfaction': sat,
            'share_of_voice': sov,
            'positive_count': pos,
            'negative_count': neg,
            'total_mentions': total,
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
