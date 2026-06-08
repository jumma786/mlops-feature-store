"""
Feature Serving API
====================
Serves pre-computed customer features for online inference.
Simulates an online feature store (without Redis dependency).

Endpoints:
  GET  /health
  GET  /features/{customer_id}      — Get all features for a customer
  POST /features/batch              — Get features for multiple customers
  GET  /segments                    — Get segment distribution
  GET  /top-customers               — Get top customers by monetary value
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OFFLINE_STORE = "feature_store/offline"


def load_feature_store() -> pd.DataFrame:
    """Load the latest feature snapshot from offline store."""
    path = os.path.join(OFFLINE_STORE, "customer_features")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature store not found at {path}. Run src/features/engineer.py first.")

    files = sorted([f for f in os.listdir(path) if f.endswith(".parquet")])
    if not files:
        raise FileNotFoundError("No feature files found.")

    latest = os.path.join(path, files[-1])
    df = pd.read_parquet(latest)
    df["Customer ID"] = df["Customer ID"].astype(str)
    logger.info(f"Feature store loaded: {df.shape} | Latest: {files[-1]}")
    return df


# Load features on startup
features_df = load_feature_store()
FEATURE_COLS = [c for c in features_df.columns if c not in ["Customer ID", "snapshot_date"]]

app = FastAPI(
    title="Feature Store API",
    description="Serves pre-computed customer features from UCI Online Retail II. Project 6/10 of MLOps Portfolio.",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class BatchRequest(BaseModel):
    customer_ids: List[str]


class CustomerFeatures(BaseModel):
    customer_id:           str
    recency_days:          Optional[int]    = None
    frequency:             Optional[int]    = None
    monetary:              Optional[float]  = None
    avg_order_value:       Optional[float]  = None
    std_order_value:       Optional[float]  = None
    max_order_value:       Optional[float]  = None
    purchase_span_days:    Optional[int]    = None
    r_score:               Optional[int]    = None
    f_score:               Optional[int]    = None
    m_score:               Optional[int]    = None
    rfm_score:             Optional[int]    = None
    segment:               Optional[str]    = None
    total_items:           Optional[int]    = None
    unique_products:       Optional[int]    = None
    avg_basket_size:       Optional[float]  = None
    avg_unit_price:        Optional[float]  = None
    unique_countries:      Optional[int]    = None
    weekend_purchase_rate: Optional[float]  = None
    found:                 bool             = True


def get_customer_features(customer_id: str) -> dict:
    row = features_df[features_df["Customer ID"] == str(customer_id)]
    if row.empty:
        return {"customer_id": customer_id, "found": False}
    data = row.iloc[0].to_dict()
    data["customer_id"] = data.pop("Customer ID")
    data["found"] = True
    # Convert numpy types
    for k, v in data.items():
        if hasattr(v, "item"):
            data[k] = v.item()
    return data


@app.get("/")
def root():
    return {
        "message": "Feature Store API — Online Retail II",
        "customers": len(features_df),
        "features": len(FEATURE_COLS),
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "customers": len(features_df),
        "feature_cols": len(FEATURE_COLS),
        "snapshot_date": str(features_df["snapshot_date"].iloc[0]) if "snapshot_date" in features_df.columns else "unknown"
    }


@app.get("/features/{customer_id}")
def get_features(customer_id: str):
    """Get all features for a single customer."""
    data = get_customer_features(customer_id)
    if not data.get("found"):
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    return data


@app.post("/features/batch")
def get_features_batch(request: BatchRequest):
    """Get features for multiple customers."""
    results = []
    for cid in request.customer_ids:
        results.append(get_customer_features(cid))
    found = sum(1 for r in results if r.get("found"))
    return {"results": results, "total": len(results), "found": found, "not_found": len(results) - found}


@app.get("/segments")
def get_segments():
    """Get customer segment distribution."""
    if "segment" not in features_df.columns:
        raise HTTPException(status_code=404, detail="Segment column not found")
    dist = features_df["segment"].value_counts().to_dict()
    total = len(features_df)
    return {
        "segments": {k: {"count": v, "pct": round(v/total*100, 1)} for k, v in dist.items()},
        "total_customers": total
    }


@app.get("/top-customers")
def get_top_customers(n: int = 10, by: str = "monetary"):
    """Get top N customers by a metric."""
    valid = ["monetary", "frequency", "recency_days", "rfm_score"]
    if by not in valid:
        raise HTTPException(status_code=400, detail=f"'by' must be one of {valid}")
    ascending = by == "recency_days"
    top = features_df.nsmallest(n, by) if ascending else features_df.nlargest(n, by)
    return {"top_customers": top[["Customer ID", by, "segment", "monetary"]].to_dict(orient="records"), "ranked_by": by}


if __name__ == "__main__":
    uvicorn.run("src.serving.api:app", host="0.0.0.0", port=8001, reload=True)
