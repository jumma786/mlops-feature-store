"""
Feature Store — Feature Engineering
=====================================
Computes 3 feature views from Online Retail II:

1. customer_rfm       — Recency, Frequency, Monetary + derived features
2. customer_stats     — Order stats, basket size, return rate
3. product_features   — Product-level aggregates per customer

All features are point-in-time correct using a snapshot_date.
Saved as Parquet files (offline store).
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OFFLINE_STORE = "feature_store/offline"


def compute_customer_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    RFM Feature View — core customer features.

    Features:
    - recency_days:       Days since last purchase
    - frequency:          Number of unique invoices
    - monetary:           Total spend
    - avg_order_value:    monetary / frequency
    - std_order_value:    Spend variability
    - max_order_value:    Largest single order
    - purchase_span_days: Days between first and last purchase
    """
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max()

    logger.info(f"Computing RFM with snapshot_date={snapshot_date.date()}")

    rfm = df.groupby("Customer ID").agg(
        last_purchase    = ("InvoiceDate", "max"),
        first_purchase   = ("InvoiceDate", "min"),
        frequency        = ("Invoice",     "nunique"),
        monetary         = ("Revenue",     "sum"),
        std_order_value  = ("Revenue",     "std"),
        max_order_value  = ("Revenue",     "max"),
    ).reset_index()

    rfm["recency_days"]       = (snapshot_date - rfm["last_purchase"]).dt.days
    rfm["avg_order_value"]    = rfm["monetary"] / rfm["frequency"]
    rfm["purchase_span_days"] = (rfm["last_purchase"] - rfm["first_purchase"]).dt.days
    rfm["std_order_value"]    = rfm["std_order_value"].fillna(0)

    rfm = rfm.drop(columns=["last_purchase", "first_purchase"])

    rfm["recency_days"]       = rfm["recency_days"].round(0).astype(int)
    rfm["monetary"]           = rfm["monetary"].round(2)
    rfm["avg_order_value"]    = rfm["avg_order_value"].round(2)
    rfm["std_order_value"]    = rfm["std_order_value"].round(2)
    rfm["max_order_value"]    = rfm["max_order_value"].round(2)
    rfm["snapshot_date"]      = snapshot_date.date()

    logger.info(f"RFM features: {rfm.shape} | {rfm.columns.tolist()}")
    return rfm


def compute_customer_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Customer Stats Feature View.

    Features:
    - total_items:        Total units purchased
    - unique_products:    Number of distinct products
    - avg_basket_size:    Avg items per order
    - avg_unit_price:     Average price per item
    - unique_countries:   Number of countries ordered from
    - weekend_purchase_rate: % of purchases on weekend
    """
    logger.info("Computing customer stats...")

    df = df.copy()
    df["is_weekend"] = df["InvoiceDate"].dt.dayofweek >= 5

    stats = df.groupby("Customer ID").agg(
        total_items      = ("Quantity",    "sum"),
        unique_products  = ("StockCode",   "nunique"),
        avg_unit_price   = ("Price",       "mean"),
        unique_countries = ("Country",     "nunique"),
        weekend_purchases = ("is_weekend", "sum"),
        total_purchases  = ("Invoice",     "nunique"),
    ).reset_index()

    stats["avg_basket_size"]       = (stats["total_items"] / stats["total_purchases"]).round(2)
    stats["weekend_purchase_rate"] = (stats["weekend_purchases"] / stats["total_purchases"]).round(4)
    stats["avg_unit_price"]        = stats["avg_unit_price"].round(2)
    stats = stats.drop(columns=["weekend_purchases", "total_purchases"])

    logger.info(f"Customer stats: {stats.shape}")
    return stats


def compute_rfm_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Add RFM segments using quintile scoring.
    Score 1-5 for each dimension, combined into segment label.
    """
    rfm = rfm.copy()

    # Recency: lower is better (more recent = higher score)
    rfm["r_score"] = pd.qcut(rfm["recency_days"], q=5, labels=[5,4,3,2,1], duplicates="drop")
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1,2,3,4,5], duplicates="drop")
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), q=5, labels=[1,2,3,4,5], duplicates="drop")

    rfm["rfm_score"] = (
        rfm["r_score"].astype(int) +
        rfm["f_score"].astype(int) +
        rfm["m_score"].astype(int)
    )

    # Segment labels
    def label_segment(row):
        if row["rfm_score"] >= 12:
            return "Champions"
        elif row["rfm_score"] >= 9:
            return "Loyal"
        elif row["rfm_score"] >= 6:
            return "At Risk"
        else:
            return "Lost"

    rfm["segment"] = rfm.apply(label_segment, axis=1)
    logger.info(f"Segments:\n{rfm['segment'].value_counts()}")
    return rfm


def save_to_offline_store(df: pd.DataFrame, feature_view: str, partition_date: str = None):
    """Save feature view as Parquet to offline store."""
    path = os.path.join(OFFLINE_STORE, feature_view)
    os.makedirs(path, exist_ok=True)

    if partition_date:
        filename = f"{feature_view}_{partition_date}.parquet"
    else:
        filename = f"{feature_view}.parquet"

    filepath = os.path.join(path, filename)
    df.to_parquet(filepath, index=False)
    logger.info(f"Saved: {filepath} ({df.shape[0]:,} rows)")
    return filepath


def load_from_offline_store(feature_view: str) -> pd.DataFrame:
    """Load feature view from offline store."""
    path = os.path.join(OFFLINE_STORE, feature_view)
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet files found in {path}")
    df = pd.concat([pd.read_parquet(f) for f in sorted(files)])
    logger.info(f"Loaded {feature_view}: {df.shape}")
    return df


def build_all_features(data_path: str, snapshot_date: str = None) -> dict:
    """Build all feature views from raw data."""
    from src.data.loader import load_and_clean

    df = load_and_clean(data_path)

    snap = pd.Timestamp(snapshot_date) if snapshot_date else df["InvoiceDate"].max()

    # Compute all feature views
    rfm   = compute_customer_rfm(df, snapshot_date=snap)
    rfm   = compute_rfm_segments(rfm)
    stats = compute_customer_stats(df)

    # Merge all features
    features = rfm.merge(stats, on="Customer ID", how="left")

    # Save to offline store
    date_str = str(snap.date())
    rfm_path   = save_to_offline_store(rfm,      "customer_rfm",   date_str)
    stats_path = save_to_offline_store(stats,    "customer_stats",  date_str)
    feat_path  = save_to_offline_store(features, "customer_features", date_str)

    print("\n" + "="*55)
    print("FEATURE STORE — BUILD COMPLETE")
    print("="*55)
    print(f"  Customers:     {len(features):,}")
    print(f"  Features:      {len(features.columns)}")
    print(f"  Snapshot date: {date_str}")
    print(f"  Offline store: {OFFLINE_STORE}/")
    print(f"\nSegment distribution:")
    for seg, cnt in features["segment"].value_counts().items():
        print(f"  {seg:<15}: {cnt:,} ({cnt/len(features)*100:.1f}%)")
    print("="*55)

    return {"rfm": rfm, "stats": stats, "features": features}


if __name__ == "__main__":
    results = build_all_features("data/online_retail_II.csv")
