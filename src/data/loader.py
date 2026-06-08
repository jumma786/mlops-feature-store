import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_raw(data_path):
    logger.info(f"Loading: {data_path}")
    df = pd.read_csv(data_path, encoding="latin-1")
    logger.info(f"Raw shape: {df.shape}")
    return df

def clean(df):
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    cancelled = df["Invoice"].astype(str).str.startswith("C")
    df = df[~cancelled]
    df = df.dropna(subset=["Customer ID"])
    df["Customer ID"] = df["Customer ID"].astype(int).astype(str)
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]
    df["Revenue"] = df["Quantity"] * df["Price"]
    df["InvoiceYear"]  = df["InvoiceDate"].dt.year
    df["InvoiceMonth"] = df["InvoiceDate"].dt.month
    df["InvoiceDay"]   = df["InvoiceDate"].dt.day
    logger.info(f"Clean shape: {df.shape}")
    logger.info(f"Unique customers: {df['Customer ID'].nunique():,}")
    logger.info(f"Total revenue: £{df['Revenue'].sum():,.2f}")
    return df

def load_and_clean(data_path):
    df = load_raw(data_path)
    df = clean(df)
    return df