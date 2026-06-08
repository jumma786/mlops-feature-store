"""
Test suite for mlops-feature-store.
Run: pytest tests/ -v --cov=src
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.engineer import (
    compute_customer_rfm,
    compute_customer_stats,
    compute_rfm_segments,
)


@pytest.fixture
def sample_df():
    """Create minimal sample Online Retail II data."""
    return pd.DataFrame({
        "Invoice":     ["INV001","INV001","INV002","INV003","INV003","INV004"],
        "StockCode":   ["A001","A002","A001","A003","A001","A002"],
        "Description": ["Product A","Product B","Product A","Product C","Product A","Product B"],
        "Quantity":    [10, 5, 3, 8, 2, 6],
        "InvoiceDate": pd.to_datetime([
            "2021-01-01","2021-01-01","2021-02-15",
            "2021-03-01","2021-03-01","2021-04-10"
        ]),
        "Price":       [2.5, 5.0, 2.5, 3.0, 2.5, 5.0],
        "Customer ID": ["C001","C001","C001","C002","C002","C002"],
        "Country":     ["UK","UK","UK","Germany","Germany","Germany"],
        "Revenue":     [25.0, 25.0, 7.5, 24.0, 5.0, 30.0],
        "is_weekend":  [False, False, False, False, False, True],
    })


class TestRFM:
    def test_rfm_shape(self, sample_df):
        rfm = compute_customer_rfm(sample_df)
        assert rfm.shape[0] == 2  # 2 customers

    def test_rfm_columns(self, sample_df):
        rfm = compute_customer_rfm(sample_df)
        for col in ["Customer ID","recency_days","frequency","monetary","avg_order_value"]:
            assert col in rfm.columns

    def test_frequency_correct(self, sample_df):
        rfm = compute_customer_rfm(sample_df)
        c001 = rfm[rfm["Customer ID"] == "C001"].iloc[0]
        assert c001["frequency"] == 2  # INV001, INV002

    def test_monetary_correct(self, sample_df):
        rfm = compute_customer_rfm(sample_df)
        c001 = rfm[rfm["Customer ID"] == "C001"].iloc[0]
        assert abs(c001["monetary"] - 57.5) < 0.01  # 25+25+7.5

    def test_recency_non_negative(self, sample_df):
        rfm = compute_customer_rfm(sample_df)
        assert (rfm["recency_days"] >= 0).all()

    def test_avg_order_value(self, sample_df):
        rfm = compute_customer_rfm(sample_df)
        c001 = rfm[rfm["Customer ID"] == "C001"].iloc[0]
        assert abs(c001["avg_order_value"] - 57.5/2) < 0.01


class TestCustomerStats:
    def test_stats_shape(self, sample_df):
        stats = compute_customer_stats(sample_df)
        assert stats.shape[0] == 2

    def test_stats_columns(self, sample_df):
        stats = compute_customer_stats(sample_df)
        for col in ["Customer ID","total_items","unique_products","avg_basket_size"]:
            assert col in stats.columns

    def test_total_items(self, sample_df):
        stats = compute_customer_stats(sample_df)
        c001 = stats[stats["Customer ID"] == "C001"].iloc[0]
        assert c001["total_items"] == 18  # 10+5+3

    def test_unique_products(self, sample_df):
        stats = compute_customer_stats(sample_df)
        c001 = stats[stats["Customer ID"] == "C001"].iloc[0]
        assert c001["unique_products"] == 2  # A001, A002

    def test_weekend_rate_range(self, sample_df):
        stats = compute_customer_stats(sample_df)
        assert (stats["weekend_purchase_rate"] >= 0).all()
        assert (stats["weekend_purchase_rate"] <= 1).all()


class TestSegments:
    def test_segments_created(self, sample_df):
        rfm = compute_customer_rfm(sample_df)
        rfm = compute_rfm_segments(rfm)
        assert "segment" in rfm.columns

    def test_valid_segments(self, sample_df):
        rfm = compute_customer_rfm(sample_df)
        rfm = compute_rfm_segments(rfm)
        valid = {"Champions","Loyal","At Risk","Lost"}
        assert set(rfm["segment"].unique()).issubset(valid)

    def test_rfm_score_range(self, sample_df):
        rfm = compute_customer_rfm(sample_df)
        rfm = compute_rfm_segments(rfm)
        assert rfm["rfm_score"].between(3, 15).all()
