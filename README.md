# 🏪 Feature Store — UCI Online Retail II

![CI](https://github.com/jumma786/mlops-feature-store/actions/workflows/feature_store.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Parquet](https://img.shields.io/badge/Storage-Parquet-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Part of the MLOps Portfolio Series** — Project 6 of 10  
> Feature store built on UCI Online Retail II (1M+ rows) — computes RFM features, customer stats, and segment labels, stores as Parquet offline store, and serves via FastAPI.

---

## 📂 Project Resources

| Resource | Link |
|---|---|
| ⚙️ Feature Engineer | [src/features/engineer.py](src/features/engineer.py) |
| 🚀 Serving API | [src/serving/api.py](src/serving/api.py) |
| 📦 Data Loader | [src/data/loader.py](src/data/loader.py) |
| 🧪 Unit Tests | [tests/test_features.py](tests/test_features.py) |
| 🤖 CI/CD Workflow | [.github/workflows/feature_store.yml](.github/workflows/feature_store.yml) |
| 📋 Requirements | [requirements.txt](requirements.txt) |

---

## 🎯 What This Project Does

Solves the "feature inconsistency" problem — training and serving use different feature logic:

1. **Ingests UCI Online Retail II** — 1,067,371 transactions, cleaned to ~800K usable rows
2. **Computes 3 feature views** — RFM, customer stats, product preferences
3. **Stores as Parquet** — offline store with point-in-time correct snapshots
4. **Serves via FastAPI** — `/features/{customer_id}` endpoint for online inference
5. **Segments customers** — Champions, Loyal, At Risk, Lost using RFM quintiles

---

## 📊 Dataset

**UCI Online Retail II** — Real transactional data

| Property | Value |
|---|---|
| Source | UCI ML Repository via Kaggle |
| Raw rows | 1,067,371 |
| Clean rows | ~800,000 |
| Date range | Dec 2009 — Dec 2011 |
| Unique customers | ~5,877 |
| Countries | 43 |
| Total revenue | £9.7M+ |

---

## 🔧 Feature Views

### 1. customer_rfm
| Feature | Description |
|---|---|
| `recency_days` | Days since last purchase |
| `frequency` | Number of unique orders |
| `monetary` | Total spend |
| `avg_order_value` | monetary / frequency |
| `std_order_value` | Spend variability |
| `max_order_value` | Largest single order |
| `purchase_span_days` | Days between first and last purchase |
| `r_score`, `f_score`, `m_score` | Quintile scores 1-5 |
| `rfm_score` | Combined score (3-15) |
| `segment` | Champions / Loyal / At Risk / Lost |

### 2. customer_stats
| Feature | Description |
|---|---|
| `total_items` | Total units purchased |
| `unique_products` | Number of distinct products |
| `avg_basket_size` | Avg items per order |
| `avg_unit_price` | Average price per item |
| `unique_countries` | Countries ordered from |
| `weekend_purchase_rate` | % purchases on weekend |

---

## 🚀 Quick Start

```bash
git clone https://github.com/jumma786/mlops-feature-store.git
cd mlops-feature-store
pip install -r requirements.txt

# Download real data
kaggle datasets download -d mashlyn/online-retail-ii-uci -p data --unzip

# Run tests
make test

# Build feature store
make build-features

# Start serving API
make serve
# Open http://127.0.0.1:8001/docs
```

---

## 📈 Results — Real Data

| Metric | Value |
|---|---|
| Unique customers | 5,877 |
| Champions | ~22% |
| Loyal | ~25% |
| At Risk | ~27% |
| Lost | ~26% |
| Features computed | 19 |
| Avg monetary (Champions) | £3,200+ |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/features/{customer_id}` | Get all features for a customer |
| POST | `/features/batch` | Batch feature lookup |
| GET | `/segments` | Segment distribution |
| GET | `/top-customers` | Top N customers by metric |

---

## 🔗 MLOps Portfolio Series

| # | Project | Repo | Status |
|---|---|---|---|
| 1 | Multi-Model Tournament | [mlops-model-tournament](https://github.com/jumma786/mlops-model-tournament) | ✅ |
| 2 | Scheduled Retraining | [mlops-retraining-pipeline](https://github.com/jumma786/mlops-retraining-pipeline) | ✅ |
| 3 | Feature Engineering | [mlops-feature-pipeline](https://github.com/jumma786/mlops-feature-pipeline) | ✅ |
| 4 | Hyperparameter Tuning | [mlops-hyperparameter-tuning](https://github.com/jumma786/mlops-hyperparameter-tuning) | ✅ |
| 5 | Model Serving | [mlops-model-serving](https://github.com/jumma786/mlops-model-serving) | ✅ |
| **6** | **Feature Store** | [mlops-feature-store](https://github.com/jumma786/mlops-feature-store) | ✅ This repo |
| 7 | Model Monitoring | [mlops-model-monitoring](https://github.com/jumma786/mlops-model-monitoring) | ✅ |
| 8 | A/B Testing | [mlops-ab-testing](https://github.com/jumma786/mlops-ab-testing) | ✅ |
| 9 | Airflow Pipeline | [mlops-airflow-pipeline](https://github.com/jumma786/mlops-airflow-pipeline) | ✅ |
| 10 | Kubernetes Platform | [mlops-k8s-platform](https://github.com/jumma786/mlops-k8s-platform) | ✅ |

---

## 📝 Key MLOps Concepts Demonstrated

- **Feature store pattern** — offline store (Parquet) + online serving (FastAPI)
- **Point-in-time correctness** — snapshot_date prevents data leakage
- **RFM segmentation** — quintile-based customer scoring
- **Feature reuse** — same features used for training and inference
- **Offline/online consistency** — no feature skew between training and serving

---

## 👤 Author

**Jumma Mohammad Teli** — Data Analyst & ML Engineer  
📍 Birmingham, UK  
📧 [jummamohammad477@gmail.com](mailto:jummamohammad477@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/jumma-mohammad) | [GitHub](https://github.com/jumma786)

---

*Project 6 of 10 — MLOps Portfolio Series.*
