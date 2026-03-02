# Chandler Drake Talley — Data Science Portfolio

**Senior data scientist who ships production ML and turns results into business impact.**  
This repo is the evidence: pipelines that avoid leakage, stratified evaluation, SHAP interpretability, and stakeholder-ready recommendations across credit risk, churn, clinical triage, time series, and quant finance. **Everything lives here—no external portfolio site.**

---

## Quick links

| | |
|---|---|
| **This repo** | [GitHub](https://github.com/cdtalley/Data-Science-Portfolio) — code, notebooks, and setup |
| **View notebooks in browser** | [nbviewer](https://nbviewer.jupyter.org/github/cdtalley/Data-Science-Portfolio/tree/main/) — no install required |
| **LinkedIn** (https://www.linkedin.com/in/drake-talley/) | *[Add your LinkedIn profile URL]* |

---

## Why this portfolio stands out

Every notebook follows the workflow a senior data scientist uses in production. These were built end-to-end by hand starting in 2020—before LLMs and GenAI—and many were developed alongside a senior data scientist with 30+ years of experience, with his review and approval. They have been refined since with pipelines, SHAP, and current best practices. Recruiters and hiring managers can run the code, see the same metrics, and judge rigor—not just slides.

| Practice | Implementation |
|---|---|
| **Reproducibility** | `set_seed(42)`, consistent `random_state` across all models and splits |
| **No data leakage** | `sklearn.Pipeline` wraps scaler → selector → estimator; fit only on train folds |
| **Stratified splits** | `stratify=y` on every `train_test_split` and `StratifiedKFold` CV |
| **Multi-metric eval** | `cross_validate` with accuracy, precision, recall, F1, ROC-AUC simultaneously |
| **Interpretability** | SHAP TreeExplainer/KernelExplainer on every supervised project |
| **Business framing** | Each project ends with stakeholder-ready recommendations and cost analysis |
| **Threshold tuning** | Precision-recall tradeoff analysis with cost-sensitive optimization |
| **Time series** | Daily trend, rolling means, day-of-week/hour seasonality, temporal heatmaps; `TimeSeriesSplit` and lag features where appropriate |

---

## Projects at a glance

| Project | One-line impact |
|--------|------------------|
| **Corporate Bankruptcy** | Predict bankruptcy risk from 96 financials → credit exposure mitigation. |
| **Telecom Churn** | Predict churn → cost-sensitive retention (e.g. $500 acquisition vs $75 offer). |
| **Heart Disease** | Predict from biometrics → clinical triage with sensitivity-first metrics and calibration. |
| **NJ Transit + Amtrak** | Predict delays from 98K trips → schedule padding, crew allocation, passenger alerts. |
| **NYC Bus** | Cluster 6.7M bus records → segment-specific scheduling and anomaly detection. |
| **Jane Street** | Predict profitable trades → position sizing and transaction-cost-aware signals. |
| **RAG + LLM** | FAISS + Sentence Transformers + local LLM → no-api-key demo. |

*Full project descriptions and tech stacks are in [Projects](#projects) below.*

---

## For recruiters & hiring managers

**5-minute tour:**  
1. Open [Modern_Classification_Workflow_Bankruptcy.ipynb](Modern_Classification_Workflow_Bankruptcy.ipynb) — end-to-end pipeline, CV, SHAP, and “why” in one place.  
2. Run `streamlit run app.py` — interactive EDA, model comparison, SHAP, and live predictions.  
3. Skim [docs/BEST_PRACTICES.md](docs/BEST_PRACTICES.md) — explains reproducibility, pipelines, and interpretability choices.

**Skills demonstrated (mapping to typical job descriptions):**

- **ML engineering:** `sklearn.Pipeline`, `ColumnTransformer`, stratified CV, hyperparameter tuning (GridSearchCV / RandomizedSearchCV).  
- **Model deployment:** FastAPI serving endpoint with `/predict`, `/predict/batch`, `/explain` (SHAP), Dockerfile, and docker-compose.  
- **Interpretability & compliance:** SHAP (TreeExplainer/KernelExplainer), feature importance, threshold tuning for cost-sensitive decisions.  
- **Evaluation:** Multi-metric reporting (precision, recall, F1, ROC-AUC), calibration for probability outputs, time-series-aware splits.  
- **Data engineering:** DuckDB-backed SQL analytics alongside pandas; production-pattern for warehouse-style EDA.  
- **Testing:** pytest suite for utilities, data loaders, API endpoints, and training pipeline.  
- **Production hygiene:** Reproducible seeds, locked envs (`pyproject.toml` / `requirements.txt`), smoke tests, Docker.  
- **Business communication:** Each project ends with stakeholder recommendations, quantified ROI, and cost/benefit framing.

---

## Model Serving API (FastAPI)

The Telecom Churn model is deployed as a production-style REST API with cost-sensitive predictions and SHAP explanations.

```bash
# Train the model (saves artifacts to artifacts/)
python -m api.train

# Start the API
uvicorn api.serve:app --reload

# Or run everything in Docker
docker-compose up
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/model/info` | Model metadata, metrics, feature names, cost assumptions |
| `POST` | `/predict` | Single customer churn prediction with risk tier and action |
| `POST` | `/predict/batch` | Batch predictions with summary statistics |
| `POST` | `/explain` | SHAP-based explanation of top churn drivers |

**Example:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Male","SeniorCitizen":0,"Partner":"Yes","Dependents":"No",
       "tenure":12,"PhoneService":"Yes","MultipleLines":"No",
       "InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"Yes",
       "DeviceProtection":"No","TechSupport":"No","StreamingTV":"No",
       "StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes",
       "PaymentMethod":"Electronic check","MonthlyCharges":70.35}'
```

---

## Interactive Dashboard

```bash
streamlit run app.py
```

Features per project:
- **EDA tab** — target distributions, feature correlations, interactive drill-downs
- **Model comparison** — train multiple models with sklearn Pipelines and stratified 5-fold CV
- **SHAP tab** — TreeExplainer feature importance plots
- **Live prediction** — adjust feature sliders and see real-time predictions with confidence scores

---

## Setup

1. **Virtual environment (recommended)**  
   Use the project venv so notebooks run with a consistent interpreter:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements-core.txt   # notebooks: pandas, sklearn, xgboost, jupyter, etc.
   pip install -r requirements.txt        # full stack: + tensorflow, RAG (streamlit, torch, transformers)
   ```
   In Cursor/VS Code: use the project venv as the notebook kernel so `numpy` and `portfolio_utils` are available: **Kernel → Select Kernel → Python Environments → `./venv/Scripts/python.exe`** (or **"Python (Data Science Portfolio)"** if it appears). This repo's `.vscode/settings.json` sets that as the default. If you see `ModuleNotFoundError: No module named 'numpy'`, the kernel is using a different Python — run `.\venv\Scripts\python.exe scripts/install_deps.py` then switch the kernel to the venv. Core deps are in `requirements-core.txt`; `requirements.txt` adds TensorFlow and the RAG app deps (install when no other process is using the venv to avoid file locks).

   Alternatively, without a venv: `pip install -r requirements.txt`. Or [uv](https://docs.astral.sh/uv/): `uv sync` (optional: `--extra rag --extra shap`).

2. **Windows: console warning / timeouts**  
   If you see `RuntimeWarning: Proactor event loop does not implement add_reader` or notebooks hang/timeout when running with `nbconvert --execute`, use the helper script so the correct asyncio policy is set before Jupyter starts:
   ```powershell
   python scripts/run_nbconvert.py --execute --inplace .\path\to\notebook.ipynb --ExecutePreprocessor.timeout=1200
   ```
   For interactive Jupyter, the warning is harmless; the kernel uses a fallback. If a cell runs for 20+ minutes with no output, stop the kernel and run with the script above or increase the timeout.

3. **Download datasets (Kaggle API)**
   ```bash
   python setup_data.py --no-jane-street
   ```
   Requires a [Kaggle API token](https://www.kaggle.com/settings) in `~/.kaggle/kaggle.json`. Omit `--no-jane-street` to include the large competition dataset.

4. **Run notebooks** — `jupyter notebook` or open in VS Code/Cursor. Select kernel **"Python (Data Science Portfolio)"**. Data loads via `portfolio_utils.data_loader` with Colab fallback.

5. **Launch dashboard** — `streamlit run app.py`

---

## Projects

### Corporate Bankruptcy Prediction
**Business problem:** Predict which companies are at risk of bankruptcy using 96 financial indicators (Taiwan Economic Journal, 1999–2009). Enables creditors to mitigate exposure before default.

**Senior analysis includes:** Pipeline (StandardScaler → SelectKBest → XGBoost), stratified 5-fold CV, precision-recall threshold tuning for cost-asymmetric decisions, SHAP feature importance, and recommendations for the credit risk team.

Tech: XGBoost, Random Forest, Gradient Boosting, Logistic Regression, SMOTE, SelectKBest, PCA, SHAP

[View notebook](Corporate_Bankruptcy_Prediction.ipynb)

---

### Telecom Customer Churn (IBM Watson Analytics)
**Business problem:** Predict which customers will churn so the retention team can intervene proactively. Retaining a customer is 5–25× cheaper than acquiring a new one.

**Senior analysis includes:** Pipeline with class-weighted models, cost-sensitive threshold optimization ($500 acquisition cost vs. $75 retention offer), SHAP-driven intervention recommendations, and A/B testing strategy.

Tech: XGBoost, Random Forest, Gradient Boosting, Logistic Regression, SVM, SMOTE, GridSearchCV, SHAP

[View notebook](Supervised_Learning_Capstone-Predicting_Telecom_Customer_Churn_(IBM_Watson_Analytics).ipynb)

---

### Heart Disease Prediction (Clinical Decision Support)
**Business problem:** Predict heart disease from patient biometrics for clinical triage. Missing a case (false negative) can be fatal — sensitivity is the priority metric.

**Senior analysis includes:** Pipeline-based modeling, calibration curve analysis (critical for clinical probability estimates), SHAP for clinical feature importance, and risk-stratified care pathway recommendations.

Tech: Random Forest, Gradient Boosting, SVM, Logistic Regression, GridSearchCV, SHAP, calibration analysis

[View notebook](Supervised_Learning_Heart_Disease_Prediction_using_Patient_Biometric_Data.ipynb)

---

### NJ Transit + Amtrak Rail Delay Prediction
**Business problem:** Predict train delays using supervised, unsupervised, and deep learning on 98K NEC rail trips. Enables proactive passenger notification and resource allocation.

**Senior analysis includes:** **Time series analysis** (daily delay trend, 7-day rolling mean, day-of-week and hour-of-day seasonality, day×hour heatmap), pipeline with SelectKBest, multi-metric stratified CV, SHAP for operational insight, and recommendations for schedule padding, crew allocation, and real-time passenger alerts.

Tech: Time series (pandas, seaborn), Decision Tree, Random Forest, Gradient Boosting, KNN, SVM, KMeans, DBSCAN, t-SNE, TensorFlow, SHAP

[View notebook](NJ_Transit_%2B_Amtrak_(NEC)_Rail_Performance_Business_Solution.ipynb)

---

### NYC Bus Clustering (Unsupervised)
**Business problem:** Segment 6.7M MTA bus location records into operational clusters for route optimization, service planning, and anomaly detection.

**Senior analysis includes:** Systematic k-selection with silhouette analysis (not just elbow method), per-cluster silhouette plots, PCA-space visualization with variance explained, and operational recommendations for segment-specific scheduling.

Tech: KMeans, DBSCAN, PCA, t-SNE, silhouette analysis

[View notebook](Unsupervised_Learning_Capstone_New_York_City_Bus_Data.ipynb)

---

### Jane Street Market Prediction (Quantitative Finance)
**Business problem:** Predict profitable trades from anonymized financial features. Precision on trade signals directly impacts P&L.

**Senior analysis includes:** Pipeline (Imputer → Scaler → XGBoost), stratified CV, SHAP for risk management transparency, and recommendations for time-series validation, position sizing, and transaction cost modeling.

Tech: XGBoost, RandomizedSearchCV, SHAP

[View notebook](Jane_Street_Market_Prediction_XGBoost_and_Hyperparameter_Tuning.ipynb)

---

### RAG + LLM Demo (Streamlit)
Retrieval-augmented generation with FAISS vector search, Sentence Transformers, and local GPT-2. No API keys required.

```bash
streamlit run Streamlit_Langchain_RAG_LLM.py
```

Tech: Streamlit, FAISS, Sentence Transformers, GPT-2, PyTorch

---

### Modern Classification Workflow (Reference)
A clean reference notebook demonstrating the full senior workflow end-to-end: `set_seed` → Pipeline → stratified CV → multi-metric evaluation → SHAP.

[View notebook](Modern_Classification_Workflow_Bankruptcy.ipynb) · [Best Practices Guide](docs/BEST_PRACTICES.md)

---

## Testing

```bash
pip install pytest httpx duckdb
pytest tests/ -v
```

| Test file | Covers |
|-----------|--------|
| `test_ml_utils.py` | Seed reproducibility, pipeline construction, SHAP fallback |
| `test_data_loader.py` | Data directory config, CSV discovery, Kaggle slug validation |
| `test_train.py` | Preprocessing (encode, drop, target), cost-sensitive threshold optimization |
| `test_api.py` | FastAPI endpoints: health, model info, predict, batch, explain |
| `test_db_utils.py` | DuckDB loader, SQL queries, context manager, pre-built analytics queries |

---

## Repo structure

```
Data-Science-Portfolio/
├── api/                    # FastAPI model serving
│   ├── train.py            #   Train pipeline, serialize artifacts
│   ├── serve.py            #   REST API (predict, explain, batch)
│   └── schemas.py          #   Pydantic request/response models
├── portfolio_utils/        # Shared Python package
│   ├── data_loader.py      #   Kaggle-backed dataset loaders
│   ├── ml_utils.py         #   Seeds, pipelines, SHAP helpers
│   └── db_utils.py         #   DuckDB SQL analytics layer
├── tests/                  # pytest suite
├── docs/                   # Best practices guide
├── scripts/                # Notebook automation (smoke tests, patching)
├── advanced_visualization/ # Dash geospatial dashboard
├── artifacts/              # Trained model + metadata (after api.train)
├── data/                   # Datasets (after setup_data.py)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── requirements-core.txt
├── requirements-api.txt
├── app.py                  # Streamlit portfolio dashboard
├── setup_data.py           # Download all datasets
└── *.ipynb                 # Project notebooks
```
