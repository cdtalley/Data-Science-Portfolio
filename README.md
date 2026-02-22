# Chandler Drake Talley — Data Science Portfolio

Production-grade ML projects spanning credit risk, customer retention, clinical decision support, transit operations, and quantitative finance. Each project includes **exploratory analysis**, **pipeline-based modeling** (no data leakage), **multi-metric evaluation**, **SHAP interpretability**, and **business recommendations**.

- **Interactive dashboard:** `streamlit run app.py` — interactive EDA, model comparisons, SHAP, and live predictions.
- **View notebooks on GitHub** or [nbviewer](https://nbviewer.jupyter.org/github/cdtalley/Data-Science-Portfolio/tree/main/).
- **Site:** [chandlerdraketalley.com/portfolio](https://chandlerdraketalley.com/portfolio/)

---

## What makes this portfolio different

Every notebook follows the workflow a senior data scientist uses in production:

| Practice | Implementation |
|---|---|
| **Reproducibility** | `set_seed(42)`, consistent `random_state` across all models and splits |
| **No data leakage** | `sklearn.Pipeline` wraps scaler → selector → estimator; fit only on train folds |
| **Stratified splits** | `stratify=y` on every `train_test_split` and `StratifiedKFold` CV |
| **Multi-metric eval** | `cross_validate` with accuracy, precision, recall, F1, ROC-AUC simultaneously |
| **Interpretability** | SHAP TreeExplainer/KernelExplainer on every supervised project |
| **Business framing** | Each project ends with stakeholder-ready recommendations and cost analysis |
| **Threshold tuning** | Precision-recall tradeoff analysis with cost-sensitive optimization |

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
   In Cursor/VS Code: choose kernel **"Python (Data Science Portfolio)"** or select `venv\Scripts\python.exe` as the interpreter. Core deps are in `requirements-core.txt`; `requirements.txt` adds TensorFlow and the RAG app deps (install when no other process is using the venv to avoid file locks).

   Alternatively, without a venv: `pip install -r requirements.txt`. Or [uv](https://docs.astral.sh/uv/): `uv sync` (optional: `--extra rag --extra shap`).

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

**Senior analysis includes:** Pipeline with SelectKBest, multi-metric stratified CV, SHAP for operational insight, and recommendations for schedule padding, crew allocation, and real-time passenger alerts.

Tech: Decision Tree, Random Forest, Gradient Boosting, KNN, SVM, KMeans, DBSCAN, t-SNE, TensorFlow, SHAP

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
