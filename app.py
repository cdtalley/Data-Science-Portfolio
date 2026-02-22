"""
Portfolio Dashboard — Interactive showcase of all data science projects.
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from portfolio_utils import set_seed, load_bankruptcy, load_telecom_churn, load_heart

set_seed(42)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Chandler Talley — DS Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("📊 Portfolio")
st.sidebar.markdown("**Chandler Drake Talley**")
st.sidebar.markdown("[GitHub](https://github.com/cdtalley/Data-Science-Portfolio) · [Website](https://chandlerdraketalley.com/portfolio/)")
st.sidebar.divider()

page = st.sidebar.radio(
    "Select project",
    [
        "🏠 Overview",
        "🏦 Bankruptcy Prediction",
        "📞 Telecom Churn",
        "❤️ Heart Disease",
    ],
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading data…")
def get_bankruptcy():
    return load_bankruptcy()


@st.cache_data(show_spinner="Loading data…")
def get_churn():
    return load_telecom_churn()


@st.cache_data(show_spinner="Loading data…")
def get_heart():
    return load_heart()


def train_pipeline(X, y, estimator_cls, estimator_kwargs, scale=True, select_k=None):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import train_test_split, cross_validate

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    if select_k and select_k < X.shape[1]:
        steps.append(("selector", SelectKBest(f_classif, k=select_k)))
    steps.append(("estimator", estimator_cls(**estimator_kwargs)))
    pipe = Pipeline(steps)

    scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    cv = cross_validate(pipe, X_train, y_train, cv=5, scoring=scoring, n_jobs=-1)
    pipe.fit(X_train, y_train)
    return pipe, cv, X_train, X_test, y_train, y_test


def show_cv_metrics(cv):
    rows = []
    for metric in ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]:
        key = f"test_{metric}"
        if key in cv:
            rows.append({
                "Metric": metric.replace("_", " ").title(),
                "Mean": round(cv[key].mean(), 4),
                "Std": round(cv[key].std(), 4),
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def show_test_metrics(pipe, X_test, y_test, labels=None):
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    st.dataframe(
        pd.DataFrame(report).T.round(3),
        use_container_width=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(4, 3))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=labels or ["0", "1"],
                    yticklabels=labels or ["0", "1"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with col2:
        if hasattr(pipe, "predict_proba"):
            try:
                auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
                st.metric("ROC-AUC", round(auc, 4))
            except Exception:
                pass


def show_shap(pipe, X_train, feature_names):
    try:
        import shap
    except ImportError:
        st.info("Install `shap` for interpretability plots: `pip install shap`")
        return
    est = pipe.named_steps["estimator"]
    if "selector" in pipe.named_steps:
        mask = pipe.named_steps["selector"].get_support()
        X_t = pipe["selector"].transform(pipe["scaler"].transform(X_train)) if "scaler" in pipe.named_steps else pipe["selector"].transform(X_train)
        names = np.array(feature_names)[mask].tolist()
    elif "scaler" in pipe.named_steps:
        X_t = pipe["scaler"].transform(X_train)
        names = feature_names
    else:
        X_t = X_train.values if hasattr(X_train, "values") else X_train
        names = feature_names
    sample = X_t[:min(300, len(X_t))]
    try:
        explainer = shap.TreeExplainer(est, sample)
        sv = explainer.shap_values(sample)
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(sv, sample, feature_names=names, max_display=12, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP unavailable for this model: {e}")


def show_live_prediction(pipe, feature_names, X_sample, labels):
    st.subheader("🔮 Live prediction")
    st.caption("Adjust feature values and see the model's prediction in real time.")
    cols = st.columns(3)
    inputs = {}
    for i, feat in enumerate(feature_names[:12]):
        with cols[i % 3]:
            mn, mx = float(X_sample[feat].min()), float(X_sample[feat].max())
            med = float(X_sample[feat].median())
            inputs[feat] = st.slider(feat[:40], mn, mx, med, key=f"pred_{feat}")
    for feat in feature_names[12:]:
        inputs[feat] = float(X_sample[feat].median())
    row = pd.DataFrame([inputs])[feature_names]
    pred = pipe.predict(row)[0]
    proba = pipe.predict_proba(row)[0] if hasattr(pipe, "predict_proba") else None
    label = labels[int(pred)] if labels else str(pred)
    col1, col2 = st.columns(2)
    col1.metric("Prediction", label)
    if proba is not None:
        col2.metric("Confidence", f"{proba.max():.1%}")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_overview():
    st.title("📊 Data Science Portfolio")
    st.markdown("""
    **Chandler Drake Talley** — Senior Data Scientist

    Interactive showcase of classification, clustering, time-series, and RAG/LLM projects.
    Select a project from the sidebar to explore **interactive EDA**, **model comparisons**,
    **SHAP interpretability**, and **live predictions**.

    ---
    """)
    st.subheader("Projects")
    projects = [
        ("🏦 Corporate Bankruptcy Prediction", "XGBoost, RF, Gradient Boosting, PCA, SMOTE", "96 financial features, 6.8K companies"),
        ("📞 Telecom Customer Churn", "Logistic Regression, Gradient Boosting, KNN, SVM, RF", "IBM Watson Analytics, 7K customers"),
        ("❤️ Heart Disease Prediction", "Decision Tree, Random Forest, SVM, TensorFlow", "UCI biometric data, 1K patients"),
        ("🚂 NJ Transit Rail Delays", "Supervised + Unsupervised + Deep Learning", "98K train trips, clustering + classification"),
        ("🚌 NYC Bus Clustering", "PCA, t-SNE, KMeans, DBSCAN", "6.7M bus location records"),
        ("💹 Jane Street Market Prediction", "XGBoost, RandomizedSearchCV, GPU", "Large-scale financial trading data"),
        ("🤖 RAG + LLM Demo", "FAISS, Sentence Transformers, GPT-2, Streamlit", "Retrieval-augmented generation"),
    ]
    for name, tech, data in projects:
        with st.expander(name):
            st.markdown(f"**Tech:** {tech}")
            st.markdown(f"**Data:** {data}")

    st.divider()
    st.subheader("Best practices applied")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔁 Reproducibility", "set_seed(42)")
    c2.metric("🔗 Pipelines", "sklearn Pipeline")
    c3.metric("📊 Metrics", "Acc / F1 / AUC")
    c4.metric("🔍 Interpretability", "SHAP")


def page_bankruptcy():
    st.title("🏦 Corporate Bankruptcy Prediction")
    st.markdown("""
    **Business problem:** Predict which companies are at risk of bankruptcy using 96 financial
    indicators from the Taiwan Economic Journal (1999–2009). Early detection helps creditors
    mitigate risk and avoid costly transactions.

    **Approach:** Pipeline (StandardScaler → SelectKBest → Classifier), stratified 5-fold CV,
    multiple metrics, SHAP interpretability, live prediction.
    """)

    df = get_bankruptcy()
    y = df["Bankrupt?"]
    X = df.drop(columns=["Bankrupt?"]).select_dtypes(include=[np.number])
    X = X.loc[:, X.nunique() > 1]
    feature_names = list(X.columns)

    tab_eda, tab_model, tab_shap, tab_predict = st.tabs(
        ["📈 EDA", "🤖 Model comparison", "🔍 SHAP", "🔮 Predict"]
    )

    with tab_eda:
        st.subheader("Target distribution")
        fig, ax = plt.subplots(figsize=(4, 3))
        y.value_counts().plot.bar(ax=ax, color=["#2196F3", "#F44336"])
        ax.set_xticklabels(["Not bankrupt", "Bankrupt"], rotation=0)
        ax.set_ylabel("Count")
        st.pyplot(fig)

        st.subheader("Feature distributions (top correlated with target)")
        corr = X.corrwith(y).abs().sort_values(ascending=False)
        top_feats = corr.head(6).index.tolist()
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        for ax, feat in zip(axes.flat, top_feats):
            for val, color, label in [(0, "#2196F3", "OK"), (1, "#F44336", "Bankrupt")]:
                subset = X.loc[y == val, feat]
                ax.hist(subset, bins=30, alpha=0.6, color=color, label=label)
            ax.set_title(feat[:30], fontsize=9)
            ax.legend(fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Correlation heatmap (top 15 features)")
        top15 = corr.head(15).index.tolist()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(X[top15].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax, annot_kws={"size": 7})
        plt.tight_layout()
        st.pyplot(fig)

    with tab_model:
        st.subheader("Model comparison (Pipeline + 5-fold stratified CV)")
        import xgboost as xgb
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression

        models = {
            "XGBoost": (xgb.XGBClassifier, {"random_state": 42}),
            "Random Forest": (RandomForestClassifier, {"n_estimators": 100, "random_state": 42}),
            "Gradient Boosting": (GradientBoostingClassifier, {"random_state": 42}),
            "Logistic Regression": (LogisticRegression, {"max_iter": 1000, "random_state": 42}),
        }
        selected = st.multiselect("Select models to compare", list(models.keys()), default=["XGBoost", "Random Forest"])
        k = st.slider("SelectKBest k (feature count)", 5, min(60, X.shape[1]), 30)

        if selected and st.button("Train & compare", type="primary"):
            results = {}
            best_pipe = None
            best_f1 = -1
            for name in selected:
                cls, kwargs = models[name]
                with st.spinner(f"Training {name}…"):
                    pipe, cv, X_train, X_test, y_train, y_test = train_pipeline(
                        X, y, cls, kwargs, scale=True, select_k=k
                    )
                f1 = cv["test_f1_weighted"].mean()
                results[name] = {
                    "Accuracy": cv["test_accuracy"].mean(),
                    "Precision": cv["test_precision_weighted"].mean(),
                    "Recall": cv["test_recall_weighted"].mean(),
                    "F1": f1,
                }
                if f1 > best_f1:
                    best_f1 = f1
                    best_pipe = pipe
                    best_data = (X_train, X_test, y_train, y_test)

            st.dataframe(pd.DataFrame(results).T.round(4), use_container_width=True)

            fig, ax = plt.subplots(figsize=(8, 4))
            pd.DataFrame(results).T.plot.bar(ax=ax, rot=0)
            ax.set_ylim(0.8, 1.0)
            ax.set_ylabel("Score")
            ax.legend(loc="lower right")
            plt.tight_layout()
            st.pyplot(fig)

            st.subheader(f"Best model holdout metrics")
            show_test_metrics(best_pipe, best_data[1], best_data[3], ["Not bankrupt", "Bankrupt"])
            st.session_state["bankruptcy_pipe"] = best_pipe
            st.session_state["bankruptcy_data"] = best_data
            st.session_state["bankruptcy_features"] = feature_names

    with tab_shap:
        st.subheader("SHAP feature importance")
        if "bankruptcy_pipe" in st.session_state:
            show_shap(
                st.session_state["bankruptcy_pipe"],
                st.session_state["bankruptcy_data"][0],
                st.session_state["bankruptcy_features"],
            )
        else:
            st.info("Train a model in the **Model comparison** tab first.")

    with tab_predict:
        if "bankruptcy_pipe" in st.session_state:
            show_live_prediction(
                st.session_state["bankruptcy_pipe"],
                st.session_state["bankruptcy_features"],
                X,
                ["Not bankrupt", "Bankrupt"],
            )
        else:
            st.info("Train a model in the **Model comparison** tab first.")


def page_churn():
    st.title("📞 Telecom Customer Churn")
    st.markdown("""
    **Business problem:** Predict which customers will leave (churn) so the retention team can
    intervene. Retaining an existing customer is 5–25× cheaper than acquiring a new one.

    **Approach:** Pipeline (StandardScaler → Classifier), stratified 5-fold CV, multiple metrics,
    SHAP interpretability, live prediction.
    """)

    df = get_churn()
    drop_cols = [c for c in ["customerID", "CustomerID", "TotalCharges", "Total Charges",
                             "Churn Label", "Churn Value", "Churn Score", "CLTV",
                             "Churn Reason", "Count", "Country", "State", "City",
                             "Zip Code", "Lat Long", "Latitude", "Longitude"] if c in df.columns]
    target_col = "Churn"
    y_raw = df[target_col]
    X_raw = df.drop(columns=[target_col] + drop_cols, errors="ignore")

    cat_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()
    X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True).astype(float)
    y = y_raw.map({"Yes": 1, "No": 0}) if y_raw.dtype == "object" else y_raw
    feature_names = list(X_encoded.columns)

    tab_eda, tab_model, tab_shap, tab_predict = st.tabs(
        ["📈 EDA", "🤖 Model comparison", "🔍 SHAP", "🔮 Predict"]
    )

    with tab_eda:
        st.subheader("Target distribution")
        fig, ax = plt.subplots(figsize=(4, 3))
        y.value_counts().plot.bar(ax=ax, color=["#2196F3", "#F44336"])
        ax.set_xticklabels(["Stayed", "Churned"], rotation=0)
        ax.set_ylabel("Count")
        st.pyplot(fig)

        st.subheader("Churn rate by feature")
        plot_col = st.selectbox("Feature", [c for c in X_raw.columns if X_raw[c].nunique() < 10])
        if plot_col:
            fig, ax = plt.subplots(figsize=(6, 3))
            ct = pd.crosstab(X_raw[plot_col], y_raw, normalize="index")
            ct.plot.bar(stacked=True, ax=ax, color=["#2196F3", "#F44336"])
            ax.set_ylabel("Proportion")
            ax.legend(["Stayed", "Churned"])
            plt.tight_layout()
            st.pyplot(fig)

    with tab_model:
        st.subheader("Model comparison (Pipeline + 5-fold stratified CV)")
        import xgboost as xgb_mod
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression

        models = {
            "XGBoost": (xgb_mod.XGBClassifier, {"random_state": 42}),
            "Random Forest": (RandomForestClassifier, {"n_estimators": 100, "random_state": 42}),
            "Gradient Boosting": (GradientBoostingClassifier, {"random_state": 42}),
            "Logistic Regression": (LogisticRegression, {"max_iter": 1000, "random_state": 42}),
        }
        selected = st.multiselect("Select models", list(models.keys()), default=["XGBoost", "Gradient Boosting"], key="churn_models")

        if selected and st.button("Train & compare", type="primary", key="churn_train"):
            results = {}
            best_pipe = None
            best_f1 = -1
            for name in selected:
                cls, kwargs = models[name]
                with st.spinner(f"Training {name}…"):
                    pipe, cv, X_train, X_test, y_train, y_test = train_pipeline(
                        X_encoded, y, cls, kwargs, scale=True
                    )
                f1 = cv["test_f1_weighted"].mean()
                results[name] = {
                    "Accuracy": cv["test_accuracy"].mean(),
                    "Precision": cv["test_precision_weighted"].mean(),
                    "Recall": cv["test_recall_weighted"].mean(),
                    "F1": f1,
                }
                if f1 > best_f1:
                    best_f1 = f1
                    best_pipe = pipe
                    best_data = (X_train, X_test, y_train, y_test)

            st.dataframe(pd.DataFrame(results).T.round(4), use_container_width=True)
            st.subheader("Best model holdout metrics")
            show_test_metrics(best_pipe, best_data[1], best_data[3], ["Stayed", "Churned"])
            st.session_state["churn_pipe"] = best_pipe
            st.session_state["churn_data"] = best_data
            st.session_state["churn_features"] = feature_names

    with tab_shap:
        st.subheader("SHAP feature importance")
        if "churn_pipe" in st.session_state:
            show_shap(
                st.session_state["churn_pipe"],
                st.session_state["churn_data"][0],
                st.session_state["churn_features"],
            )
        else:
            st.info("Train a model in the **Model comparison** tab first.")

    with tab_predict:
        if "churn_pipe" in st.session_state:
            show_live_prediction(
                st.session_state["churn_pipe"],
                st.session_state["churn_features"],
                X_encoded,
                ["Stayed", "Churned"],
            )
        else:
            st.info("Train a model in the **Model comparison** tab first.")


def page_heart():
    st.title("❤️ Heart Disease Prediction")
    st.markdown("""
    **Business problem:** Predict heart disease from patient biometrics so clinicians can
    prioritize high-risk patients for early intervention.

    **Approach:** Pipeline (StandardScaler → Classifier), stratified 5-fold CV, multiple metrics,
    SHAP interpretability, live prediction.
    """)

    df = get_heart()
    target_col = [c for c in df.columns if c.lower() == "target" or c.lower() == "output"][0] if any(c.lower() in ("target", "output") for c in df.columns) else df.columns[-1]
    y = df[target_col]
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    feature_names = list(X.columns)

    tab_eda, tab_model, tab_shap, tab_predict = st.tabs(
        ["📈 EDA", "🤖 Model comparison", "🔍 SHAP", "🔮 Predict"]
    )

    with tab_eda:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Target distribution")
            fig, ax = plt.subplots(figsize=(4, 3))
            y.value_counts().plot.bar(ax=ax, color=["#2196F3", "#F44336"])
            ax.set_xticklabels(["No disease", "Disease"], rotation=0)
            st.pyplot(fig)
        with c2:
            st.subheader("Age distribution by target")
            age_col = [c for c in X.columns if "age" in c.lower()]
            if age_col:
                fig, ax = plt.subplots(figsize=(4, 3))
                for val, color, label in [(0, "#2196F3", "No disease"), (1, "#F44336", "Disease")]:
                    ax.hist(X.loc[y == val, age_col[0]], bins=20, alpha=0.6, color=color, label=label)
                ax.legend()
                ax.set_xlabel("Age")
                st.pyplot(fig)

        st.subheader("Correlation heatmap")
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax, annot_kws={"size": 8})
        plt.tight_layout()
        st.pyplot(fig)

    with tab_model:
        st.subheader("Model comparison")
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression

        models = {
            "Random Forest": (RandomForestClassifier, {"n_estimators": 100, "random_state": 42}),
            "Gradient Boosting": (GradientBoostingClassifier, {"random_state": 42}),
            "Logistic Regression": (LogisticRegression, {"max_iter": 1000, "random_state": 42}),
        }
        selected = st.multiselect("Select models", list(models.keys()), default=["Random Forest", "Logistic Regression"], key="heart_models")

        if selected and st.button("Train & compare", type="primary", key="heart_train"):
            results = {}
            best_pipe, best_f1 = None, -1
            for name in selected:
                cls, kwargs = models[name]
                with st.spinner(f"Training {name}…"):
                    pipe, cv, X_train, X_test, y_train, y_test = train_pipeline(X, y, cls, kwargs)
                f1 = cv["test_f1_weighted"].mean()
                results[name] = {
                    "Accuracy": cv["test_accuracy"].mean(),
                    "Precision": cv["test_precision_weighted"].mean(),
                    "Recall": cv["test_recall_weighted"].mean(),
                    "F1": f1,
                }
                if f1 > best_f1:
                    best_f1, best_pipe = f1, pipe
                    best_data = (X_train, X_test, y_train, y_test)

            st.dataframe(pd.DataFrame(results).T.round(4), use_container_width=True)
            show_test_metrics(best_pipe, best_data[1], best_data[3], ["No disease", "Disease"])
            st.session_state["heart_pipe"] = best_pipe
            st.session_state["heart_data"] = best_data
            st.session_state["heart_features"] = feature_names

    with tab_shap:
        if "heart_pipe" in st.session_state:
            show_shap(st.session_state["heart_pipe"], st.session_state["heart_data"][0], st.session_state["heart_features"])
        else:
            st.info("Train a model first.")

    with tab_predict:
        if "heart_pipe" in st.session_state:
            show_live_prediction(st.session_state["heart_pipe"], st.session_state["heart_features"], X, ["No disease", "Disease"])
        else:
            st.info("Train a model first.")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if page == "🏠 Overview":
    page_overview()
elif page == "🏦 Bankruptcy Prediction":
    page_bankruptcy()
elif page == "📞 Telecom Churn":
    page_churn()
elif page == "❤️ Heart Disease":
    page_heart()
