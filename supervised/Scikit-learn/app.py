"""
Streamlit dashboard for the diabetes Linear Regression model (train.py).

It loads model.pkl and lets you:
  - move sliders for a patient's measurements and see the live prediction
  - inspect how good the model is (actual vs predicted, residuals)
  - see what the model learned (coefficients)
  - explore the raw dataset

Run it:
    cd supervised/Scikit-learn
    ../../.venv/bin/streamlit run app.py
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).parent

st.set_page_config(page_title="Diabetes Regression", page_icon="🩺", layout="wide")

# Human-readable names for the dataset's 10 features
FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "bmi": "Body mass index (BMI)",
    "bp": "Average blood pressure",
    "s1": "s1 — total cholesterol",
    "s2": "s2 — LDL cholesterol",
    "s3": "s3 — HDL cholesterol",
    "s4": "s4 — cholesterol ratio",
    "s5": "s5 — triglycerides (log)",
    "s6": "s6 — blood sugar",
}


# @st.cache_data: run once, remember the result — the app reruns this file
# on every slider move, caching keeps it instant.
@st.cache_data
def load_data():
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = pd.Series(diabetes.target, name="progression")
    return X, y


@st.cache_resource
def load_model():
    with open(HERE / "model.pkl", "rb") as file:
        return pickle.load(file)


X, y = load_data()
model = load_model()

# Recreate the exact same split as train.py so evaluation is honest
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
y_pred_test = model.predict(X_test)

st.title("🩺 Diabetes Progression Predictor")
st.markdown(
    "A **LinearRegression** model (trained by `train.py`, loaded from `model.pkl`) "
    "predicting one-year disease progression from 10 standardized measurements "
    "of 442 real patients."
)

tab_predict, tab_load, tab_performance, tab_learned, tab_data = st.tabs(
    ["🔮 Predict", "📥 Load your data", "📊 Model performance",
     "🧠 What the model learned", "🗂 Data explorer"]
)

# =====================================================================
# TAB 1 — interactive prediction
# =====================================================================
with tab_predict:
    st.subheader("Set the patient's measurements")
    st.caption(
        "Features are standardized (0 = dataset average). "
        "Slider ranges match the real patients in the dataset."
    )

    cols = st.columns(2)
    patient = {}
    for i, feature in enumerate(X.columns):
        with cols[i % 2]:
            patient[feature] = st.slider(
                FEATURE_LABELS[feature],
                min_value=float(X[feature].min()),
                max_value=float(X[feature].max()),
                value=0.0,
                step=0.001,
                format="%.3f",
            )

    patient_df = pd.DataFrame([patient])
    prediction = float(model.predict(patient_df)[0])

    st.divider()
    left, right = st.columns([1, 2])

    with left:
        st.metric(
            "Predicted progression score",
            f"{prediction:.0f}",
            delta=f"{prediction - y.mean():+.0f} vs average patient",
        )
        st.caption(f"Dataset range: {y.min():.0f} – {y.max():.0f}, average {y.mean():.0f}")

    with right:
        # Where does this patient fall in the real distribution?
        fig, ax = plt.subplots(figsize=(7, 2.8))
        ax.hist(y, bins=30, color="#9ecae1", edgecolor="white")
        ax.axvline(prediction, color="#d62728", linewidth=2.5)
        ax.annotate(
            f"this patient: {prediction:.0f}",
            xy=(prediction, ax.get_ylim()[1] * 0.9),
            xytext=(5, 0), textcoords="offset points",
            color="#d62728", fontweight="bold",
        )
        ax.set_xlabel("progression score of the 442 real patients")
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        st.pyplot(fig, width="stretch")

# =====================================================================
# TAB 2 — load your own patients: file upload OR manual entry
# =====================================================================
def predictions_view(patients: pd.DataFrame, key: str):
    """Predict for a table of patients and show results + download button."""
    results = patients.copy()
    results["predicted_progression"] = model.predict(patients[list(X.columns)]).round(1)

    st.dataframe(results, width="stretch", height=250)

    c1, c2 = st.columns([2, 1])
    with c1:
        if len(results) > 1:
            fig, ax = plt.subplots(figsize=(7, 2.5))
            ax.hist(results["predicted_progression"], bins=min(20, len(results)),
                    color="#9ecae1", edgecolor="white")
            ax.axvline(y.mean(), color="gray", linestyle="--", label="dataset average")
            ax.set_xlabel("predicted progression score")
            ax.set_yticks([])
            ax.legend()
            st.pyplot(fig, width="stretch")
    with c2:
        st.metric("Patients", len(results))
        st.metric("Average prediction", f"{results['predicted_progression'].mean():.0f}")
        st.download_button(
            "⬇️ Download results as CSV",
            results.to_csv(index=False).encode(),
            file_name="predictions.csv",
            mime="text/csv",
            key=f"download_{key}",
        )


with tab_load:
    st.subheader("Bring your own patients")
    mode = st.radio(
        "How do you want to provide the data?",
        ["📄 Upload a file (CSV / Excel / JSON)", "⌨️ Enter patients one by one"],
        horizontal=True,
    )

    st.caption(
        "Required columns (same standardized units as the dataset): "
        + ", ".join(f"`{c}`" for c in X.columns)
    )

    # ---------------- file upload ----------------
    if mode.startswith("📄"):
        # A template the user can fill in — 3 real patients as examples
        template = X.head(3).round(4)
        st.download_button(
            "⬇️ Download a template CSV to fill in",
            template.to_csv(index=False).encode(),
            file_name="patients_template.csv",
            mime="text/csv",
        )

        uploaded = st.file_uploader(
            "Drop your file here",
            type=["csv", "xlsx", "xls", "json"],
            help="The file must contain the 10 feature columns listed above. "
                 "Extra columns are ignored.",
        )

        if uploaded is not None:
            try:
                name = uploaded.name.lower()
                if name.endswith(".csv"):
                    patients = pd.read_csv(uploaded)
                elif name.endswith((".xlsx", ".xls")):
                    patients = pd.read_excel(uploaded)
                else:
                    patients = pd.read_json(uploaded)
            except Exception as exc:
                st.error(f"Could not read the file: {exc}")
                st.stop()

            # normalize column names so "BMI " or "Age" still match
            patients.columns = patients.columns.str.strip().str.lower()

            missing = [c for c in X.columns if c not in patients.columns]
            if missing:
                st.error(
                    f"Missing required columns: {', '.join(missing)}. "
                    "Download the template above to see the expected format."
                )
                st.stop()

            # keep only model columns, force numeric, reject broken rows
            patients = patients[list(X.columns)].apply(pd.to_numeric, errors="coerce")
            bad_rows = patients.isna().any(axis=1)
            if bad_rows.any():
                st.warning(
                    f"Skipped {bad_rows.sum()} row(s) with missing or "
                    "non-numeric values."
                )
                patients = patients[~bad_rows]

            if patients.empty:
                st.error("No valid rows left to predict on.")
            else:
                st.success(f"Loaded **{uploaded.name}** — {len(patients)} valid patient(s)")
                predictions_view(patients, key="upload")

    # ---------------- manual one-by-one entry ----------------
    else:
        # session_state survives reruns — this is the app's "memory"
        if "manual_patients" not in st.session_state:
            st.session_state.manual_patients = []

        with st.form("add_patient", clear_on_submit=False):
            st.markdown("**New patient** (0.0 = dataset average for every field)")
            cols = st.columns(5)
            values = {}
            for i, feature in enumerate(X.columns):
                with cols[i % 5]:
                    values[feature] = st.number_input(
                        FEATURE_LABELS[feature],
                        min_value=float(X[feature].min() * 2),
                        max_value=float(X[feature].max() * 2),
                        value=0.0,
                        step=0.005,
                        format="%.3f",
                    )
            if st.form_submit_button("➕ Add patient"):
                st.session_state.manual_patients.append(values)

        if st.session_state.manual_patients:
            st.markdown(f"**Your list — {len(st.session_state.manual_patients)} patient(s)**")
            patients = pd.DataFrame(st.session_state.manual_patients)
            predictions_view(patients, key="manual")
            if st.button("🗑 Clear the list"):
                st.session_state.manual_patients = []
                st.rerun()
        else:
            st.info("Fill the form above and press **Add patient** — "
                    "each one is added to a list and predicted instantly.")


# =====================================================================
# TAB 3 — how good is the model?
# =====================================================================
with tab_performance:
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    m1, m2, m3 = st.columns(3)
    m1.metric("Test patients", len(y_test))
    m2.metric("MAE (avg. error)", f"{mae:.1f} points")
    m3.metric("R² score", f"{r2:.3f}")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Actual vs predicted** — perfect model = all dots on the line")
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(y_test, y_pred_test, alpha=0.6, color="#3182bd")
        lims = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
        ax.plot(lims, lims, "--", color="gray", label="perfect prediction")
        ax.set_xlabel("actual progression")
        ax.set_ylabel("predicted progression")
        ax.legend()
        st.pyplot(fig, width="stretch")

    with c2:
        st.markdown("**Residuals** — errors should hover randomly around 0")
        residuals = y_test - y_pred_test
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(y_pred_test, residuals, alpha=0.6, color="#e6550d")
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_xlabel("predicted progression")
        ax.set_ylabel("error (actual − predicted)")
        st.pyplot(fig, width="stretch")

    st.info(
        f"Read it like this: on average the model is off by **~{mae:.0f} points** "
        f"and explains **{r2:.0%}** of the variation between patients. "
        "Linear regression is a simple model — a good honest baseline."
    )

# =====================================================================
# TAB 4 — coefficients: the model's learned recipe
# =====================================================================
with tab_learned:
    st.markdown(
        "LinearRegression literally learns one number per feature: \n"
        "`prediction = intercept + coef₁·feature₁ + coef₂·feature₂ + …` \n\n"
        "Bigger bar = bigger influence on the prediction."
    )
    coefs = pd.Series(model.coef_, index=X.columns).sort_values()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#d62728" if c < 0 else "#2ca02c" for c in coefs]
    ax.barh([FEATURE_LABELS[f] for f in coefs.index], coefs.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("coefficient (effect on progression score)")
    st.pyplot(fig, width="stretch")

    strongest = coefs.abs().idxmax()
    st.success(
        f"Strongest driver: **{FEATURE_LABELS[strongest]}** — "
        "green pushes the prediction up, red pulls it down. "
        f"Intercept (baseline patient): **{model.intercept_:.0f}**."
    )

# =====================================================================
# TAB 5 — explore the raw data
# =====================================================================
with tab_data:
    feature = st.selectbox(
        "Pick a feature to plot against disease progression:",
        list(X.columns),
        format_func=lambda f: FEATURE_LABELS[f],
        index=2,  # bmi — the most interesting one
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(X[feature], y, alpha=0.5, color="#756bb1")
    # simple trend line for the chosen feature alone
    slope, intercept = np.polyfit(X[feature], y, 1)
    xs = np.linspace(X[feature].min(), X[feature].max(), 50)
    ax.plot(xs, slope * xs + intercept, color="#d62728", linewidth=2, label="trend")
    ax.set_xlabel(FEATURE_LABELS[feature])
    ax.set_ylabel("progression score")
    ax.legend()
    st.pyplot(fig, width="stretch")

    st.dataframe(pd.concat([X, y], axis=1), width="stretch", height=300)
