from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent

def locate(filename, folder):
    candidates = [
        ROOT / filename,                 # files uploaded directly to repository root
        ROOT / folder / filename,        # files kept in their original folder
        ROOT / "clinisignal_ai" / filename,
        ROOT / "clinisignal_ai" / folder / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]

DATA_PATH = locate("synthetic_icu.csv", "data")
MODEL_PATH = locate("logistic_regression.joblib", "models")
METRICS_PATH = locate("evaluation_metrics.json", "models")

FEATURES = [
    "heart_rate_bpm", "map_mmhg", "resp_rate_min", "temperature_c",
    "spo2_pct", "wbc_10e9_l", "lactate_mmol_l", "creatinine_mg_dl",
    "platelets_10e9_l",
]
LABELS = {
    "heart_rate_bpm": ("Heart rate", "bpm", 40.0, 180.0, 80.0),
    "map_mmhg": ("Mean arterial pressure (MAP)", "mmHg", 40.0, 140.0, 90.0),
    "resp_rate_min": ("Respiratory rate", "breaths/min", 6.0, 45.0, 18.0),
    "temperature_c": ("Temperature", "°C", 34.0, 41.0, 37.0),
    "spo2_pct": ("Oxygen saturation (SpO₂)", "%", 70.0, 100.0, 97.0),
    "wbc_10e9_l": ("White blood cells (WBC)", "10⁹/L", 1.0, 35.0, 8.0),
    "lactate_mmol_l": ("Lactate", "mmol/L", 0.2, 12.0, 1.4),
    "creatinine_mg_dl": ("Creatinine", "mg/dL", 0.2, 5.0, 1.0),
    "platelets_10e9_l": ("Platelets", "10⁹/L", 20.0, 600.0, 230.0),
}

st.set_page_config(page_title="CLINISIGNAL AI | Research Demo", page_icon="🩺", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1400px;}
.demo-subtitle {color:#9CA3AF; font-size:1.05rem; margin-top:-12px;}
.small-note {font-size:.86rem; color:#9CA3AF;}
</style>
""", unsafe_allow_html=True)

st.title("🩺 CLINISIGNAL AI")
st.markdown('<div class="demo-subtitle">ICU Early-Warning Research Prototype · Interactive Technical Demonstration</div>', unsafe_allow_html=True)
st.warning("SYNTHETIC DEMO ONLY — All records and event labels are artificial. This prototype is not a medical device, does not diagnose sepsis, and must never guide patient care or replace clinician judgment.")

missing = [str(p.name) for p in (DATA_PATH, MODEL_PATH) if not p.is_file()]
if missing:
    st.error("Required project file(s) not found: " + ", ".join(missing))
    st.caption("The app checks both the repository root and the original data/ or models/ folders. Confirm that synthetic_icu.csv and logistic_regression.joblib are uploaded.")
    st.stop()

try:
    df = pd.read_csv(DATA_PATH)
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]
    threshold = float(artifact["threshold"])
except Exception as exc:
    st.error(f"Could not load the dataset or trained model: {exc}")
    st.stop()

required_columns = FEATURES + ["patient_id", "hour_from_admission", "risk_event_within_6h"]
missing_columns = [c for c in required_columns if c not in df.columns]
if missing_columns:
    st.error("Dataset is missing required columns: " + ", ".join(missing_columns))
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Synthetic patients", f"{df.patient_id.nunique():,}")
m2.metric("Hourly observations", f"{len(df):,}")
m3.metric("Positive synthetic labels", f"{int(df.risk_event_within_6h.sum()):,}")
m4.metric("Model threshold", f"{threshold:.2f}")

st.divider()
tab_demo, tab_patients, tab_model, tab_about = st.tabs(["Risk estimation", "Patient explorer", "Model evaluation", "About & limitations"])

with tab_demo:
    st.subheader("Simulated patient observation")
    st.caption("Enter nine synthetic measurements or load an example observation from the dataset.")
    mode = st.radio("Input mode", ["Use a sample observation", "Enter values manually"], horizontal=True)
    sample = None
    if mode == "Use a sample observation":
        positive_only = st.checkbox("Show positive-label examples only", value=False)
        pool = df[df.risk_event_within_6h == 1] if positive_only else df
        sample_idx = st.selectbox(
            "Select synthetic observation",
            pool.index.tolist(),
            format_func=lambda i: f"{pool.loc[i, 'patient_id']} · Hour {int(pool.loc[i, 'hour_from_admission'])} · {'Positive' if pool.loc[i, 'risk_event_within_6h'] else 'Negative'} synthetic label"
        )
        sample = pool.loc[sample_idx]
    with st.form("risk_form"):
        cols = st.columns(3)
        values = {}
        for i, key in enumerate(FEATURES):
            label, unit, low, high, default = LABELS[key]
            value = float(sample[key]) if sample is not None and pd.notna(sample[key]) else default
            with cols[i % 3]:
                values[key] = st.number_input(
                    f"{label} ({unit})", min_value=float(low), max_value=float(high),
                    value=float(np.clip(value, low, high)), step=0.1,
                    format="%.2f", key=f"input_{key}"
                )
        submitted = st.form_submit_button("Estimate simulated event risk", type="primary", use_container_width=True)
    if submitted:
        X = pd.DataFrame([values], columns=FEATURES)
        probability = float(pipeline.predict_proba(X)[0, 1])
        predicted = int(probability >= threshold)
        st.divider()
        st.subheader("Model output")
        a, b = st.columns([1, 2])
        with a:
            st.metric("Estimated probability", f"{probability:.1%}")
            st.metric("Threshold", f"{threshold:.2f}")
            st.write("**Simulated classification:**", "Positive label" if predicted else "Negative label")
        with b:
            st.progress(min(max(probability, 0.0), 1.0), text="Model probability for the artificial 6-hour event label")
            st.info("This is a model score for a programmed synthetic target. It is not a patient's clinical risk and must not be interpreted as a sepsis probability.")
        st.caption("Output is generated by the saved Logistic Regression pipeline. SHAP explanations are not implemented in this build.")

with tab_patients:
    st.subheader("Synthetic patient explorer")
    patient_ids = sorted(df.patient_id.unique())
    selected = st.selectbox("Select synthetic patient", patient_ids)
    patient = df[df.patient_id == selected].sort_values("hour_from_admission")
    c1, c2, c3 = st.columns(3)
    c1.metric("Observations", len(patient))
    c2.metric("Positive labels", int(patient.risk_event_within_6h.sum()))
    c3.metric("Synthetic event", "Scheduled" if ("synthetic_event_hour" in patient.columns and patient.synthetic_event_hour.notna().any()) else "None")
    plot_df = patient.melt(
        id_vars=["hour_from_admission"],
        value_vars=["heart_rate_bpm", "map_mmhg", "resp_rate_min"],
        var_name="indicator", value_name="value"
    )
    st.plotly_chart(px.line(plot_df, x="hour_from_admission", y="value", color="indicator", markers=True, title="Selected indicators over time"), use_container_width=True)
    display_df = patient.drop(columns=["synthetic_event_hour"], errors="ignore")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.caption("All patient IDs and observations are synthetic. Event labels are generated by programmed rules.")

with tab_model:
    st.subheader("Held-out synthetic test-set evaluation")
    if METRICS_PATH.is_file():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.caption("Patient-level split: each synthetic patient belongs to one partition only. Threshold selected on validation data.")
        x1, x2, x3, x4, x5 = st.columns(5)
        x1.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
        x2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}" if metrics.get("roc_auc") is not None else "N/A")
        x3.metric("Sensitivity", f"{metrics['sensitivity']:.1%}")
        x4.metric("Precision", f"{metrics['precision']:.1%}")
        x5.metric("F1", f"{metrics['f1']:.3f}")
        st.write("**Patient split**", metrics["patient_counts"])
        tn, fp, fn, tp = metrics["confusion_matrix_tn_fp_fn_tp"]
        st.write(f"Confusion matrix — TN: {tn} · FP: {fp} · FN: {fn} · TP: {tp}")
        st.warning(metrics["warning"])
    else:
        st.info("Evaluation metrics file not found. The prediction demo remains available.")

with tab_about:
    st.subheader("Prototype status & important limitations")
    st.markdown("""
    - **Implemented:** synthetic dataset, patient-level split, preprocessing pipeline, Logistic Regression baseline, validation-selected threshold, interactive risk-estimation demo, and patient explorer.
    - **Planned:** SHAP-based explanations, broader model audit, and further dashboard refinement.
    - **Synthetic-only evidence:** all observations and labels are generated using programmed rules. Reported metrics describe only this artificial dataset and split.
    - **Not a clinical diagnosis:** the target is a simulated event label, not a clinically validated Sepsis-3 diagnosis.
    - **Not for clinical use:** not intended for diagnosis or treatment and must not replace clinician judgment.
    - **Future work:** data-generation audit, external validation using appropriately governed clinical data, and clinical oversight.
    """)
    st.caption("CLINISIGNAL AI · Research prototype · Synthetic data only")
