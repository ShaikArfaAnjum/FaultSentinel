"""
FaultSentinel: Machine Fault Detection Using Unsupervised Learning
Interactive Streamlit Condition Monitoring Dashboard
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import scipy.io
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
os.environ['MPLCONFIGDIR'] = os.path.abspath('.tmp_mpl')
matplotlib.use('Agg')

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FaultSentinel | Machine Fault Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, high-contrast, modern presentation
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.2rem;
    }
    .team-badge {
        background: linear-gradient(90deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 14px 20px;
        margin-bottom: 1.6rem;
        font-size: 0.92rem;
        color: #E2E8F0;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
    }
    
    /* Modern High-Contrast Metric Cards */
    .metric-card {
        background: #111827 !important;
        border-radius: 14px !important;
        padding: 20px 14px !important;
        border: 1px solid #1F2937 !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.45) !important;
        text-align: center !important;
        position: relative !important;
        overflow: hidden !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease !important;
    }
    .metric-card:hover {
        transform: translateY(-3px) !important;
        border-color: #374151 !important;
        box-shadow: 0 14px 28px -4px rgba(0, 0, 0, 0.6) !important;
    }
    .card-total {
        border-top: 4px solid #38BDF8 !important;
    }
    .card-normal {
        border-top: 4px solid #10B981 !important;
    }
    .card-anomaly {
        border-top: 4px solid #EF4444 !important;
    }
    .card-threshold {
        border-top: 4px solid #F59E0B !important;
    }
    .metric-icon {
        font-size: 1.35rem !important;
        margin-bottom: 4px !important;
        line-height: 1 !important;
    }
    .metric-title {
        font-size: 0.76rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: #9CA3AF !important;
        margin-bottom: 6px !important;
    }
    .metric-value {
        font-size: 2.3rem !important;
        font-weight: 800 !important;
        line-height: 1.1 !important;
        margin: 4px 0 8px 0 !important;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important;
    }
    .metric-value-total {
        color: #38BDF8 !important;
        text-shadow: 0 0 14px rgba(56, 189, 248, 0.35) !important;
    }
    .metric-value-normal {
        color: #34D399 !important;
        text-shadow: 0 0 14px rgba(52, 211, 153, 0.35) !important;
    }
    .metric-value-anomaly {
        color: #F87171 !important;
        text-shadow: 0 0 14px rgba(248, 113, 113, 0.35) !important;
    }
    .metric-value-threshold {
        color: #FBBF24 !important;
        text-shadow: 0 0 14px rgba(251, 191, 36, 0.35) !important;
    }
    .metric-subtitle {
        font-size: 0.76rem !important;
        font-weight: 500 !important;
        color: #D1D5DB !important;
        background: #1F2937 !important;
        padding: 3px 10px !important;
        border-radius: 9999px !important;
        display: inline-block !important;
        border: 1px solid #374151 !important;
    }

    .verdict-normal {
        background: linear-gradient(135deg, #064E3B 0%, #065F46 100%) !important;
        border: 2px solid #10B981 !important;
        border-radius: 14px !important;
        padding: 20px 24px !important;
        text-align: center !important;
        box-shadow: 0 8px 24px -4px rgba(16, 185, 129, 0.3) !important;
        margin: 18px 0 !important;
    }
    .verdict-normal-title {
        font-size: 1.45rem !important;
        font-weight: 800 !important;
        color: #6EE7B7 !important;
        letter-spacing: 0.03em !important;
        margin-bottom: 6px !important;
    }
    .verdict-anomaly {
        background: linear-gradient(135deg, #7F1D1D 0%, #991B1B 100%) !important;
        border: 2px solid #EF4444 !important;
        border-radius: 14px !important;
        padding: 20px 24px !important;
        text-align: center !important;
        box-shadow: 0 8px 24px -4px rgba(239, 68, 68, 0.4) !important;
        margin: 18px 0 !important;
    }
    .verdict-anomaly-title {
        font-size: 1.45rem !important;
        font-weight: 800 !important;
        color: #FCA5A5 !important;
        letter-spacing: 0.03em !important;
        margin-bottom: 6px !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Constants & Hyperparameters
# -----------------------------------------------------------------------------
WINDOW_SIZE = 2048
STEP_SIZE = 1024
TIME_STEPS = 128
FEATURES_PER_STEP = 16

CWRU_DESCRIPTIONS = {
    '100.mat': 'Normal Baseline (3 HP Load, 1730 RPM) — Held-out Healthy Test',
    '97.mat':  'Normal Baseline (0 HP Load, 1797 RPM) — Healthy Training Data',
    '98.mat':  'Normal Baseline (1 HP Load, 1772 RPM) — Healthy Training Data',
    '99.mat':  'Normal Baseline (2 HP Load, 1750 RPM) — Healthy Training Data',
    '105.mat': 'Inner Race Fault (0.007" Defect, 0 HP, 1797 RPM)',
    '106.mat': 'Inner Race Fault (0.007" Defect, 1 HP, 1772 RPM)',
    '107.mat': 'Inner Race Fault (0.007" Defect, 2 HP, 1748 RPM)',
    '108.mat': 'Inner Race Fault (0.007" Defect, 3 HP, 1721 RPM)',
    '118.mat': 'Ball Fault (0.007" Defect, 0 HP, 1797 RPM)',
    '119.mat': 'Ball Fault (0.007" Defect, 1 HP, 1772 RPM)',
    '120.mat': 'Ball Fault (0.007" Defect, 2 HP, 1748 RPM)',
    '121.mat': 'Ball Fault (0.007" Defect, 3 HP, 1721 RPM)',
    '130.mat': 'Outer Race Fault (0.007" Defect, 0 HP, 1797 RPM)',
    '131.mat': 'Outer Race Fault (0.007" Defect, 1 HP, 1772 RPM)',
    '132.mat': 'Outer Race Fault (0.007" Defect, 2 HP, 1748 RPM)',
    '133.mat': 'Outer Race Fault (0.007" Defect, 3 HP, 1721 RPM)'
}

# -----------------------------------------------------------------------------
# Model & Scaler Loading (Cached for Performance)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained machine learning & deep learning models...")
def load_all_models():
    import tensorflow as tf
    
    # Scalers
    sig_scaler = joblib.load("models/feature_extractors/autoencoder/signal_scaler.joblib")
    hy_scaler = joblib.load("models/feature_extractors/autoencoder/hybrid_scaler.joblib")
    det_feat_scaler = joblib.load("models/anomaly_detectors/isolation_forest/hybrid_lstm_plus_autoencoder_detector_feature_scaler.joblib")
    
    # Deep Feature Extractors
    lstm_enc = tf.keras.models.load_model("models/feature_extractors/lstm_encoder/lstm_encoder.keras", compile=False)
    ae_enc = tf.keras.models.load_model("models/feature_extractors/autoencoder/ae_encoder.keras", compile=False)
    
    # Anomaly Detectors
    if_model = joblib.load("models/anomaly_detectors/isolation_forest/hybrid_lstm_plus_autoencoder_isolation_forest.joblib")
    lof_model = joblib.load("models/anomaly_detectors/lof/hybrid_lstm_plus_autoencoder_lof.joblib")
    ocsvm_model = joblib.load("models/anomaly_detectors/one_class_svm/hybrid_lstm_plus_autoencoder_one_class_svm.joblib")
    
    # Ensemble Config
    with open("models/anomaly_detectors/one_class_svm/ensemble_config.json") as f:
        ens_cfg = json.load(f)
        
    return {
        "sig_scaler": sig_scaler,
        "hy_scaler": hy_scaler,
        "det_feat_scaler": det_feat_scaler,
        "lstm_enc": lstm_enc,
        "ae_enc": ae_enc,
        "if_model": if_model,
        "lof_model": lof_model,
        "ocsvm_model": ocsvm_model,
        "ens_cfg": ens_cfg
    }

def generate_sliding_windows(signal, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    n_windows = (len(signal) - window_size) // step_size + 1
    windows = np.empty((n_windows, window_size), dtype=np.float32)
    for i in range(n_windows):
        start = i * step_size
        windows[i] = signal[start : start + window_size]
    return windows

def parse_mat_signal(file_obj_or_path):
    mat = scipy.io.loadmat(file_obj_or_path)
    de_keys = [k for k in mat.keys() if 'DE_time' in k]
    if not de_keys:
        raise ValueError("Could not find a Drive End ('DE_time') vibration channel in the file.")
    return mat[de_keys[0]].flatten(), de_keys[0]

# -----------------------------------------------------------------------------
# Main Dashboard UI
# -----------------------------------------------------------------------------
def main():
    # Top Header
    st.markdown('<div class="main-header">🛡️ FaultSentinel: Machine Fault Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Unsupervised Novelty Detection & Machinery Health Monitoring on CWRU Bearing Dataset</div>', unsafe_allow_html=True)

    
    # Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/fluency/96/engine.png", width=70)
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Select Module:",
        [
            "🔍 Live Fault Diagnosis",
            "📊 Model Comparison Table",
            "📈 Visual Diagnostics",
            
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Configuration")
    st.sidebar.info("""
    - **Methodology:** Unsupervised
    - **Training:** Healthy Data Only
    - **Sampling Rate:** 12,000 Hz
    - **Window Size:** 2048 samples (~170.7 ms)
    - **Window Overlap:** 50% (Step: 1024)
    - **Ensemble Threshold (τ):** 1.5761
    - **Machine Alert Rule:** ≥ 15% Anomaly
    """)

    # Load Models
    try:
        models_dict = load_all_models()
    except Exception as e:
        st.error(f"Error loading saved models: {e}")
        st.stop()

    # -------------------------------------------------------------------------
    # TAB 1: LIVE FAULT DIAGNOSIS
    # -------------------------------------------------------------------------
    if app_mode == "🔍 Live Fault Diagnosis":
        st.subheader("🔍 Real-Time Vibration Signal Analysis")
        st.markdown("Select a genuine CWRU vibration recording from the library or upload a custom `.mat` file.")

        col_input1, col_input2 = st.columns([1, 1])

        with col_input1:
            input_source = st.radio("Choose Input Source:", ["CWRU Benchmark Library", "Upload Custom .mat File"])
            
            signal_data = None
            filename = None
            variable_name = None

            if input_source == "CWRU Benchmark Library":
                selected_file = st.selectbox(
                    "Select Recording:",
                    list(CWRU_DESCRIPTIONS.keys()),
                    format_func=lambda x: f"{x} — {CWRU_DESCRIPTIONS[x]}"
                )
                filepath = os.path.join("dataset", selected_file)
                if os.path.exists(filepath):
                    signal_data, variable_name = parse_mat_signal(filepath)
                    filename = selected_file
                else:
                    st.error(f"File {filepath} not found on disk.")
            else:
                uploaded = st.file_uploader("Upload MATLAB (.mat) file:", type=["mat"])
                if uploaded is not None:
                    try:
                        signal_data, variable_name = parse_mat_signal(uploaded)
                        filename = uploaded.name
                    except Exception as e:
                        st.error(f"Error reading file: {e}")

        with col_input2:
            if signal_data is not None:
                st.markdown("#### Signal Metadata")
                st.write(f"- **Filename:** `{filename}`")
                st.write(f"- **Vibration Channel:** `{variable_name}`")
                st.write(f"- **Total Samples:** `{len(signal_data):,}`")
                duration_sec = len(signal_data) / 12000.0
                st.write(f"- **Duration:** `{duration_sec:.2f} seconds` (@ 12,000 Hz)")
                st.write(f"- **Signal Standard Deviation:** `{np.std(signal_data):.4f} g`")

        if signal_data is not None:
            # Preview waveform
            with st.expander("📊 View Raw Vibration Waveform (First 2048 Samples)", expanded=False):
                fig, ax = plt.subplots(figsize=(10, 3.2))
                fig.patch.set_facecolor('#0F172A')
                ax.set_facecolor('#1E293B')
                ax.plot(signal_data[:2048], color="#38BDF8", lw=1.2)
                ax.set_title(f"Raw Vibration Waveform snippet: {filename}", fontsize=11, fontweight="bold", color="#F1F5F9")
                ax.set_xlabel("Sample Index", color="#94A3B8")
                ax.set_ylabel("Amplitude (g)", color="#94A3B8")
                ax.tick_params(colors="#94A3B8")
                for spine in ax.spines.values():
                    spine.set_color("#334155")
                ax.grid(True, color="#334155", linestyle="--", alpha=0.5)
                st.pyplot(fig)
                plt.close(fig)

            # Analyze Button
            if st.button("🚀 Run Anomaly Diagnosis Pipeline", type="primary", use_container_width=True):
                with st.spinner("Processing signal through deep learning encoders and unsupervised detectors..."):
                    # 1. Sliding Windows
                    windows_raw = generate_sliding_windows(signal_data)
                    windows_scaled = models_dict["sig_scaler"].transform(windows_raw)
                    windows_seq = windows_scaled.reshape(-1, TIME_STEPS, FEATURES_PER_STEP)
                    n_windows = len(windows_seq)

                    # 2. Deep Feature Extraction
                    f_lstm = models_dict["lstm_enc"].predict(windows_seq[:, :-1, :], verbose=0)
                    f_ae = models_dict["ae_enc"].predict(windows_seq, verbose=0)
                    f_hybrid = models_dict["hy_scaler"].transform(np.concatenate([f_lstm, f_ae], axis=1))

                    # 3. Detector Feature Scaling
                    f_hybrid_scaled = models_dict["det_feat_scaler"].transform(f_hybrid)

                    # 4. Detector Scoring (Inverted so higher = more anomalous)
                    s_if = -models_dict["if_model"].score_samples(f_hybrid_scaled)
                    s_lof = -models_dict["lof_model"].decision_function(f_hybrid_scaled)
                    s_ocsvm = -models_dict["ocsvm_model"].decision_function(f_hybrid_scaled)

                    # 5. Score-Level Standardization & Fusion
                    ens_cfg = models_dict["ens_cfg"]
                    k_if = "Hybrid LSTM + Autoencoder___Isolation Forest"
                    k_lof = "Hybrid LSTM + Autoencoder___Local Outlier Factor"
                    k_svm = "Hybrid LSTM + Autoencoder___One-Class SVM"

                    z_if = (s_if - ens_cfg["calibration"][k_if]["mean"]) / ens_cfg["calibration"][k_if]["std"]
                    z_lof = (s_lof - ens_cfg["calibration"][k_lof]["mean"]) / ens_cfg["calibration"][k_lof]["std"]
                    z_svm = (s_ocsvm - ens_cfg["calibration"][k_svm]["mean"]) / ens_cfg["calibration"][k_svm]["std"]

                    s_ensemble = (z_if + z_lof + z_svm) / 3.0
                    threshold = ens_cfg["threshold"]

                    # 6. Classification
                    window_predictions = (s_ensemble >= threshold).astype(int)
                    n_anomalous = int(np.sum(window_predictions))
                    n_normal = n_windows - n_anomalous
                    anomaly_rate = (n_anomalous / n_windows) * 100.0

                    is_anomaly = anomaly_rate >= 15.0
                    verdict = "ANOMALY DETECTED" if is_anomaly else "NORMAL BASELINE"

                # Results Display
                st.markdown("---")
                st.markdown("### 📋 Diagnostic Assessment")

                if is_anomaly:
                    st.markdown(f"""
                    <div class="verdict-anomaly">
                        <div class="verdict-anomaly-title">⚠️ MACHINE STATE: {verdict}</div>
                        <div style="font-size: 1.05rem; font-weight: 500; color: #FEE2E2;">
                            Vibration patterns deviate significantly from healthy baseline ({anomaly_rate:.1f}% anomalous windows). Incipient mechanical fault detected.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="verdict-normal">
                        <div class="verdict-normal-title">✅ MACHINE STATE: {verdict}</div>
                        <div style="font-size: 1.05rem; font-weight: 500; color: #ECFDF5;">
                            Vibration signals strictly conform to healthy baseline (only {anomaly_rate:.1f}% anomalous windows below tolerance). Equipment is operating normally.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"""
                    <div class="metric-card card-total">
                        <div class="metric-icon">📦</div>
                        <div class="metric-title">Total Windows</div>
                        <div class="metric-value metric-value-total">{n_windows:,}</div>
                        <div class="metric-subtitle">2,048 samples each</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-card card-normal">
                        <div class="metric-icon">🟢</div>
                        <div class="metric-title">Normal Windows</div>
                        <div class="metric-value metric-value-normal">{n_normal:,}</div>
                        <div class="metric-subtitle">{(n_normal/n_windows)*100:.1f}% healthy</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="metric-card card-anomaly">
                        <div class="metric-icon">🔴</div>
                        <div class="metric-title">Anomalous Windows</div>
                        <div class="metric-value metric-value-anomaly">{n_anomalous:,}</div>
                        <div class="metric-subtitle">{anomaly_rate:.1f}% fault rate</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m4:
                    st.markdown(f"""
                    <div class="metric-card card-threshold">
                        <div class="metric-icon">🎯</div>
                        <div class="metric-title">Ensemble Threshold</div>
                        <div class="metric-value metric-value-threshold">{threshold:.4f}</div>
                        <div class="metric-subtitle">Calibrated 99th %-tile (τ)</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Visualizing Timeline
                st.markdown("#### 📈 Window-by-Window Anomaly Score Trajectory")
                fig_timeline, ax_timeline = plt.subplots(figsize=(12, 4.2))
                fig_timeline.patch.set_facecolor('#0F172A')
                ax_timeline.set_facecolor('#1E293B')
                window_indices = np.arange(n_windows)
                
                # Normal windows in emerald green, anomalous in ruby red
                normal_mask = window_predictions == 0
                anom_mask = window_predictions == 1
                
                ax_timeline.plot(window_indices, s_ensemble, color="#64748B", lw=0.9, alpha=0.6, label="Score Trajectory")
                ax_timeline.scatter(window_indices[normal_mask], s_ensemble[normal_mask], color="#10B981", s=18, alpha=0.8, edgecolors="none", label=f"Normal ({n_normal})")
                ax_timeline.scatter(window_indices[anom_mask], s_ensemble[anom_mask], color="#EF4444", s=32, alpha=0.95, edgecolors="#FCA5A5", linewidths=0.5, label=f"Anomalous ({n_anomalous})")
                
                # Threshold line
                ax_timeline.axhline(threshold, color="#F59E0B", linestyle="--", lw=2.0, label=f"Decision Threshold (τ = {threshold:.4f})")
                ax_timeline.set_title(f"Standardized Ensemble Score Profile across Windows: {filename}", fontsize=11, fontweight="bold", color="#F1F5F9")
                ax_timeline.set_xlabel("Sliding Window Index (Chronological)", color="#94A3B8", fontsize=10)
                ax_timeline.set_ylabel("Standardized Anomaly Score (Z-Score)", color="#94A3B8", fontsize=10)
                ax_timeline.tick_params(colors="#94A3B8")
                for spine in ax_timeline.spines.values():
                    spine.set_color("#334155")
                ax_timeline.legend(loc="upper right", facecolor="#0F172A", edgecolor="#334155", labelcolor="#E2E8F0")
                ax_timeline.grid(True, color="#334155", linestyle="--", alpha=0.5)
                st.pyplot(fig_timeline)
                plt.close(fig_timeline)

                # Individual Detector Breakdown Expander
                with st.expander("🔍 Individual Detector Score Breakdown"):
                    c_if, c_lof, c_svm = st.columns(3)
                    with c_if:
                        st.markdown(f"""
                        <div style="background: #1E293B; border-radius: 10px; padding: 14px; border: 1px solid #334155;">
                            <h5 style="color: #38BDF8; margin: 0 0 6px 0;">🌲 Isolation Forest</h5>
                            <div style="color: #CBD5E1; font-size: 0.9rem;">Mean Score: <code style="color: #F1F5F9;">{np.mean(s_if):.4f}</code></div>
                            <div style="color: #CBD5E1; font-size: 0.9rem;">Max Score: <code style="color: #F1F5F9;">{np.max(s_if):.4f}</code></div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c_lof:
                        st.markdown(f"""
                        <div style="background: #1E293B; border-radius: 10px; padding: 14px; border: 1px solid #334155;">
                            <h5 style="color: #34D399; margin: 0 0 6px 0;">📍 Local Outlier Factor</h5>
                            <div style="color: #CBD5E1; font-size: 0.9rem;">Mean Score: <code style="color: #F1F5F9;">{np.mean(s_lof):.4f}</code></div>
                            <div style="color: #CBD5E1; font-size: 0.9rem;">Max Score: <code style="color: #F1F5F9;">{np.max(s_lof):.4f}</code></div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c_svm:
                        st.markdown(f"""
                        <div style="background: #1E293B; border-radius: 10px; padding: 14px; border: 1px solid #334155;">
                            <h5 style="color: #FBBF24; margin: 0 0 6px 0;">📐 One-Class SVM</h5>
                            <div style="color: #CBD5E1; font-size: 0.9rem;">Mean Score: <code style="color: #F1F5F9;">{np.mean(s_ocsvm):.4f}</code></div>
                            <div style="color: #CBD5E1; font-size: 0.9rem;">Max Score: <code style="color: #F1F5F9;">{np.max(s_ocsvm):.4f}</code></div>
                        </div>
                        """, unsafe_allow_html=True)

                # Export JSON Report
                report = {
                    "File Name": filename,
                    "Total Windows Processed": n_windows,
                    "Normal Windows": n_normal,
                    "Anomaly Windows": n_anomalous,
                    "Anomaly Percentage (%)": round(anomaly_rate, 2),
                    "Mean Ensemble Score": round(float(np.mean(s_ensemble)), 4),
                    "Calibrated Threshold": round(threshold, 4),
                    "Final Machine State Verdict": verdict
                }
                st.download_button(
                    label="💾 Download Diagnostic JSON Report",
                    data=json.dumps(report, indent=4),
                    file_name=f"FaultSentinel_{filename}_Report.json",
                    mime="application/json"
                )

    # -------------------------------------------------------------------------
    # TAB 2: MODEL COMPARISON TABLE
    # -------------------------------------------------------------------------
    elif app_mode == "📊 Model Comparison Table":
        st.subheader("📊 Experimental Model Performance Comparison")
        st.markdown("""
        All 9 combinations of feature extractors (LSTM, LSTM Autoencoder, Hybrid) and anomaly detectors 
        (Isolation Forest, Local Outlier Factor, One-Class SVM) evaluated on the held-out test set ($N = 1,885$ windows).
        """)

        if os.path.exists("results/tables/final_model_comparison.csv"):
            df = pd.read_csv("results/tables/final_model_comparison.csv")
            
            # Format dataframe for display
            df_display = df.copy()
            for col in ["Accuracy", "Precision", "Recall", "F1-score"]:
                df_display[col] = df_display[col].apply(lambda x: f"{x*100:.2f}%")
            for col in ["ROC-AUC", "PR-AUC", "Threshold"]:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")
                
            st.dataframe(df_display, use_container_width=True)

            st.markdown("### 🏆 Key Findings:")
            st.markdown("""
            1. **Autoencoder Bottleneck Features:** Achieved the highest individual performance (ROC-AUC $>0.99$, reaching **98.94% Accuracy** and **99.29% F1-score** with LOF).
            2. **Local Outlier Factor (`novelty=True`):** Consistently outperformed tree and hyperplane boundaries on dense deep latent clusters.
            3. **Score-Level Ensemble:** Fuses complementary perspectives, achieving **93.69% Accuracy**, **99.01% Precision**, **95.64% F1-score**, and **0.9872 ROC-AUC** with zero supervised tuning.
            """)
        else:
            st.warning("Comparison table not found at results/tables/final_model_comparison.csv")

    # -------------------------------------------------------------------------
    # TAB 3: VISUAL DIAGNOSTICS
    # -------------------------------------------------------------------------
    elif app_mode == "📈 Visual Diagnostics":
        st.subheader("📈 Diagnostic Figures & Evaluation Curves")
        st.markdown("High-resolution diagnostic visualizations saved directly from the experimental pipeline.")

        figures = [
            ("results/figures/training_loss_curves.png", "1. Deep Feature Extractor Training Loss Curves (LSTM & Autoencoder)"),
            ("results/figures/confusion_matrix_ensemble.png", "2. Confusion Matrix Heatmap — Score-Level Ensemble"),
            ("results/figures/roc_curves.png", "3. Receiver Operating Characteristic (ROC) Curves Benchmark"),
            ("results/figures/pr_curves.png", "4. Precision-Recall (PR) Curves Benchmark"),
            ("results/figures/anomaly_score_distribution.png", "5. Calibrated Anomaly Score Distributions (Healthy vs Anomalous)"),
            ("results/figures/sample_vibration_signals.png", "6. Sample Vibration Waveforms (Baseline vs Seeded Faults)")
        ]

        for path, title in figures:
            if os.path.exists(path):
                st.markdown(f"#### {title}")
                st.image(path, use_container_width=True)
                st.markdown("---")

    
    

if __name__ == "__main__":
    main()

