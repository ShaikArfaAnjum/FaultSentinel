# FaultSentinel: Machine Fault Detection Using Unsupervised Learning

[![Python 3.11.9](https://img.shields.io/badge/python-3.11.9-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.x-green.svg)](https://scikit-learn.org)
[![Dataset](https://img.shields.io/badge/Dataset-CWRU%20Bearing%20Data-red.svg)](https://engineering.case.edu/bearingdatacenter)

---

## 1. Project Overview & Objective

Industrial rotating machinery (such as electric motors, gearboxes, pumps, and turbines) is subject to mechanical fatigue, bearing degradation, and sudden failure. In realistic industrial manufacturing environments, machines predominantly operate in a healthy state, while catastrophic failure signals are rare, costly, and hazardous to collect. Consequently, supervised multi-class classifiers that require exhaustive sets of labeled fault recordings are impractical for initial anomaly screening.

**FaultSentinel** is an end-to-end unsupervised machine fault detection and novelty screening framework built upon vibration recordings from the **Case Western Reserve University (CWRU) Bearing Data Center**. 

### Primary Objective
> **To develop a rigorous, unsupervised machine fault detection system that learns exclusively from healthy bearing vibration signals and accurately identifies unseen vibration signals as either Normal or Anomaly without relying on fault labels for training, representation learning, or threshold determination.**

---

## 2. Team Details & Academic Submission

* **Project Title:** FaultSentinel: Machine Fault Detection Using Unsupervised Learning
* **Course Submission:** Section 12
* **Submitted to:** Mr. Sourav Mondal

### Team Members
1. **K. Bhavya** — Roll No: `241FA04172`
2. **Y. Pavani** — Roll No: `241FA04436`
3. **Sk. Hiba** — Roll No: `241FA04495`
4. **Sk. Arfa** — Roll No: `241FA04931`

---

## 3. Repository Folder Structure

```text
MLProject/
│
├── dataset/                                   # Downloaded CWRU MATLAB (.mat) vibration files
│   ├── 97.mat, 98.mat, 99.mat, 100.mat        # Healthy baseline recordings
│   ├── 105.mat, 106.mat, 107.mat, 108.mat     # Inner race fault recordings
│   ├── 118.mat, 119.mat, 120.mat, 121.mat     # Ball fault recordings
│   └── 130.mat, 131.mat, 132.mat, 133.mat     # Outer race fault recordings
│
├── notebooks/
│   └── FaultSentinel_Complete_Project.ipynb  # Single comprehensive Jupyter Notebook (Sections 1-24)
│
├── models/
│   ├── feature_extractors/
│   │   ├── lstm_encoder/                      # Saved LSTM Sequence Autoencoder / Encoder
│   │   └── autoencoder/                       # Saved Full LSTM Autoencoder & Preprocessing Scalers
│   │
│   └── anomaly_detectors/
│       ├── isolation_forest/                  # Fitted Isolation Forest detectors
│       ├── lof/                               # Fitted Local Outlier Factor detectors (novelty=True)
│       └── one_class_svm/                     # Fitted One-Class SVM detectors & Ensemble Config
│
├── results/
│   ├── figures/                               # Confusion matrices, ROC curves, PR curves, score plots
│   ├── tables/                                # Experimental & literature comparison CSV files
│   ├── metrics/                               # Saved quantitative evaluation metric JSON files
│   └── predictions/                           # New-input inference logs & verdicts
│
├── requirements.txt                           # Verified Python dependencies
├── README.md                                  # Complete project documentation
└── references.md                              # Verified academic literature citations & DOIs
```

---

## 4. Methodology & Theoretical Architecture

```
[Raw Vibration Signal (.mat)]
             │
             ▼
[Signal Verification & Scaling]  (Fitted ONLY on Healthy Training Recordings)
             │
             ▼
[Recording-Wise Sliding Windows] (Window: 2048 samples, Step: 1024 samples, 50% Overlap)
             │
    ┌────────┴────────────────────────┬────────────────────────┐
    ▼                                 ▼                        ▼
[LSTM Encoder Features]      [LSTM Autoencoder Latent]     [Hybrid Concatenation]
(Self-Supervised Sequence)   (Reconstruction Bottleneck)   (Fused Feature Vector)
    │                                 │                        │
    └────────────────┬────────────────┴────────────────────────┘
                     │
                     ▼
          [Unsupervised Anomaly Detectors]
          ├── Isolation Forest (iForest)
          ├── Local Outlier Factor (LOF with novelty=True)
          └── One-Class Support Vector Machine (OCSVM)
                     │
                     ▼
          [Score Calibration & Standardization]
          (MinMax / Robust Scaling on Healthy Validation Scores)
                     │
                     ▼
          [Unsupervised Score-Level Ensemble Fusion]
          (Equal-Weight Score Fusion: S_ensemble = mean(S_IF, S_LOF, S_OCSVM))
                     │
                     ▼
          [Unsupervised Thresholding] (95th/99th Percentile of Healthy Validation)
                     │
                     ▼
             [Final Verdict: NORMAL or ANOMALY]
```

### Key Scientific Principles
1. **Zero Fault-Data Leakage:** All scalers, LSTM representation learners, and anomaly detectors are trained exclusively on healthy baseline vibration recordings (`97.mat`, `98.mat`, `99.mat`).
2. **Recording-Wise Splitting:** Data is partitioned by file identifier before sliding-window segmentation, completely avoiding overlapping window leakage between training and testing sets (Rosa et al., 2024).
3. **Novelty Detection Protocol:** Local Outlier Factor is instantiated with `novelty=True` to compute outlier factors on unseen test points against the fitted healthy neighborhood graph.
4. **Unsupervised Score Fusion:** No supervised meta-classifiers (e.g. Logistic Regression or Gradient Boosting) are used. The ensemble combines calibrated anomaly scores with thresholding calibrated strictly on healthy validation data.

---

## 5. Dataset Attribution & Automatic Download

The dataset is sourced from the **Case Western Reserve University (CWRU) Bearing Data Center**.
* Test rig: 2 hp electric motor driving a dynamometer via torque transducer.
* Accelerometer: PCB 353B33 accelerometer placed at the Drive End (DE) housing sampled at 12,000 samples/sec (12k).
* Candidate files:
  - **Healthy Baseline:** `97.mat` (0 HP), `98.mat` (1 HP), `99.mat` (2 HP), `100.mat` (3 HP)
  - **Inner Race Faults (0.007"):** `105.mat`, `106.mat`, `107.mat`, `108.mat`
  - **Ball Faults (0.007"):** `118.mat`, `119.mat`, `120.mat`, `121.mat`
  - **Outer Race Faults (0.007", 6 o'clock):** `130.mat`, `131.mat`, `132.mat`, `133.mat`

The notebook includes an automated download utility with checksum, status, and MATLAB loadability checks using `scipy.io.loadmat`.

---

## 6. Installation & Execution

### Prerequisites
* Python 3.11.9
* Virtual environment (`.venv`)

### Setup Instructions
1. Activate the virtual environment:
   ```powershell
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit Web Application:
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501` to use the interactive dashboard.

4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/FaultSentinel_Complete_Project.ipynb
   ```

---

## 7. Prediction on New Input Signals

The project includes an inference pipeline (`predict_recording(file_path)`):
1. Loads the specified `.mat` file.
2. Extracts the drive-end vibration signal (`X..._DE_time`).
3. Applies the saved healthy-fitted scaler.
4. Generates non-leaking sliding windows of size 2,048 samples.
5. Extracts features using the saved models (LSTM, Autoencoder, Hybrid).
6. Computes anomaly scores across detectors and ensemble.
7. Evaluates against the calibrated unsupervised threshold.
8. Outputs the summary breakdown (% anomalous windows) and the final **NORMAL** or **ANOMALY** verdict.

---

## 8. Limitations & Scope

* **Single Sensor Placement:** The current pipeline focuses on the Drive End (DE) accelerometer. Fan End (FE) accelerometers can provide complementary spatial signals.
* **Domain / Speed Shift:** Models trained on 0-2 HP healthy data are evaluated on varying loads (0-3 HP); severe rotational speed shifts across different motor architectures would benefit from rotational-order tracking or domain adaptation.
* **Acoustic Noise:** In harsh industrial environments, environmental ambient acoustic noise may elevate baseline reconstruction error, requiring periodic healthy recalibration.

