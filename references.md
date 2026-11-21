# References — FaultSentinel Research Foundation

This document contains verified bibliographic citations and research foundations for the **FaultSentinel** unsupervised machine fault detection project. Each reference has been verified against official publisher repositories (IEEE, MDPI, arXiv, and journal archives).

---

## 1. Primary Verified Research Papers

### 1. Shu et al. (2022) — Convolutional Kernels & Isolation Forest
* **Title:** An anomaly detection method based on random convolutional kernel and isolation forest for equipment state monitoring
* **Authors:** Xinhao Shu, Shigang Zhang, Yue Li, Mengqiao Chen
* **Publication:** *Eksploatacja i Niezawodność – Maintenance and Reliability*, Vol. 24, No. 4, pp. 754–765 (2022)
* **DOI:** [10.17531/ein.2022.4.16](https://doi.org/10.17531/ein.2022.4.16)
* **Dataset:** Complex equipment multi-dimensional state monitoring time series
* **Methodology:** Random convolutional kernels for automatic multi-scale feature transformation combined with Isolation Forest for outlier-resilient anomaly detection.
* **Learning Paradigm:** Unsupervised Anomaly Detection
* **Relevance to FaultSentinel:** Validates using tree-based ensemble isolation methods on latent representations of machine vibration time series without requiring faulty training labels.

---

### 2. Priarone et al. (2024) — Unsupervised Novelty Detection Benchmarking
* **Title:** Unsupervised Novelty Detection Methods Benchmarking with Wavelet Decomposition
* **Authors:** Ariel Priarone, Umberto Albertin, C. Cena, M. Martini, Marcello Chiaberge
* **Publication:** *8th International Conference on System Reliability and Safety (ICSRS)* / arXiv:2409.18683 (2024)
* **Link:** [arXiv:2409.18683](https://arxiv.org/abs/2409.18683)
* **Dataset:** Mechanical actuator vibration under operational variations
* **Methodology:** Wavelet signal decomposition followed by benchmarked novelty detection algorithms including Isolation Forest, Local Outlier Factor (LOF), and One-Class SVM.
* **Learning Paradigm:** Unsupervised Novelty Detection
* **Relevance to FaultSentinel:** Provides an experimental benchmark directly comparing the exact three detectors (IF, LOF, OCSVM) on vibration anomalies under unsupervised conditions.

---

### 3. McKinnon et al. (2020) — Wind Turbine SCADA Anomaly Detection
* **Title:** Comparison of New Anomaly Detection Technique for Wind Turbine Condition Monitoring Using Gearbox SCADA Data
* **Authors:** Conor McKinnon, James Carroll, Alasdair McDonald, Sofia Koukoura, David Infield, Conaill Soraghan
* **Publication:** *Energies*, Vol. 13, No. 19, Article 5152 (2020)
* **DOI:** [10.3390/en13195152](https://doi.org/10.3390/en13195152)
* **Dataset:** Wind turbine gearbox SCADA 10-minute time-series monitoring
* **Methodology:** Quantitative comparison of Isolation Forest, One-Class SVM, and Elliptic Envelope on industrial drive-train anomalies.
* **Learning Paradigm:** Unsupervised / Semi-supervised Anomaly Detection
* **Reported Metrics:** Accuracy ~82% for Isolation Forest and One-Class SVM.
* **Relevance to FaultSentinel:** Serves as a validated baseline demonstrating that boundary-based and density/isolation-based unsupervised detectors reliably isolate rotary machine faults.

---

### 4. Xu et al. (2024) — Deep Recurrent Bearing Fault Diagnosis
* **Title:** Rolling Bearing Fault Diagnosis Based on CNN-LSTM with FFT and SVD
* **Authors:** Ming Xu, Qi Yu, Song Chen, et al.
* **Publication:** *Information* (MDPI), Vol. 15, Issue 7, Article 399 (2024)
* **DOI:** [10.3390/info15070399](https://doi.org/10.3390/info15070399)
* **Dataset:** Case Western Reserve University (CWRU) Bearing Dataset
* **Methodology:** Fast Fourier Transform (FFT) and Singular Value Decomposition (SVD) coupled with spatial-temporal CNN-LSTM networks.
* **Learning Paradigm:** Supervised Multi-class Fault Classification
* **Reported Metrics:** Diagnostic classification accuracy > 99.2%.
* **Relevance to FaultSentinel:** Used as a contextual supervised reference on CWRU to illustrate the contrast between supervised multi-class diagnosis (which requires labelled failure samples) and unsupervised novelty detection (which learns solely from healthy baselines).

---

### 5. Rosa, Braga & Silva (2024) — CWRU Benchmarking & Leakage-Free Splitting
* **Title:** Benchmarking deep learning models for bearing fault diagnosis using the CWRU dataset: A multi-label approach
* **Authors:** Rodrigo Kobashikawa Rosa, Danilo Braga, Danilo Silva
* **Publication:** arXiv:2407.13524 (2024)
* **Link:** [arXiv:2407.13524](https://arxiv.org/abs/2407.13524)
* **Dataset:** Case Western Reserve University (CWRU) Bearing Dataset
* **Methodology:** Critical evaluation of train-test data splitting methodology in bearing fault diagnosis, demonstrating severe data leakage caused by random window assignment across files.
* **Learning Paradigm:** Multi-label benchmark / Methodological validation
* **Relevance to FaultSentinel:** Directly informs our strict recording-wise train/test separation protocol, proving why splitting files prior to overlapping windowing is scientifically necessary to avoid artificially inflated test metrics.

---

### 6. Zhong et al. (2019) — Cascaded Isolation Forest for Turbomachinery
* **Title:** A novel unsupervised anomaly detection for gas turbine using Isolation Forest
* **Authors:** Shisheng Zhong, Song Fu, Lin Lin, Xuyun Fu, Zhiquan Cui, Rui Wang
* **Publication:** *2019 Prognostics and System Health Management Conference (PHM-Qingdao)*, IEEE, pp. 1–6 (2019)
* **DOI:** [10.1109/PHM-Qingdao46334.2019.8942918](https://doi.org/10.1109/PHM-Qingdao46334.2019.8942918)
* **Dataset:** Aeroengine flight-cycle sensor monitoring data
* **Methodology:** Sliding window segmentation with cascaded dual-stage Isolation Forest for anomalous state isolation.
* **Learning Paradigm:** Unsupervised Anomaly Detection
* **Relevance to FaultSentinel:** Highlights the efficacy of isolation trees on segmented mechanical time series when no failure data exists during training.

---

### 7. Amarbayasgalan & Ryu (2024) — Autoencoder Feature Learning
* **Title:** Unsupervised Feature-Construction-Based Motor Fault Diagnosis
* **Authors:** Tsatsral Amarbayasgalan, Keun Ho Ryu
* **Publication:** *Sensors* (MDPI), Vol. 24, No. 10, Article 2978 (2024)
* **DOI:** [10.3390/s24102978](https://doi.org/10.3390/s24102978)
* **Dataset:** Industrial motor vibration datasets
* **Methodology:** Deep Autoencoders for unsupervised latent feature construction and reconstruction error dynamics.
* **Learning Paradigm:** Unsupervised Representation Learning
* **Relevance to FaultSentinel:** Directly supports our Section 6 (LSTM Autoencoder) design, demonstrating that deep latent embeddings extract salient health-indicator representations.

---

### 8. Xu, Lei & Zhou (2018) — Local Outlier Factor Machinery Monitoring
* **Title:** A LOF-based method for abnormal segment detection in machinery condition monitoring
* **Authors:** Xuefang Xu, Yaguo Lei, Xin Zhou
* **Publication:** *2018 Prognostics and System Health Management Conference (PHM-Chongqing)*, IEEE, pp. 268–272 (2018)
* **DOI:** [10.1109/PHM-Chongqing.2018.00051](https://doi.org/10.1109/PHM-Chongqing.2018.00051)
* **Dataset:** Industrial machinery condition monitoring time series
* **Methodology:** Windowed time-domain feature extraction with PCA dimensionality reduction and Local Outlier Factor scoring.
* **Learning Paradigm:** Unsupervised Outlier Detection
* **Relevance to FaultSentinel:** Supports our Section 9 (Local Outlier Factor) design, confirming density-based neighborhood deviations capture mechanical degradation.

---

## 2. Dataset Attribution
* **Dataset Name:** Case Western Reserve University (CWRU) Bearing Data Center
* **Institution:** Case School of Engineering, Case Western Reserve University, Cleveland, Ohio, USA
* **Website:** [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter)
* **Apparatus:** 2 hp Reliance Electric motor, dynamometer, torque transducer, deep groove ball bearings (6205-2RS JEM SKF at drive end; 6203-2RS JEM SKF at fan end).
* **Accelerometers:** PCB 353B33 high-frequency accelerometers installed at drive end (DE) housing and fan end (FE) housing, sampled at 12,000 Hz and 48,000 Hz.

