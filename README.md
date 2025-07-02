# Adversarial Robustness & Anomaly Detection for Financial ML Models

This repository implements a rigorous, theoretically sound module for detecting adversarial and out-of-distribution (OOD) anomalies in financial machine learning systems.

---

## 🌟 Overview

Financial systems are vulnerable to both unexpected (OOD) anomalies and adversarial attacks crafted to evade detection. This module integrates robust statistical methods and deep learning techniques to secure models against such risks.

---

## 💡 Key Components

### 🔬 Mahalanobis Distance

Measures the distance of a point from the mean of the normal (in-distribution) data using covariance structure:

$$
D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}
$$

---

### 🤖 Autoencoder Reconstruction Error

Trains an autoencoder to reconstruct normal samples. Anomalous points produce higher reconstruction error:

$$
E(x) = \|x - \hat{x}\|^2
$$

---

### 🔗 Ensemble Score

Combines Mahalanobis and Autoencoder signals:

$$
S(x) = \frac{D_M(x)}{\max D_M} + \frac{E(x)}{\max E}
$$

---

### 📈 Extreme Value Theory (EVT) Thresholding

Rather than arbitrary percentiles, EVT models score tails with a Generalized Pareto Distribution (GPD) to set robust anomaly thresholds.

---

### 🛡️ Adversarial Robustness Test

Evaluates resilience against Fast Gradient Sign Method (FGSM) perturbations, ensuring the detector flags manipulated inputs.

---

### 🧪 Statistical Confidence

Bootstrap resampling is used to estimate 95% confidence intervals for ROC AUC and PR AUC metrics, adding rigorous statistical evaluation.

---

## 🚀 Usage

### 💻 Run the module

```bash
python adversarial-robustness-anomaly-detection.py
```

It will:

* Generate synthetic AR(1) time-series financial data.
* Inject anomalies simulating sudden shocks or regime shifts.
* Train an autoencoder and compute Mahalanobis distances.
* Compute ensemble scores and apply EVT thresholds.
* Evaluate detection metrics and plot results.
* Test against FGSM adversarial attacks.

---

### 🖼️ Outputs

* **Confusion matrix** summarizing detection performance.
* **ROC and PR curves** with AUC metrics and confidence intervals.
* **Plots** showing score distributions and thresholds.
* Adversarial robustness report indicating how many perturbed points were flagged.

---

## 📄 Code Structure

* **Time-Series Simulation**: AR(1) signals to emulate financial returns.
* **Mahalanobis Detector**: Covariance-based classical anomaly score.
* **Autoencoder Detector**: Deep learning reconstruction-based anomaly score.
* **Ensemble Scoring & EVT**: Combines detectors and calibrates thresholds robustly.
* **Evaluation**: Metrics and bootstrap confidence intervals.
* **Adversarial Test**: FGSM attack simulation for robustness validation.

---

## 🤝 Contributing

Contributions to improve robustness tests (e.g., new attack strategies, additional time-series structures) are welcome!

---

## ⚖️ License

MIT License.

---

## 📚 References

* **Mahalanobis Distance**: Mahalanobis, P. C. (1936). "On the generalised distance in statistics."
* **Autoencoders**: Hinton, G. E., & Salakhutdinov, R. R. (2006). "Reducing the dimensionality of data with neural networks."
* **Extreme Value Theory**: Coles, S. (2001). "An Introduction to Statistical Modeling of Extreme Values."
* **Adversarial Attacks**: Goodfellow, I., et al. (2015). "Explaining and Harnessing Adversarial Examples."

---

## ✉️ Contact

For questions or collaborations, please open an issue or contact the maintainer.

---

### 🚨 **Academic & Mathematical Rigor**

This module has been developed with full theoretical rigor, including proofs of EVT threshold consistency, adversarial attack resilience, and statistical evaluation through bootstrap confidence intervals. See the included LaTeX write-up (`docs/paper.tex`) for detailed mathematical explanations and proofs.

---
