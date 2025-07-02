import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.utils import resample
from keras.models import Sequential
from keras.layers import Dense, Input
from scipy.stats import genpareto
import tensorflow as tf

np.random.seed(42)

# ===================
# Time-Series (AR(1)) Data Simulation
# ===================
def generate_ar1(n, phi=0.9, sigma=1.0):
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t-1] + np.random.normal(scale=sigma)
    return x

n_points = 1100
features = np.vstack([generate_ar1(n_points), generate_ar1(n_points), generate_ar1(n_points)]).T

# Inject anomalies
n_anomalies = 100
anomaly_idx = np.random.choice(n_points, size=n_anomalies, replace=False)
features[anomaly_idx] += np.random.normal(10, 5, size=(n_anomalies, 3))

labels = np.zeros(n_points)
labels[anomaly_idx] = 1

scaler = StandardScaler()
data_scaled = scaler.fit_transform(features)

# ===================
# Mahalanobis Detector
# ===================
cov = EmpiricalCovariance().fit(data_scaled[labels == 0])
mahal_dist = cov.mahalanobis(data_scaled)

# EVT threshold for Mahalanobis
normal_mahal = mahal_dist[labels == 0]
params = genpareto.fit(normal_mahal - np.min(normal_mahal))
ev_threshold_mahal = np.min(normal_mahal) + genpareto.ppf(0.99, *params)

# ===================
# Autoencoder Detector
# ===================
X_train = data_scaled[labels == 0]
autoencoder = Sequential([
    Input(shape=(3,)),
    Dense(5, activation='relu'),
    Dense(2, activation='relu'),
    Dense(5, activation='relu'),
    Dense(3, activation='linear')
])
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, verbose=0)

recon = autoencoder.predict(data_scaled)
recon_errors = np.mean(np.square(data_scaled - recon), axis=1)

# EVT threshold for Autoencoder
normal_recon = recon_errors[labels == 0]
params_recon = genpareto.fit(normal_recon - np.min(normal_recon))
ev_threshold_recon = np.min(normal_recon) + genpareto.ppf(0.99, *params_recon)

# ===================
# Ensemble Score
# ===================
ensemble_score = (mahal_dist / np.max(mahal_dist)) + (recon_errors / np.max(recon_errors))
ensemble_threshold = np.percentile(ensemble_score[labels == 0], 99)

# ===================
# Evaluation Metrics
# ===================
fpr, tpr, _ = roc_curve(labels, ensemble_score)
roc_auc = auc(fpr, tpr)
precision, recall, _ = precision_recall_curve(labels, ensemble_score)
pr_auc = auc(recall, precision)

pred_labels = (ensemble_score > ensemble_threshold).astype(int)
conf_mat = confusion_matrix(labels, pred_labels)

# ===================
# Bootstrap Confidence Intervals
# ===================
n_bootstrap = 1000
auc_scores = []
pr_scores = []

for _ in range(n_bootstrap):
    idx = np.random.choice(np.arange(len(labels)), size=len(labels), replace=True)
    boot_labels = labels[idx]
    boot_scores = ensemble_score[idx]
    try:
        fpr_b, tpr_b, _ = roc_curve(boot_labels, boot_scores)
        roc_auc_b = auc(fpr_b, tpr_b)
        precision_b, recall_b, _ = precision_recall_curve(boot_labels, boot_scores)
        pr_auc_b = auc(recall_b, precision_b)
        auc_scores.append(roc_auc_b)
        pr_scores.append(pr_auc_b)
    except:
        continue

roc_ci = (np.percentile(auc_scores, 2.5), np.percentile(auc_scores, 97.5))
pr_ci = (np.percentile(pr_scores, 2.5), np.percentile(pr_scores, 97.5))

# ===================
# FGSM Adversarial Attack Test
# ===================
X_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
with tf.GradientTape() as tape:
    tape.watch(X_tf)
    recon_tf = autoencoder(X_tf)
    loss = tf.reduce_mean(tf.square(X_tf - reco
