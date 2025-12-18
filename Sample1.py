#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.io as sio
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix


import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Load the .mat file
mat_data = sio.loadmat('sample_1.mat')

data = mat_data['data'].squeeze() 

FS = 24000    # Measurement Frequency

# ---------- Time vector ----------
t = np.arange(len(data)) / FS

plt.figure(figsize=(18, 6))
start_sec = 0    # <-- enter start time in seconds
end_sec   = 0.01       # <-- enter end time in seconds

start_idx = int(start_sec * FS)
end_idx   = int(end_sec * FS)
plt.plot(t[start_idx:end_idx], data[start_idx:end_idx], linewidth=1, color='red') # plot some data  
plt.xlabel("Time (s)")
plt.ylabel("Voltage (µV)")
plt.title("Raw Extracellular Neural Recording")
plt.grid(False)
plt.show()


# In[3]:


# ---------- Filter (300–3000 Hz) ----------

sos = signal.butter(3, [300, 3000], btype='bandpass', fs=FS, output='sos')
filtered = signal.sosfiltfilt(sos, data)

# ---------- Plot Filtered Signal ----------
plt.figure(figsize=(18, 6))
start_sec = 0    # <-- enter start time in seconds
end_sec   = 0.01       # <-- enter end time in seconds

start_idx = int(start_sec * FS)
end_idx   = int(end_sec * FS)
plt.plot(t[start_idx:end_idx], filtered[start_idx:end_idx], linewidth=1, color='blue') # plot some  filtered data  

plt.title("Filtered Neural Signal (300–3000 Hz)", fontsize=14)
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Voltage (µV)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[4]:


plt.figure(figsize=(18, 6))

start_sec = 0    # <-- enter start time in seconds
end_sec   = 0.01       # <-- enter end time in seconds

start_idx = int(start_sec * FS)
end_idx   = int(end_sec * FS)
plt.plot(t[start_idx:end_idx], data[start_idx:end_idx], label="Raw Signal", color='red', linewidth=1)
plt.plot(t[start_idx:end_idx], filtered[start_idx:end_idx], label="Filtered Signal", color='blue', linewidth=1)

plt.title("Raw vs Filtered Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Voltage (µV)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# In[5]:


# ----- Plot Thresholds & Spikes on Filtered Signal -----
plt.figure(figsize=(18, 6))

# ---------------- Plotting Filtered signal (fixed window) ----------------
start_sec = 0    # <-- enter start time in seconds
end_sec   = 0.09 # <-- enter end time in seconds

start_idx = int(start_sec * FS)
end_idx   = int(end_sec * FS)

plt.plot(t[start_idx:end_idx], filtered[start_idx:end_idx], label="Filtered Signal",color='blue', linewidth=1)

# -------- Computing Noise & Thresholds --------
sigma = np.median(np.abs(filtered)) / 0.6745
K = 3.26
pos_threshold = K * sigma

# -------- Detecting Spikes with Refractory --------
candidate_indices = np.where(filtered > pos_threshold)[0]

refrac_ms = 1.0
refrac_samples = int(round(refrac_ms * FS / 1000.0))

pos_spike_indices = []
i = 0

while i < len(candidate_indices):
    
    # start of a group of close-by samples
    group_start = candidate_indices[i]
    group_end = group_start

    # extend group while next index is within refractory distance
    while (i < len(candidate_indices) - 1 and candidate_indices[i+1] - candidate_indices[i] < refrac_samples):
        i += 1
        group_end = candidate_indices[i]

    # now from group_start to group_end is one spike or blob
    local_region = filtered[group_start:group_end+1]       # all signal in this blob
    local_peak_rel = np.argmax(local_region)               # index of maximum value inside the spike of the blob
    precise_peak_idx = group_start + local_peak_rel        # index in full signal

    pos_spike_indices.append(precise_peak_idx)

    # move to next group
    i += 1

pos_spike_indices = np.array(pos_spike_indices)

# ---------------- Fix: plot only spikes in the window ----------------
spikes_in_window = pos_spike_indices[(pos_spike_indices >= start_idx) & (pos_spike_indices <  end_idx)]

# ---------------- Plot threshold line ----------------
plt.axhline(pos_threshold,color='green',linestyle='--',linewidth=1.5,label=f"Positive Threshold ({pos_threshold:.2f})")

# ---------------- Plot spike markers (only the window) ----------------
if len(spikes_in_window) > 0:
    plt.scatter(t[spikes_in_window], filtered[spikes_in_window],color='green', s=40, label="Positive Spikes")

plt.title("Filtered Neural Signal with Spike Thresholds", fontsize=14)
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Voltage (µV)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

print("Number of positive spikes detected:", len(pos_spike_indices))


# In[6]:


pos_spike_indices = np.sort(pos_spike_indices) 

# 1) EXTRACT WAVEFORM SNIPPETS AROUND EACH SPIKE
PRE_MS  = 0.8   # time in milliseconds before a spike
POST_MS = 2   # time in milliseconds after a spike

pre  = int(round(PRE_MS  * FS / 1000.0))     # Converting milliseconds to samples
post = int(round(POST_MS * FS / 1000.0))

valid_spikes = (pos_spike_indices - pre >= 0) & (pos_spike_indices + post < len(filtered)) # Removing the spikes that are too near the start or end of the signal
pos_spike_indices = pos_spike_indices[valid_spikes]

snips = np.stack([filtered[i-pre:i+post] for i in pos_spike_indices], axis=0)

# Baseline alignment
snips = snips - np.median(snips[:, :pre], axis=1, keepdims=True)


# In[7]:


# 1) PCA 
# We keep up to 3 components, but not more than the number of features
feats = PCA(n_components=min(3, snips.shape[1])).fit_transform(snips)


# In[8]:


# -----------------------------------------------------------
# 2D PCA SCATTER PLOT (PC1 vs PC2)
# -----------------------------------------------------------

fig = plt.figure(figsize=(8, 6))
plt.scatter(
    feats[:, 0],   # PC1
    feats[:, 1],   # PC2
    s=15,
    alpha=0.6
)

plt.title("PCA Feature Space (PC1 vs PC2)", fontsize=16, fontweight='bold')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()


# In[9]:


sil_scores = []    # For silhouette scores
k_values = list(range(2, min(8, snips.shape[0])))  # testing the number of clusters k = 2 .. 7

print("\nSilhouette scores:")
for k in k_values:
    km = KMeans(n_clusters=k, random_state=0, n_init=20).fit(feats)
    lbl = km.labels_
    
    # Silhouette score
    if len(np.unique(lbl)) > 1:
        sc = silhouette_score(feats, lbl)
    else:
        sc = -1
        
    sil_scores.append(sc)
    print(f"  k={k}: silhouette={sc:.4f}")

# --- SILHOUETTE PLOT ---
plt.figure(figsize=(10, 4))
plt.plot(k_values, sil_scores, marker='o', linewidth=2)
plt.title("Silhouette Score vs Number of Clusters (k)", fontsize=12, fontweight='bold')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(alpha=0.3)
plt.xticks(k_values)
plt.tight_layout()
plt.show()


# In[47]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ---------- clustering (your code) ----------
if snips.shape[0] >= 10:
    k = 3   # number of clusters
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=20)
    kmeans.fit(feats)
    labels = kmeans.labels_
else:
    # Fallback: simple 2-"neuron" split based on spike polarity
    labels = np.zeros_like(spike_idx, dtype=int)
    pos_set = set(pos_spike_indices.tolist())
    for j, i in enumerate(spike_idx):
        if i in pos_set:
            labels[j] = 1
    k = len(np.unique(labels))

clusters, counts = np.unique(labels, return_counts=True)

# sort clusters by size descending (largest -> 0)
order = clusters[np.argsort(-counts)]
remap = {old: new for new, old in enumerate(order)}

labels = np.array([remap[l] for l in labels])

# ---------- plot like your example ----------
color_map = {0: "red", 1: "blue", 2: "green"}  # match example: red, blue, green

plt.figure(figsize=(8, 6))

for c in range(k):
    idx = labels == c
    plt.scatter(
        feats[idx, 0], feats[idx, 1],
        s=10, alpha=0.85,
        color=color_map.get(c, "gray"),
        label=f"Cluster/Neuron {c}"
    )

plt.title("2D PCA Feature Space (PC1 vs PC2)", fontsize=20, fontweight="bold")
plt.xlabel("PC1", fontweight="bold", fontsize=20)
plt.ylabel("PC2", fontweight="bold", fontsize=20)
plt.grid(True, alpha=0.25)
plt.legend(loc="lower left", frameon=True)
plt.tight_layout()
plt.show()


# In[11]:


# -----------------------------------------------------------
# 3) PLOT MEAN ± STD FOR EACH NEURON (COLORED)
# -----------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

t_w = (np.arange(-pre, post) / FS) * 1000.0  
clusters = np.unique(labels)

# color mapping (same as PCA plot)
color_map = {
    0: "red",
    1: "blue",
    2: "green"
}

ncols = 3
nrows = int(np.ceil(len(clusters) / ncols))

plt.figure(figsize=(10, 3 * max(1, nrows)))

for plot_id, neuron_id in enumerate(clusters, start=1):
    idx = labels == neuron_id

    mean_wf = snips[idx].mean(axis=0)
    std_wf  = snips[idx].std(axis=0, ddof=1) if idx.sum() > 1 else np.zeros_like(mean_wf)

    color = color_map.get(int(neuron_id), "black")

    ax = plt.subplot(nrows, ncols, plot_id)

    ax.plot(
        t_w, mean_wf,
        color=color,
        lw=2,
        label=f"Mean (n={idx.sum()})"
    )

    ax.fill_between(
        t_w,
        mean_wf - std_wf,
        mean_wf + std_wf,
        color=color,
        alpha=0.3,
        label="±1 SD"
    )

    ax.axvline(0, ls="--", lw=1, color="k", alpha=0.6)

    ax.set_title(f"Neuron {neuron_id}", fontweight="bold")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.legend()

plt.tight_layout()
plt.show()


# In[12]:


num_neurons = len(clusters)
print(f"Estimated number of neurons: {num_neurons}")

# Count spikes per neuron
unique_ids, counts = np.unique(labels, return_counts=True)

total_spikes = len(pos_spike_indices)
print(f"Total spikes detected: {total_spikes}")

# Create a dictionary for safety (in case some neuron IDs are missing)
spike_count_per_neuron = dict(zip(unique_ids, counts))

for neuron_id in range(num_neurons):
    count = spike_count_per_neuron.get(neuron_id, 0)
    print(f"Neuron {neuron_id}: {count}")


# In[13]:


mat = sio.loadmat("sample_1.mat")

# ground-truth spike indices (from 'spike_times')
real_spike_idx = np.squeeze(mat["spike_times"][0, 0]).astype(int)

# ground-truth neuron labels (from 'spike_class')
spike_class = np.squeeze(mat["spike_class"][0, 0]).astype(int)

# --- Print total number of GT spikes ---
print("Total ground-truth spikes:", len(real_spike_idx))

# --- Print spikes per neuron class ---
unique, counts = np.unique(spike_class, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Neuron {u}: {c} spikes")


# In[14]:


matched_real = []
matched_detected = []

match_window_ms = 50   # e.g., 50 ms both before & after
match_window = int(match_window_ms * FS / 1000.0)

for i, real_t in enumerate(real_spike_idx):
    diff = pos_spike_indices - real_t
    valid = np.where(np.abs(diff) <= match_window)[0]   # <--- NEW
     
    if len(valid) > 0:
        best = valid[np.argmin(np.abs(diff[valid]))]    # pick closest in time
        if best not in matched_detected:
            matched_real.append(i)
            matched_detected.append(best)

matched_real = np.array(matched_real)
matched_detected = np.array(matched_detected)

print("Matched spikes:", len(matched_detected))


# In[15]:


n_detected = len(pos_spike_indices)  # By our code
n_real     = len(real_spike_idx)     # Ground Truth

TP = len(matched_detected)
FP = n_detected - TP
FN = n_real - TP

print("True Positives :", TP)
print("False Positives:", FP)
print("False Negatives:", FN)


# In[16]:


precision = TP / (TP + FP) 
print(f"Precision: {precision*100:.2f}%")


# In[17]:


recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
print(f"Recall: {recall*100:.2f}%")


# In[18]:


f1 = 2 * precision * recall / (precision + recall)
print(f"F1-score: {f1*100:.2f}%")


# In[19]:


# ground truth classes and predicted clusters for matched spikes only
gt_matched   = spike_class[matched_real]
cl_matched   = labels[matched_detected]

CM = confusion_matrix(gt_matched, cl_matched)
print("Confusion Matrix (rows = Ground Truth classes, cols =  Predicted Clusters):")
print(CM)


# In[20]:


print("\nCluster ↔ Ground Truth Class mapping:\n")

# build an array that stores, for each detected spike, the GT class (-1 = no match)
gt_for_detected = -np.ones(len(pos_spike_indices), dtype=int)
gt_for_detected[matched_detected] = spike_class[matched_real]

for cl in np.unique(labels):
    idx_in_cluster = np.where(labels == cl)[0]              # detected spikes in this cluster
    gt_in_cluster  = gt_for_detected[idx_in_cluster]        # their GT classes
    gt_in_cluster  = gt_in_cluster[gt_in_cluster != -1]     # only those that are matched

    if len(gt_in_cluster) == 0:
        print(f"Cluster {cl}: no matched spikes")
        continue

    unique_cls, counts = np.unique(gt_in_cluster, return_counts=True)

    print(f"Cluster {cl}:")
    for cls, cnt in zip(unique_cls, counts):
        pct = 100 * cnt / len(gt_in_cluster)
        print(f"  {cnt}/{len(gt_in_cluster)} spikes → GT class {cls} ({pct:.1f}%)")
    print()


# In[21]:


# 1) PCA
# Keep up to 3 components, but not more than the number of features
pca = PCA(n_components=min(3, snips.shape[1]))
feats = pca.fit_transform(snips)

# --- NEW: Variance explanation metrics ---
explained = pca.explained_variance_ratio_      # array like [PC1, PC2, PC3]

total_explained = explained.sum()              # total variance explained

print("\nPCA Variance Explained:")
print(f"  Total variance explained by retained PCs: {total_explained*100:.2f}%")

for i, var in enumerate(explained, start=1):
    print(f"  PC{i} explains: {var*100:.2f}% variance")

