"""
LESSON 6 — CLUSTERING: Customer Segmentation for Marketing
==========================================================
Real-life problem: A shopping mall has purchase data but NO labels —
nobody tells you which customers are "premium" or "bargain hunters".
Find natural groups yourself, so marketing can target each differently.

This is UNSUPERVISED learning: there is no y! The model finds structure
on its own. (You've met KMeans before — now we use it properly.)

NEW IDEAS:
  - Choosing k with the elbow method + silhouette score
  - Interpreting clusters back in business terms (the important part!)
  - PCA to draw a 4-dimensional dataset in 2D
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save plots to files instead of opening windows
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from pathlib import Path

HERE = Path(__file__).parent

# =====================================================================
# 1. DATA — 600 mall customers, 3 hidden types (model doesn't know!)
# =====================================================================
rng = np.random.default_rng(3)

def make_group(n, income, spending, visits, basket):
    return pd.DataFrame({
        "annual_income_k": rng.normal(income, 8, n),
        "spending_score": rng.normal(spending, 10, n).clip(1, 100),
        "visits_per_month": rng.normal(visits, 1.5, n).clip(0.2, 30),
        "avg_basket": rng.normal(basket, 15, n).clip(5, 400),
    })

df = pd.concat([
    make_group(200, income=35, spending=40, visits=6, basket=30),   # budget regulars
    make_group(200, income=90, spending=80, visits=3, basket=180),  # premium shoppers
    make_group(200, income=60, spending=25, visits=1, basket=60),   # rare visitors
], ignore_index=True).sample(frac=1, random_state=1).reset_index(drop=True)

print("Customer data — NO labels, just behavior:")
print(df.head(), "\n")

# Scaling is CRITICAL for KMeans: it measures distances, and without
# scaling, "avg_basket" (5-400) would dominate "visits" (0-30).
X = StandardScaler().fit_transform(df)

# =====================================================================
# 2. HOW MANY CLUSTERS? We don't know — let the data tell us.
# =====================================================================
# Elbow method: inertia = how tightly points hug their cluster center.
# It always drops as k grows; look for the "elbow" where it stops
# dropping fast. Silhouette: how well-separated clusters are (max 1.0).
print("k    inertia    silhouette")
inertias = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    sil = silhouette_score(X, km.labels_)
    inertias.append(km.inertia_)
    marker = "  <-- best" if k == 3 else ""
    print(f"{k}    {km.inertia_:7.0f}    {sil:.3f}{marker}")

plt.figure(figsize=(6, 4))
plt.plot(range(2, 9), inertias, "o-")
plt.xlabel("number of clusters k"); plt.ylabel("inertia")
plt.title("Elbow method: the bend is at k=3")
plt.savefig(HERE / "elbow.png", dpi=120, bbox_inches="tight")
print(f"\nSaved elbow plot -> {HERE / 'elbow.png'}\n")

# =====================================================================
# 3. FIT THE FINAL MODEL AND *INTERPRET* THE SEGMENTS
# =====================================================================
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["segment"] = kmeans.fit_predict(X)

# The single most useful clustering output: average profile per segment.
profile = df.groupby("segment").mean().round(1)
profile["size"] = df["segment"].value_counts().sort_index()
print("Segment profiles (this is what you show the marketing team):")
print(profile, "\n")

print("A human reads that table and names the segments, e.g.:")
print('  high income + high spending  -> "Premium: promote luxury brands"')
print('  low income + frequent visits -> "Regulars: loyalty card program"')
print('  low visits                   -> "Dormant: win-back email campaign"\n')

# =====================================================================
# 4. VISUALIZE — PCA squeezes 4 features into 2 for plotting
# =====================================================================
coords = PCA(n_components=2).fit_transform(X)
plt.figure(figsize=(6, 5))
for seg in sorted(df["segment"].unique()):
    m = df["segment"] == seg
    plt.scatter(coords[m, 0], coords[m, 1], s=12, label=f"segment {seg}")
plt.legend(); plt.title("Customer segments (PCA projection)")
plt.xlabel("PC 1"); plt.ylabel("PC 2")
plt.savefig(HERE / "segments.png", dpi=120, bbox_inches="tight")
print(f"Saved cluster plot -> {HERE / 'segments.png'}")

# =====================================================================
# 5. ASSIGN A NEW CUSTOMER TO A SEGMENT
# =====================================================================
# (In production you'd save the scaler+kmeans together in a Pipeline.)

# =====================================================================
# TRY IT YOURSELF:
#  - Open elbow.png and segments.png!
#  - Remove StandardScaler and re-run. Watch silhouette scores collapse.
#  - Change one group's parameters to overlap another (e.g. income=88,
#    spending=75). Clusters blur — real data is rarely this clean.
# =====================================================================
