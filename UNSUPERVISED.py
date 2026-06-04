from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =====================================================
# LOAD DATASET
# =====================================================

DATA_DIR = Path(__file__).resolve().parent
INPUT_FILE = DATA_DIR / "sm.csv"

print("Loading Dataset...")

df = pd.read_csv(
    INPUT_FILE,
    na_values=["NA", "?", "NULL", "null", "unknown", "Unknown"],
    keep_default_na=True,
)

print("Original Shape:", df.shape)

# =====================================================
# DATA CLEANING
# =====================================================

print("\nChecking Missing Values")
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

# Numeric columns
numeric_columns = [
    "followers",
    "following",
    "likes",
    "comments",
    "video_view_count",
    "caption_length",
    "hashtag_count",
    "posts_count",
    "post_hour"
]

numeric_columns = [col for col in numeric_columns if col in df.columns]

# Convert numeric columns
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing values using median
df[numeric_columns] = df[numeric_columns].fillna(
    df[numeric_columns].median()
)

# Encode media type
if "media_type" in df.columns:

    df["media_type"] = (
        df["media_type"]
        .astype(str)
        .str.lower()
        .str.strip()
        .replace({
            "img": "image",
            "image ": "image",
            "video ": "video"
        })
        .map({
            "image": 1,
            "video": 0
        })
    )

    df["media_type"] = df["media_type"].fillna(0)

print("\nShape After Cleaning:", df.shape)

# =====================================================
# FEATURE SELECTION
# =====================================================

# Demo 1 from PDF
features = [
    "likes",
    "comments"
]

print("\nSelected Features:")
print(features)

X = df[features]

# =====================================================
# FEATURE SCALING
# =====================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nData Standardized Successfully")

# =====================================================
# ELBOW METHOD
# =====================================================

print("\nRunning Elbow Method...")

wcss = []

for k in range(1, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    wcss,
    marker="o"
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.grid(True)

plt.savefig(DATA_DIR / "elbow_method.png")
plt.show()

# =====================================================
# SILHOUETTE ANALYSIS
# =====================================================

print("\nSilhouette Scores")

best_score = -1
best_k = 2

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    print(f"K = {k}  |  Score = {score:.4f}")

    if score > best_score:
        best_score = score
        best_k = k

print("\nBest K:", best_k)
print("Best Silhouette Score:", round(best_score, 4))

# =====================================================
# FINAL K-MEANS MODEL
# =====================================================

print("\nTraining Final K-Means Model")

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X_scaled)

# =====================================================
# CLUSTER COUNTS
# =====================================================

print("\nCluster Distribution")

print(df["cluster"].value_counts())

# =====================================================
# CLUSTER SUMMARY
# =====================================================

cluster_summary = (
    df.groupby("cluster")[features]
    .mean()
)

print("\nCluster Summary")

print(cluster_summary)

cluster_summary.to_csv(
    DATA_DIR / "cluster_summary.csv"
)

# =====================================================
# VISUALIZATION
# =====================================================

centroids = scaler.inverse_transform(
    kmeans.cluster_centers_
)

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    df["likes"],
    df["comments"],
    c=df["cluster"],
    cmap="viridis",
    alpha=0.7
)

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    color="red",
    s=300,
    marker="X",
    label="Centroids"
)

plt.xlabel("Likes")
plt.ylabel("Comments")

plt.title(
    "K-Means Clustering (Likes vs Comments)"
)

plt.legend()

plt.colorbar(scatter)

plt.tight_layout()

plt.savefig(
    DATA_DIR / "cluster_visualization.png"
)

plt.show()

# =====================================================
# SAVE OUTPUT
# =====================================================

df.to_csv(
    DATA_DIR / "social_media_clustered.csv",
    index=False
)

print("\nFiles Generated:")
print("1. elbow_method.png")
print("2. cluster_visualization.png")
print("3. cluster_summary.csv")
print("4. social_media_clustered.csv")

print("\nProject Completed Successfully")