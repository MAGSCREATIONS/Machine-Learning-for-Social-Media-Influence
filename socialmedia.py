from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans

DATA_DIR = Path(__file__).resolve().parent
INPUT_FILE = DATA_DIR / "sm.csv"
OUTPUT_FILE = DATA_DIR / "sm_preprocessed.csv"

NA_VALUES = ["NA", "?", "NULL", "null", "unknown", "Unknown"]
NUMERIC_COLUMNS = [
    "followers",
    "following",
    "likes",
    "video_view_count",
    "caption_length",
    "hashtag_count",
    "posts_count",
    "post_hour",
    "post_day_of_week",
]

TARGET_PERCENTILE = 0.90
KNN_NEIGHBORS = 5

print("Loading data from:", INPUT_FILE)
df = pd.read_csv(INPUT_FILE, na_values=NA_VALUES, keep_default_na=True, skipinitialspace=True)

print("Original shape:", df.shape)
if df.shape[1] > 2:
    df = df.iloc[:, :-2]
    print("Dropped last 2 columns, new shape:", df.shape)
print(df.isnull().sum())

# Keep only numeric columns that exist after dropping columns
current_numeric_columns = [col for col in NUMERIC_COLUMNS if col in df.columns]

# Strip whitespace and normalize text columns
text_columns = df.select_dtypes(include=["object", "string"]).columns
for col in text_columns:
    df[col] = df[col].astype("string").str.strip()

# Normalize categorical labels
if "media_type" in df.columns:
    df["media_type"] = (
        df["media_type"]
        .astype("string")
        .str.lower()
        .str.strip()
        .replace({"img": "image", "video ": "video", "image ": "image"})
        .map({"image": 1, "video": 0})
    )

# Convert numeric columns and coerce invalid values to NaN
for col in current_numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop exact duplicates
df = df.drop_duplicates()

if "video_view_count" in df.columns:
    if "likes" in df.columns:
        ratio_mask = (df["likes"] > 0) & (df["video_view_count"] > 0)
        if ratio_mask.any():
            view_like_ratio = (
                df.loc[ratio_mask, "video_view_count"] / df.loc[ratio_mask, "likes"]
            ).median()
            print(f"Estimated view-per-like ratio from nonzero likes: {view_like_ratio:.2f}")
            estimate_mask = (df["video_view_count"] <= 0) & (df["likes"] > 0)
            df.loc[estimate_mask, "video_view_count"] = (
                df.loc[estimate_mask, "likes"] * view_like_ratio
            ).round().astype(int)

    zero_view_mask = df["video_view_count"] <= 0
    if zero_view_mask.any():
        video_median = df.loc[df["video_view_count"] > 0, "video_view_count"].median()
        if np.isnan(video_median):
            video_median = df["video_view_count"].median()
        df.loc[zero_view_mask, "video_view_count"] = video_median

# Drop category_name and comments columns completely if present
# Comments are not used for downstream outputs.
df = df.drop(columns=["category_name", "comments"], errors="ignore")

# Refresh numeric columns after dropping comments
current_numeric_columns = [col for col in NUMERIC_COLUMNS if col in df.columns]

# Impute missing values
numeric_medians = df[current_numeric_columns].median()
df[current_numeric_columns] = df[current_numeric_columns].fillna(numeric_medians)

if "media_type" in df.columns:
    media_mode = df["media_type"].mode(dropna=True)
    df["media_type"] = df["media_type"].fillna(media_mode.iloc[0] if not media_mode.empty else "unknown")

# Create ratio features with clear names
if "video_view_count" in df.columns:
    if "followers" in df.columns:
        df["views_per_follower"] = np.where(
            df["followers"] > 0,
            df["video_view_count"] / df["followers"],
            np.nan,
        )
    if "hashtag_count" in df.columns:
        df["views_per_hashtag"] = np.where(
            df["hashtag_count"] != 0,
            df["video_view_count"] / df["hashtag_count"],
            np.nan,
        )

# Round ratio features to two decimal places
ratio_cols = [
    "views_per_follower",
    "views_per_hashtag",
]
for col in ratio_cols:
    if col in df.columns:
        df[col] = df[col].round(2)

# Fill any ratio NaNs with 0 so KNN can train cleanly
for col in ratio_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Create virality features and target label
if {"video_view_count", "likes", "followers"}.issubset(df.columns):
    df["like_rate"] = np.where(
        df["followers"] > 0,
        df["likes"] / df["followers"],
        0,
    )
    df["virality_score"] = (
        0.65 * np.log1p(df["views_per_follower"])
        + 0.35 * np.log1p(df["like_rate"])
    )
    virality_threshold = df["virality_score"].quantile(TARGET_PERCENTILE)
    df["target"] = (df["virality_score"] >= virality_threshold).astype(int)
    target_percent = round((1.0 - TARGET_PERCENTILE) * 100, 1)
    print(f"Target threshold for top {target_percent}%: {virality_threshold:.4f}")
    print(df["target"].value_counts(normalize=True).to_string())

    # Heatmap for virality-related features
    feature_heatmap_cols = [
        c for c in [
            "like_rate",
            "views_per_follower",
            "views_per_hashtag",
            "virality_score",
            "target",
        ]
        if c in df.columns
    ]
    if feature_heatmap_cols:
        corr_matrix = df[feature_heatmap_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5,
        )
        plt.title("Virality feature correlation heatmap")
        plt.tight_layout()
        heatmap_file = DATA_DIR / "virality_feature_heatmap.png"
        plt.savefig(heatmap_file)
        plt.show()

# Round integer-like numeric columns to ints
for col in NUMERIC_COLUMNS:
    if col in df.columns:
        df[col] = df[col].round().astype(int)

print("Preprocessed shape:", df.shape)
print(df.isnull().sum())

print("Saving preprocessed data to:", OUTPUT_FILE)
df.to_csv(OUTPUT_FILE, index=False)

# Split preprocessed data into 80% training and 20% test
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

TRAINING_FILE = DATA_DIR / "sm_training.csv"
TEST_FILE = DATA_DIR / "sm_test.csv"
df_train.to_csv(TRAINING_FILE, index=False)
df_test.to_csv(TEST_FILE, index=False)
print(f"Saved training data ({len(df_train)} rows) to: {TRAINING_FILE}")
print(f"Saved testing data ({len(df_test)} rows) to: {TEST_FILE}")

# Standardize numeric features (excluding target label)
df_standardized = df.copy()
numeric_features = [
    col
    for col in df_standardized.select_dtypes(include=[np.number]).columns
    if col != "target"
]

# Replace inf with NaN and then fill NaN values with median
for col in numeric_features:
    df_standardized[col] = df_standardized[col].replace([np.inf, -np.inf], np.nan)
    if df_standardized[col].isnull().any():
        df_standardized[col] = df_standardized[col].fillna(df_standardized[col].median())

scaler = StandardScaler()
df_standardized[numeric_features] = scaler.fit_transform(df_standardized[numeric_features])

STANDARDIZED_FILE = DATA_DIR / "sm_standardized.csv"
print("Saving standardized data to:", STANDARDIZED_FILE)
df_standardized.to_csv(STANDARDIZED_FILE, index=False)

# Standardize train and test sets separately using the same scaler
df_train_std = df_train.copy()
df_test_std = df_test.copy()

# Replace inf with NaN and fill for both train and test
for col in numeric_features:
    df_train_std[col] = df_train_std[col].replace([np.inf, -np.inf], np.nan)
    df_test_std[col] = df_test_std[col].replace([np.inf, -np.inf], np.nan)

    if df_train_std[col].isnull().any():
        df_train_std[col] = df_train_std[col].fillna(df_train_std[col].median())
    if df_test_std[col].isnull().any():
        df_test_std[col] = df_test_std[col].fillna(df_test_std[col].median())

# Fit scaler on train data and transform both sets
scaler_train = StandardScaler()
df_train_std[numeric_features] = scaler_train.fit_transform(df_train_std[numeric_features])
df_test_std[numeric_features] = scaler_train.transform(df_test_std[numeric_features])

TRAINING_STD_FILE = DATA_DIR / "sm_training_standardized.csv"
TEST_STD_FILE = DATA_DIR / "sm_test_standardized.csv"
df_train_std.to_csv(TRAINING_STD_FILE, index=False)
df_test_std.to_csv(TEST_STD_FILE, index=False)
print(f"Saved standardized training data to: {TRAINING_STD_FILE}")
print(f"Saved standardized testing data to: {TEST_STD_FILE}")

# KMeans clustering analysis on standardized training data (use inertia elbow only)
cluster_features = [col for col in numeric_features if col != "virality_score"]
if cluster_features:
    X_clust = df_train_std[cluster_features]
    ks = list(range(2, 11))
    inertias = []
    for k in ks:
        kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_tmp.fit(X_clust)
        inertias.append(kmeans_tmp.inertia_)

    # Geometric elbow detection: choose k with max distance to line between first and last points
    x = np.array(ks)
    y = np.array(inertias)
    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]
    denom = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    if denom == 0:
        best_k_inertia = ks[0]
    else:
        distances = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom
        best_k_inertia = ks[int(np.argmax(distances))]

    print(f"KMeans: best k by inertia elbow: {best_k_inertia}")

    plt.figure(figsize=(8, 4))
    plt.plot(ks, inertias, marker='o')
    plt.title('KMeans inertia (elbow)')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.xticks(ks)
    plt.grid(True)
    elbow_file = DATA_DIR / 'kmeans_elbow.png'
    plt.tight_layout()
    plt.savefig(elbow_file)
    plt.show()

    # Fit final KMeans and attach cluster labels to standardized train/test
    final_kmeans = KMeans(n_clusters=best_k_inertia, random_state=42, n_init=10)
    final_kmeans.fit(X_clust)
    df_train_std['cluster'] = final_kmeans.labels_
    df_test_std['cluster'] = final_kmeans.predict(df_test_std[cluster_features])
    print('KMeans cluster counts (train):')
    print(df_train_std['cluster'].value_counts().to_string())

# KNN model training and validation on the standardized 80/20 split
feature_columns = [col for col in numeric_features if col != "virality_score"]
if "target" in df_train_std.columns and feature_columns:
    X_train = df_train_std[feature_columns]
    y_train = df_train_std["target"].astype(int)
    X_test = df_test_std[feature_columns]
    y_test = df_test_std["target"].astype(int)

    print("Training target balance:\n", y_train.value_counts(normalize=True).to_string())
    print("Testing target balance:\n", y_test.value_counts(normalize=True).to_string())

    # Plot K versus test accuracy to choose a good neighbor count
    k_values = list(range(1, 21))
    k_accuracies = []
    for k in k_values:
        k_model = KNeighborsClassifier(n_neighbors=k)
        k_model.fit(X_train, y_train)
        k_accuracies.append(accuracy_score(y_test, k_model.predict(X_test)))

    best_k = k_values[int(np.argmax(k_accuracies))]
    best_k_accuracy = max(k_accuracies)
    print(f"Best k by test accuracy: {best_k} ({best_k_accuracy:.4f})")

    plt.figure(figsize=(8, 4))
    plt.plot(k_values, k_accuracies, marker="o", linestyle="-")
    plt.title("KNN test accuracy by k")
    plt.xlabel("k (neighbors)")
    plt.ylabel("Test accuracy")
    plt.xticks(k_values)
    plt.grid(True)
    plt.tight_layout()
    k_curve_file = DATA_DIR / "knn_k_vs_accuracy.png"
    plt.savefig(k_curve_file)
    plt.show()

    # Use the best k found from the validation loop above
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print(f"KNN test accuracy: {accuracy:.4f}")
    print("KNN classification report:\n", report)
    print("KNN confusion matrix:\n", cm)

    # Cross-validation on training set
    cv_results = cross_validate(
        knn,
        X_train,
        y_train,
        cv=5,
        scoring=["accuracy", "precision_macro", "recall_macro"],
        return_train_score=False,
    )
    print("KNN cross-validation results:")
    print(f"  accuracy mean: {cv_results['test_accuracy'].mean():.4f}")
    print(f"  precision mean: {cv_results['test_precision_macro'].mean():.4f}")
    print(f"  recall mean: {cv_results['test_recall_macro'].mean():.4f}")
    print(f"  accuracy std: {cv_results['test_accuracy'].std():.4f}")

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("KNN confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_file = DATA_DIR / "knn_confusion_matrix.png"
    plt.savefig(cm_file)
    plt.show()

    # Learn a data-driven virality formula using logistic regression
    lr_features = [
        f for f in ["views_per_follower", "like_rate", "views_per_hashtag"]
        if f in X_train.columns
    ]
    if lr_features:
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train[lr_features], y_train)
        lr_pred = lr.predict(X_test[lr_features])
        print("\nLogistic regression on current virality features:")
        for name, coef in zip(lr_features, lr.coef_[0]):
            print(f"  {name}: {coef:.4f}")
        print(f"  intercept: {lr.intercept_[0]:.4f}")
        print("LR accuracy:", round(accuracy_score(y_test, lr_pred), 4))
        print("LR classification report:\n", classification_report(y_test, lr_pred, digits=4))

print("Preprocessed data saved and KNN validation complete.")

# Pie chart: media type share
media_counts = (
    df["media_type"]
    .map({1: "image", 0: "video"})
    .value_counts()
)

hashtag_counts = df["hashtag_count"].value_counts().nlargest(10)

plt.figure(figsize=(6, 6))
media_counts.plot.pie(
    autopct="%1.1f%%",
    startangle=90,
    counterclock=False,
    colors=sns.color_palette("pastel"),
)
plt.title("Media type distribution")
plt.ylabel("")
plt.tight_layout()
plt.show()

# Finding top 10 hashtag_count values
plt.figure(figsize=(10, 6))
plt.bar(
    hashtag_counts.index.astype(str),
    hashtag_counts.values,
    color=sns.color_palette("magma", len(hashtag_counts)),
)
plt.title("Top 10 hashtag_count values")
plt.xlabel("Hashtag count")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
