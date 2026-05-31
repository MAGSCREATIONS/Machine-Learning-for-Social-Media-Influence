from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent
INPUT_FILE = DATA_DIR / "sm.csv"
OUTPUT_FILE = DATA_DIR / "sm_preprocessed.csv"

NA_VALUES = ["NA", "?", "NULL", "null", "unknown", "Unknown"]
NUMERIC_COLUMNS = [
    "followers",
    "following",
    "likes",
    "comments",
    "video_view_count",
    "caption_length",
    "hashtag_count",
    "posts_count",
    "post_hour",
    "post_day_of_week",
]

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
    video_median = df.loc[df["video_view_count"] > 0, "video_view_count"].median()
    if np.isnan(video_median):
        video_median = df["video_view_count"].median()
    df.loc[df["video_view_count"] == 0, "video_view_count"] = video_median

# Drop category_name column completely if present
df = df.drop(columns=["category_name"], errors="ignore")

# Impute missing values
numeric_medians = df[current_numeric_columns].median()
df[current_numeric_columns] = df[current_numeric_columns].fillna(numeric_medians)

if "media_type" in df.columns:
    media_mode = df["media_type"].mode(dropna=True)
    df["media_type"] = df["media_type"].fillna(media_mode.iloc[0] if not media_mode.empty else "unknown")

# Create ratio features with clear names
if "video_view_count" in df.columns:
    if "likes" in df.columns:
        df["views_per_like"] = np.where(
            df["likes"] != 0,
            df["video_view_count"] / df["likes"],
            np.nan,
        )
    if "comments" in df.columns:
        df["views_per_comment"] = np.where(
            df["comments"] != 0,
            df["video_view_count"] / df["comments"],
            np.nan,
        )
    if "followers" in df.columns:
        df["views_per_follower"] = np.where(
            df["video_view_count"] != 0,
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
    "views_per_like",
    "views_per_comment",
    "views_per_follower",
    "views_per_hashtag",
]
for col in ratio_cols:
    if col in df.columns:
        df[col] = df[col].round(2)

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

# Standardize numeric features
df_standardized = df.copy()
numeric_features = df_standardized.select_dtypes(include=[np.number]).columns

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

# Fit scaler on train data and transform both
scaler_train = StandardScaler()
df_train_std[numeric_features] = scaler_train.fit_transform(df_train_std[numeric_features])
df_test_std[numeric_features] = scaler_train.transform(df_test_std[numeric_features])

TRAINING_STD_FILE = DATA_DIR / "sm_training_standardized.csv"
TEST_STD_FILE = DATA_DIR / "sm_test_standardized.csv"
df_train_std.to_csv(TRAINING_STD_FILE, index=False)
df_test_std.to_csv(TEST_STD_FILE, index=False)
print(f"Saved standardized training data to: {TRAINING_STD_FILE}")
print(f"Saved standardized testing data to: {TEST_STD_FILE}")

print(df.to_string())
print("Saving preprocessed data to:", OUTPUT_FILE)

hashtag_counts = df["hashtag_count"].value_counts().nlargest(10)

# Pie chart: media type share
media_counts = (
    df["media_type"]
    .map({1: "image", 0: "video"})
    .value_counts()
)

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
sns.barplot(x=hashtag_counts.index.astype(str), y=hashtag_counts.values, palette="magma")
plt.title("Top 10 hashtag_count values")
plt.xlabel("Hashtag count")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
