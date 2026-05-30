from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path(__file__).resolve().parent / "plaksha"
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
print(df.isnull().sum())

# Strip whitespace and normalize text columns
text_columns = df.select_dtypes(include=["object", "string"]).columns
for col in text_columns:
    df[col] = df[col].astype("string").str.strip()

# Normalize categorical labels
if "media_type" in df.columns:
    df["media_type"] = (
        df["media_type"]
        .str.lower()
        .replace({"img": "image", "video ": "video", "image ": "image"})
    )

if "category_name" in df.columns:
    df["category_name"] = (
        df["category_name"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace({"unknown": np.nan})
    )

# Convert numeric columns and coerce invalid values to NaN
for col in NUMERIC_COLUMNS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop exact duplicates
df = df.drop_duplicates()

# Impute missing values
numeric_medians = df[NUMERIC_COLUMNS].median()
df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].fillna(numeric_medians)

if "media_type" in df.columns:
    media_mode = df["media_type"].mode(dropna=True)
    df["media_type"] = df["media_type"].fillna(media_mode.iloc[0] if not media_mode.empty else "unknown")

if "category_name" in df.columns:
    df["category_name"] = df["category_name"].fillna("unknown")

# Round integer-like numeric columns to ints
for col in NUMERIC_COLUMNS:
    if col in df.columns:
        df[col] = df[col].round().astype(int)

print("Preprocessed shape:", df.shape)
print(df.isnull().sum())

print("Saving preprocessed data to:", OUTPUT_FILE)
df.to_csv(OUTPUT_FILE, index=False)

print(df.to_string())
print("Saving preprocessed data to:", OUTPUT_FILE)

hashtag_counts = df["hashtag_count"].value_counts().nlargest(10)

# Bar chart: top 10 category names
top_categories = df["category_name"].value_counts().nlargest(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_categories.values, y=top_categories.index, palette="viridis")
plt.title("Top 10 category names")
plt.xlabel("Count")
plt.ylabel("Category")
plt.tight_layout()
plt.show()
# Pie chart: media type share
media_counts = (
    df["media_type"]
    .astype(str)
    .str.strip()
    .replace({"Img": "image", "VIDEO": "video", "video ": "video", "image ": "image"})
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

#finding top 10 hashtag_count values

plt.figure(figsize=(10, 6))
sns.barplot(x=hashtag_counts.index.astype(str), y=hashtag_counts.values, palette="magma")
plt.title("Top 10 hashtag_count values")
plt.xlabel("Hashtag count")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
