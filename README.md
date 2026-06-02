# Machine-Learning-for-Social-Media-Influence

## What This Script Does

Think of the dataset as a messy notebook full of information. This script acts as a helper that cleans, organizes, and prepares the data before it is used for analysis and machine-learning models.

The pipeline performs:

- Data cleaning
- Feature engineering
- Virality score generation
- Target label creation
- Data standardization
- Dataset splitting
- K-Means clustering
- KNN model training
- Model evaluation
- Cross-validation
- Logistic Regression analysis
- Data visualization

---

## Data Loading

- Reads the dataset from `sm.csv`.
- Loads the data into a Pandas DataFrame for processing.
- Treats the following values as missing data:
  - `NA`
  - `?`
  - `NULL`
  - `null`
  - `unknown`
  - `Unknown`

---

## Data Cleaning

### Removing Unnecessary Columns

- Drops the last two columns of the dataset if they exist.
- Removes the `category_name` column because it is not required for analysis.
- Removes the `comments` column because it is not used in feature engineering or model training.

### Fixing Text Data

- Removes extra spaces from text fields.
- Standardizes categorical labels to ensure consistency.

Examples:

- `Img` → `image`
- `image ` → `image`
- `video ` → `video`

### Standardizing Media Type

The cleaned media types are converted into numerical values:

- Image → `1`
- Video → `0`

### Fixing Numeric Data

- Converts numeric values stored as text into proper numeric format.
- Invalid values are automatically converted into missing values (`NaN`).

### Removing Duplicate Records

- Identical rows are removed so that each record appears only once.

---

## Handling Missing Values

### Numeric Columns

Missing numerical values are replaced using the median of the corresponding column.

This applies to columns such as:

- followers
- following
- likes
- video_view_count
- caption_length
- hashtag_count
- posts_count
- post_hour
- post_day_of_week

### Categorical Columns

Missing values in the `media_type` column are filled using the most common category (mode).

---

## Handling Video View Counts

Some records may contain a video view count of zero.

To avoid unrealistic values affecting analysis:

- A view-per-like ratio is estimated from valid records.
- Missing or zero view counts are estimated where possible.
- Remaining zero values are replaced using the median view count.

---

## Feature Engineering

The script creates additional engagement-related features.

### Views Per Follower

Measures how many views are obtained relative to the creator's followers.

`views_per_follower = video_view_count / followers`

### Views Per Hashtag

Measures how many views are obtained per hashtag used.

`views_per_hashtag = video_view_count / hashtag_count`

### Feature Formatting

- Ratio features are rounded to two decimal places.
- Missing ratio values are replaced with 0.
- Numeric columns are converted to integer format where appropriate.

---

## Creating the Virality Target

The script creates a target variable that identifies whether a post is viral.

### Like Rate

Measures how many likes a post receives relative to the creator's follower count.

`like_rate = likes / followers`

### Virality Score

A combined score is calculated using both view rate and like rate.

`virality_score = 0.65 × log(1 + views_per_follower) + 0.35 × log(1 + like_rate)`

The logarithmic transformation reduces the impact of extreme values and creates a more balanced scoring system.

### Target Label

The top 10% of posts based on virality score are labeled as viral.

- Viral Post = `1`
- Non-Viral Post = `0`

---

## Correlation Analysis

The script generates a correlation heatmap for important virality-related features.

Features included:

- like_rate
- views_per_follower
- views_per_hashtag
- virality_score
- target

Output file:

`virality_feature_heatmap.png`

---

## Data Export

The script generates the following files:

- `sm_preprocessed.csv`
- `sm_training.csv`
- `sm_test.csv`
- `sm_standardized.csv`
- `sm_training_standardized.csv`
- `sm_test_standardized.csv`

---

## Dataset Splitting

The dataset is divided into:

- 80% Training Data
- 20% Testing Data

---

## Data Standardization

The project uses Scikit-Learn's `StandardScaler` to standardize numerical features.

Formula:

`z = (x - mean) / standard deviation`

After standardization:

- Mean ≈ 0
- Standard Deviation ≈ 1

This improves the performance of distance-based algorithms such as KNN and K-Means.

---

## K-Means Clustering

The project performs unsupervised learning using K-Means clustering.

### Purpose

Groups similar social media posts together based on their numerical characteristics.

### Cluster Selection

- Tests values of `k` from 2 to 10.
- Calculates inertia for each value.
- Uses the elbow method to automatically select the optimal number of clusters.

### Output

`kmeans_elbow.png`

### Final Clustering

The selected K-Means model is fitted on the standardized training data and cluster labels are assigned to both training and testing datasets.

---

## K-Nearest Neighbors (KNN) Classification

The project uses the KNN algorithm to predict whether a social media post is viral.

### Automatic K Selection

The script tests:

`k = 1` to `k = 20`

The value producing the highest test accuracy is selected automatically.

### Output

`knn_k_vs_accuracy.png`

### Training Process

The model is trained using the standardized training dataset and predicts:

- Viral Posts (`1`)
- Non-Viral Posts (`0`)

---

## Model Evaluation

The trained KNN model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Output file:

`knn_confusion_matrix.png`

---

## Cross-Validation

To verify model consistency, 5-fold cross-validation is performed.

Metrics calculated:

- Mean Accuracy
- Mean Precision
- Mean Recall
- Accuracy Standard Deviation

---

## Logistic Regression Analysis

A Logistic Regression model is trained to understand which features contribute most to virality.

### Features Used

- views_per_follower
- like_rate
- views_per_hashtag

### Output

The script prints:

- Feature coefficients
- Model intercept
- Accuracy
- Classification report

---

## Data Visualization

The script generates the following visualizations:

1. Virality Feature Correlation Heatmap
2. K-Means Elbow Plot
3. KNN Accuracy vs K Plot
4. KNN Confusion Matrix
5. Media Type Distribution Pie Chart
6. Top 10 Hashtag Count Bar Chart

---

## Generated Output Files

### Processed Datasets

- `sm_preprocessed.csv`
- `sm_training.csv`
- `sm_test.csv`
- `sm_standardized.csv`
- `sm_training_standardized.csv`
- `sm_test_standardized.csv`

### Visualizations

- `virality_feature_heatmap.png`
- `kmeans_elbow.png`
- `knn_k_vs_accuracy.png`
- `knn_confusion_matrix.png`

---

## Libraries Used

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- pathlib

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Project Workflow

```text
Raw Dataset (sm.csv)
        │
        ▼
Data Cleaning
        │
        ▼
Missing Value Handling
        │
        ▼
Feature Engineering
        │
        ▼
Virality Score Generation
        │
        ▼
Target Label Creation
        │
        ▼
Train-Test Split (80/20)
        │
        ▼
Standardization
        │
 ┌──────┴─────────┐
 ▼                ▼
K-Means       KNN Classifier
Clustering    Training
 ▼                ▼
Cluster      Prediction
Analysis     Evaluation
                  ▼
         Cross-Validation
                  ▼
        Logistic Regression
                  ▼
            Visualizations
```

---

## Conclusion

This project builds a complete machine-learning pipeline for social media influence analysis. It cleans and preprocesses raw data, creates engagement-based features, generates a virality score, and predicts whether a post is viral. The workflow combines K-Means clustering, KNN classification, Logistic Regression, and visualizations to provide both predictive performance and insights into social media engagement patterns.
