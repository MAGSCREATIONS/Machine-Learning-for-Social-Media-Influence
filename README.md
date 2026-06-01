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
- KNN model training
- Model evaluation
- Cross-validation
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

- The script calculates the median of valid video view counts.
- Any zero value is replaced with this median value.

---

## Feature Engineering

The script creates additional engagement-related features that may improve machine-learning performance.

### Views Per Like

Measures how many views are obtained for each like.

`views_per_like = video_view_count / likes`

### Views Per Follower

Measures how many views are obtained relative to the creator's followers.

`views_per_follower = video_view_count / followers`

### Views Per Hashtag

Measures how many views are obtained per hashtag used.

`views_per_hashtag = video_view_count / hashtag_count`

### Feature Formatting

- Ratio features are rounded to two decimal places.
- Numeric columns are converted to integer format where appropriate.

---

## Creating the Virality Target

To train a machine-learning model, the script creates a target variable that identifies whether a post is viral.

### View Rate

Measures how many views a post receives relative to the creator's follower count.

`view_rate = video_view_count / followers`

### Like Rate

Measures how many likes a post receives relative to the creator's follower count.

`like_rate = likes / followers`

### Virality Score

A combined score is calculated using both view rate and like rate.

`virality_score = 0.65 × log(1 + view_rate) + 0.35 × log(1 + like_rate)`

The weights were chosen manually to give greater importance to audience reach while still considering user engagement.

- 65% weight is assigned to view rate.
- 35% weight is assigned to like rate.

This means that posts reaching a larger audience receive a higher virality score, while likes contribute as a secondary measure of engagement.

The logarithmic transformation (`log(1 + x)`) is used to reduce the effect of extremely large values and create a more balanced scoring system.

### Target Label

The script identifies the top 10% of posts based on their virality score.

- Viral Post = `1`
- Non-Viral Post = `0`

This target variable is used for machine-learning classification.

---

## Correlation Analysis

The script generates a correlation heatmap for important virality-related features.

Features included:

- view_rate
- like_rate
- views_per_like
- views_per_follower
- views_per_hashtag
- virality_score
- target

The heatmap helps visualize relationships between features and identify which variables are most strongly associated with virality.

Output file:

`virality_feature_heatmap.png`

---

## Data Export

The script generates several output files:

- `sm_preprocessed.csv`
- `sm_training.csv`
- `sm_test.csv`
- `sm_standardized.csv`
- `sm_training_standardized.csv`
- `sm_test_standardized.csv`

These files are used for machine-learning training, testing, and evaluation.

---

## Dataset Splitting

The dataset is divided into:

- 80% Training Data
- 20% Testing Data

The training dataset is used for model training, while the testing dataset is used for model evaluation.

---

## Data Standardization

Machine-learning algorithms such as K-Nearest Neighbors perform better when all features are on a similar scale.

The script applies Standardization using Scikit-Learn's `StandardScaler`.

### What Standardization Does

For each numerical feature:

1. The mean is subtracted.
2. The result is divided by the standard deviation.

Formula:

`z = (x - mean) / standard deviation`

After standardization:

- Mean becomes approximately 0.
- Standard deviation becomes approximately 1.

---

## K-Nearest Neighbors (KNN) Classification

The project uses the K-Nearest Neighbors (KNN) algorithm to predict whether a social media post is viral.

### Why KNN?

KNN classifies a new data point by examining the classes of its nearest neighbors.

In this project:

- Number of Neighbors (K) = 5
- Viral Posts = 1
- Non-Viral Posts = 0

### Training Process

The model is trained using the standardized training dataset.

The target variable is:

`target`

---

## Model Evaluation

The trained KNN model is evaluated using the testing dataset.

The following performance metrics are calculated:

### Accuracy

Measures the percentage of correct predictions.

### Precision

Measures how many posts predicted as viral were actually viral.

### Recall

Measures how many actual viral posts were correctly identified.

### Classification Report

Provides a detailed summary of:

- Precision
- Recall
- F1-Score
- Support

for each class.

### Confusion Matrix

A confusion matrix is generated to compare:

- Actual labels
- Predicted labels

Output file:

`knn_confusion_matrix.png`

---

## Cross-Validation

To verify that the model performs consistently, 5-fold cross-validation is performed on the training data.

The script calculates:

- Mean Accuracy
- Mean Precision
- Mean Recall
- Accuracy Standard Deviation

Cross-validation provides a more reliable estimate of model performance than a single train-test split.

---

## Data Visualization

The script generates the following visualizations:

### 1. Virality Feature Correlation Heatmap

Shows relationships between virality-related features.

### 2. KNN Confusion Matrix

Shows model prediction performance.

### 3. Media Type Distribution Pie Chart

Shows the percentage of image posts and video posts.

### 4. Top 10 Hashtag Count Bar Chart

Displays the most frequently occurring hashtag counts in the dataset.

---

## Final Outcome

After processing, the dataset becomes:

- Cleaner
- More consistent
- Free of duplicate records
- Properly formatted
- Suitable for machine-learning applications

The complete pipeline prepares the data, generates a virality target, trains a KNN classifier, evaluates its performance, and predicts whether social media posts are likely to be viral or non-viral.
