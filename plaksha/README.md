# SUPERVISED.py Project

## Overview

This project contains a Python script, `SUPERVISED.py`, that loads a social-media dataset from `sm.csv`, preprocesses it, generates features, trains a K-Nearest Neighbors classifier to predict a virality target, and saves results and plots.

## Files

- `SUPERVISED.py` - main script.
- `sm.csv` - input data file.
- `sm_preprocessed.csv` - cleaned and feature-engineered dataset.
- `sm_training.csv` - training split from the preprocessed dataset.
- `sm_test.csv` - test split from the preprocessed dataset.
- `sm_standardized.csv` - standardized version of the full dataset.
- `sm_training_standardized.csv` - standardized training split.
- `sm_test_standardized.csv` - standardized test split.
- `virality_feature_heatmap.png` - correlation heatmap for virality-related features.
- `knn_k_vs_accuracy.png` - cross-validated accuracy by K for KNN.
- `knn_confusion_matrix.png` - confusion matrix for the final KNN test evaluation.

## What the script does

1. Loads `sm.csv` with standard NA-handling values.
2. Prints the original shape and missing-value counts.
3. Cleans data:
   - trims whitespace on text fields,
   - normalizes `media_type` values to `1` for image and `0` for video,
   - coerces numeric columns to numbers and converts invalid entries to `NaN`,
   - drops duplicate rows,
   - estimates or imputes zero/invalid `video_view_count` values,
   - drops unwanted columns `category_name` and `comments`.
4. Imputes missing numeric values with column medians.
5. Fills missing `media_type` with its mode.
6. Creates feature ratios:
   - `views_per_follower`
   - `views_per_hashtag`
7. Creates virality features:
   - `like_rate`
   - `virality_score`
   - `target` label for the top 10% of rows by `virality_score`
8. Saves the preprocessed dataset as `sm_preprocessed.csv`.

## Train/Test split and scaling

- The script splits data into training (80%) and test (20%).
- It uses a stratified split when `target` exists and has multiple classes.
- It standardizes numeric features across the full dataset and also standardizes train/test splits separately using the training scaler.
- Standardized files are saved as:
  - `sm_standardized.csv`
  - `sm_training_standardized.csv`
  - `sm_test_standardized.csv`

## Modeling

- Uses K-Nearest Neighbors (`KNeighborsClassifier`).
- Selects the best `k` based on 5-fold cross-validation accuracy on the training set.
- Trains the final KNN model with the selected `k`.
- Evaluates performance on the held-out test set.
- Prints:
  - test accuracy,
  - classification report,
  - confusion matrix,
  - cross-validation metrics.

## Plots saved by the script

- `virality_feature_heatmap.png`
- `knn_k_vs_accuracy.png`
- `knn_confusion_matrix.png`

## How to run

1. Install required packages if needed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
2. Run the script from the `plaksha` folder:
   ```bash
   python SUPERVISED.py
   ```

## Notes

- The script assumes `sm.csv` is present in the same directory.
- The `target` is created by labeling the top 10% of rows by a custom `virality_score`.
- Selecting `k` using cross-validation avoids leaking test data into model selection.

