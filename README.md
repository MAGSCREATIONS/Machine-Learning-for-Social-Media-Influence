# Machine-Learning-for-Social-Media-Influence

## What This Script Does

Think of the dataset as a messy notebook full of information. This script acts as a helper that cleans, organizes, and prepares the data before it is used for analysis or machine-learning models.

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

## Data Cleaning

### Removing Unnecessary Columns

- Drops the last two columns of the dataset if they exist.
- Removes the `category_name` column because it is not required for analysis.

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

## Handling Missing Values

### Numeric Columns

Missing numerical values are replaced using the median of the corresponding column.

This applies to columns such as:

- followers
- following
- likes
- comments
- video_view_count
- caption_length
- hashtag_count
- posts_count
- post_hour
- post_day_of_week

### Categorical Columns

Missing values in the `media_type` column are filled using the most common category (mode).

## Handling Video View Counts

Some records may contain a video view count of zero.

To avoid unrealistic values affecting analysis:

- The script calculates the median of valid video view counts.
- Any zero value is replaced with this median value.

## Feature Engineering

The script creates additional engagement-related features that may improve machine-learning performance.

### Views Per Like

Measures how many views are obtained for each like.

`views_per_like = video_view_count / likes`

### Views Per Comment

Measures how many views are obtained for each comment.

`views_per_comment = video_view_count / comments`

### Views Per Follower

Measures how many views are obtained relative to the creator's followers.

`views_per_follower = video_view_count / followers`

### Views Per Hashtag

Measures how many views are obtained per hashtag used.

`views_per_hashtag = video_view_count / hashtag_count`

### Feature Formatting

- Ratio features are rounded to two decimal places.
- Numeric columns are converted to integer format where appropriate.

## Data Export

After preprocessing is complete:

- The cleaned dataset is saved as `sm_preprocessed.csv`.

This file is then used for further analysis, visualization, and machine-learning tasks.

## Data Visualization

The script generates the following visualizations:

### 1. Media Type Distribution

A pie chart showing the proportion of:

- Image posts
- Video posts

This helps understand the overall content distribution.

### 2. Top 10 Hashtag Counts

A bar chart displaying:

- The ten most common hashtag counts
- Their frequencies within the dataset

This helps identify hashtag usage patterns.

## Why the Previous Version Had a Problem

The earlier version of the script recognized values such as:

- `NA`
- `NULL`
- `null`

as missing data.

However, it did not recognize:

- `unknown`
- `Unknown`

As a result, these values were treated as normal text instead of missing information. Because of this, some records were not cleaned or filled correctly.

The updated version now recognizes both `unknown` and `Unknown` as missing values, ensuring that they are processed correctly during data cleaning.

## Final Outcome

After preprocessing, the dataset becomes:

- Cleaner
- More consistent
- Free of duplicate records
- Properly formatted
- Suitable for machine-learning applications

The added engagement features and improved handling of missing values help produce more reliable analysis and better predictive models.
