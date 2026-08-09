# Analytics Module — Zepto Data & AI Platform

## Overview

This module implements a complete analytics pipeline on the classic **Titanic** dataset.
It covers data profiling, exploratory data analysis (EDA), a visual data story,
classification modelling, class-imbalance handling, hyperparameter tuning, regression,
model comparison, and pipeline persistence.

**Entry point:**

```bash
cd analytics
python main.py
```

---

## Dataset

| Property | Value |
|---|---|
| Source | `sns.load_dataset("titanic")` — loaded exactly once in `eda.py` |
| Offline copy | `analytics/titanic.csv` (committed to the repository) |
| Shape (raw) | 891 rows × 15 columns |
| Shape (clean) | 889 rows × 14 columns |

---

## Part A — EDA

### Step 1: Load

Loaded via `sns.load_dataset("titanic")` and saved to `titanic.csv`.

### Step 2: Profile

```
Shape: (891, 15)
Columns: survived, pclass, sex, age, sibsp, parch, fare, embarked,
         class, who, adult_male, deck, embark_town, alive, alone
```

### Step 3: Missing Value Analysis

| Column | % Missing | Decision |
|---|---|---|
| deck | 77.22% | Drop column — too high to impute |
| age | 19.87% | Impute with median (within 5–30% band) |
| embarked | 0.22% | Drop 2 rows (< 5% threshold) |
| embark_town | 0.22% | Drop rows (same source as embarked) |

### Step 4: Cleaning Decisions

- **deck** (77.2% missing) → dropped entirely. Imputing a column with more missing values than present values would introduce more noise than signal.
- **age** (19.87% missing) → imputed with median (28.0). Within the defensible 5–30% range; median is robust to outliers and avoids data loss.
- **embarked / embark_town** (0.22% missing) → dropped 2 rows. Below 5% threshold, so row deletion has negligible impact on the dataset.

### Step 5: Age Analysis

- Q1 = 22.00, Q3 = 35.00, IQR = 13.00
- Outliers outside 1.5 × IQR: **65**
- Distribution: right-tailed; young adults are the largest group.

Chart: `outputs/charts/age_distribution.png`

### Step 6: Fare Analysis

| Statistic | Value |
|---|---|
| Mean | 32.10 |
| Median | 14.45 |
| Mode | 8.05 |
| Skewness | 4.80 |
| Outliers (1.5×IQR) | 114 |

**Conclusion:** Mean > Median > Mode → strongly **right-skewed (positive)**. Most passengers paid low fares; a small number of first-class passengers paid very high fares, pulling the mean far above the median.

Chart: `outputs/charts/fare_distribution.png`

### Step 7: Survival Analysis (Boolean Masking)

**By sex:**

| Sex | Survival Rate |
|---|---|
| Female | 74.0% |
| Male | 18.9% |

**By passenger class:**

| Class | Survival Rate |
|---|---|
| 1st | 62.6% |
| 2nd | 47.3% |
| 3rd | 24.2% |

**By sex + class:**

| Group | Survival Rate |
|---|---|
| Female, Class 1 | 96.7% |
| Female, Class 2 | 92.1% |
| Female, Class 3 | 50.0% |
| Male, Class 1 | 36.9% |
| Male, Class 2 | 15.7% |
| Male, Class 3 | 13.5% |

### Step 8: Correlation Matrix

Columns used: `survived, pclass, age, sibsp, parch, fare`
(`adult_male` and `alone` excluded — derived/redundant features)

**Two strongest off-diagonal correlations:**

| Pair | Correlation | Direction |
|---|---|---|
| fare ↔ pclass | -0.5482 | Negative — higher class (lower number) = higher fare |
| parch ↔ sibsp | +0.4145 | Positive — families travel together |

Chart: `outputs/charts/correlation_matrix.png`

### Step 9: Data Story — 4 Multivariate Charts

**Chart 1 — Survival Rate by Sex and Class** (`story1_survival_sex_class.png`)
Female passengers survived at significantly higher rates than males in every class. First-class females approached a ~97% survival rate while third-class males fell below 15%, showing the combined effect of the "women and children first" protocol and the access advantage of upper-deck cabin placement for wealthier passengers.

**Chart 2 — Age by Survival and Sex** (`story2_age_survival_sex.png`)
Among survivors, the age distribution for females is broad, indicating women of all ages benefited from evacuation priority. A small bump at younger ages in the survivor group for both sexes reflects the prioritisation of children. Non-surviving males skew toward younger adults, consistent with young men being in the lower-class cabins with limited lifeboat access.

**Chart 3 — Fare by Class and Survival** (`story3_fare_class_survival.png`)
First-class fares are dramatically higher and more variable than second or third class, reflecting luxury pricing. Within each class, survivors tend to have paid slightly higher fares, possibly because more expensive cabins were positioned closer to lifeboats on upper decks. Third-class fares are tightly clustered at low values, reflecting little pricing variation in steerage.

**Chart 4 — Embarkation Port and Survival** (`story4_embarkation_survival.png`)
Southampton contributed the largest number of passengers and consequently the most absolute survivors, though its survival proportion is lower than Cherbourg. Cherbourg passengers had the highest survival rate relative to their count, likely because Cherbourg attracted disproportionately more first-class travellers. Queenstown passengers were predominantly third-class emigrants, resulting in a lower overall survival proportion for that boarding port.

### Step 10: Standardisation Sanity Check (EDA only)

| Column | Before (mean / std) | After (mean / std) |
|---|---|---|
| age | 29.32 / 12.98 | 0.000000 / 1.000000 |
| fare | 32.10 / 49.70 | 0.000000 / 1.000000 |

> **Note:** These z-scores are discarded after the check. The modelling pipeline fits `StandardScaler` only on training data to prevent data leakage.

---

## Part B — Machine Learning Pipeline

**Features used:** `age, fare, sibsp, parch, pclass` (numeric) + `sex, embarked` (categorical)
**Target:** `survived`

### Step 1: Class Balance

| Class | Count | % |
|---|---|---|
| Not survived (0) | 549 | 61.8% |
| Survived (1) | 340 | 38.2% |

Moderate imbalance — addressed with baseline, `class_weight="balanced"`, and SMOTE.

### Step 2: Train/Test Split

- **Train:** 711 rows | **Test:** 178 rows
- `stratify=y` ensures the 38.2% survivor ratio is preserved in both sets.
- Train survived rate: 0.3826 | Test survived rate: 0.3820 ✓

### Step 3: Preprocessing Pipeline

- **Numeric features:** `SimpleImputer(median)` → `StandardScaler`
- **Categorical features:** `SimpleImputer(most_frequent)` → `OneHotEncoder`
- Combined with `ColumnTransformer` inside a `sklearn.Pipeline`
- Preprocessor is **fit only on training data** — no data leakage.

### Step 4–5: Classification Results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8090 | 0.7833 | 0.6912 | 0.7344 | 0.8610 |
| Decision Tree | 0.8090 | 0.8148 | 0.6471 | 0.7213 | 0.8560 |
| **Random Forest** | **0.8202** | 0.7812 | **0.7353** | **0.7576** | 0.8215 |

Charts: `cm_logistic_regression.png`, `cm_decision_tree.png`, `cm_random_forest.png`, `roc_curves.png`, `decision_tree.png`

### Step 6: Imbalance Comparison (Logistic Regression)

| Strategy | Precision | Recall | F1 |
|---|---|---|---|
| Baseline | 0.7833 | 0.6912 | 0.7344 |
| Balanced Weight | 0.7183 | **0.7500** | 0.7338 |
| SMOTE | 0.7353 | 0.7353 | **0.7353** |

- SMOTE resampled training set from 711 → 878 rows.
- `class_weight="balanced"` achieves the highest Recall (catching more true survivors).
- For a survival prediction task, higher Recall is preferred because missing a real survivor (false negative) is more costly than a false alarm.

### Step 7: Random Forest Hyperparameter Tuning (GridSearchCV)

```
param_grid:
  n_estimators: [100, 200]
  max_depth:    [5, 10, None]
  max_features: ["sqrt", "log2"]

Best parameters: n_estimators=100, max_depth=5, max_features="sqrt"
Best CV F1 score: 0.7420
OOB Score:        0.8087
```

### Step 8: Regression — Predict Fare

| Metric | Value |
|---|---|
| MAE | 24.4711 |
| RMSE | 60.1040 |
| R² | 0.3020 |
| Adjusted R² | 0.2548 |

**Interpretation:** R² = 0.30 means the selected features (pclass, age, sibsp, parch, sex, embarked) explain about 30% of fare variance. The residual plot shows heteroscedasticity — errors fan out at higher fares. A log-transformation of fare would likely improve this. This is expected: fare reflects cabin-level pricing not fully captured by passenger-class alone.

Chart: `regression_residuals.png`

### Step 9: Model Comparison and Final Recommendation

**Random Forest** is the best classifier:
- Highest F1 = 0.7576, Accuracy = 0.8202, AUC = 0.8215
- Ensemble of 100 trees reduces variance without significant bias increase
- Best-tuned parameters: `n_estimators=100, max_depth=5, max_features="sqrt"`

Saved to: `outputs/model_results.csv`, `outputs/regression_results.txt`

### Step 10: Pipeline Save and Reload

```
Saved:  analytics/models/best_pipeline.joblib
Steps:  ['preprocessor', 'model']
```

Reload verified — accepts raw (unprocessed) features and preprocesses automatically. ✓

---

## Output Files

```
analytics/
├── titanic.csv                        # Offline dataset copy
├── outputs/
│   ├── model_results.csv              # Classification comparison table
│   ├── regression_results.txt         # Fare regression metrics
│   └── charts/
│       ├── age_distribution.png
│       ├── fare_distribution.png
│       ├── correlation_matrix.png
│       ├── story1_survival_sex_class.png
│       ├── story2_age_survival_sex.png
│       ├── story3_fare_class_survival.png
│       ├── story4_embarkation_survival.png
│       ├── cm_logistic_regression.png
│       ├── cm_decision_tree.png
│       ├── cm_random_forest.png
│       ├── roc_curves.png
│       ├── decision_tree.png
│       └── regression_residuals.png
└── models/
    └── best_pipeline.joblib           # Full sklearn Pipeline (preprocessor + RF)
```

---

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point — runs EDA then modelling |
| `eda.py` | Steps 1–10 of Part A (profiling, EDA, data story) |
| `modeling.py` | Steps 1–10 of Part B (ML pipeline) |
| `constants.py` | Shared `Path` constants |
| `requirements.txt` | Python dependencies |
| `titanic.csv` | Dataset offline copy |
