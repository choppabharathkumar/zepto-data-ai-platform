"""
eda.py

Part A: Data profiling, missing value analysis, exploratory data analysis,
and data story for the Titanic dataset.

All charts are saved to outputs/charts/.
All written interpretations live in analytics/README.md.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from constants import CHARTS_DIR, TITANIC_CSV


def _ensure_dirs():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_fig(filename):
    path = CHARTS_DIR / filename
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Saved chart: {filename}")


# ── 1. Load ───────────────────────────────────────────────────────────────────

def load_titanic():
    """Load Titanic dataset once from seaborn and save offline copy."""
    df = sns.load_dataset("titanic")
    df.to_csv(TITANIC_CSV, index=False)
    print(f"[1] Loaded Titanic: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"    Columns: {list(df.columns)}")
    print(f"    Saved -> {TITANIC_CSV.name}")
    return df


# ── 2. Profile ────────────────────────────────────────────────────────────────

def profile(df):
    """Print shape, dtypes, and descriptive statistics."""
    print("\n[2] PROFILING")
    print(f"    Shape: {df.shape}")
    print("\n--- df.info() ---")
    df.info()
    print("\n--- df.describe() ---")
    print(df.describe(include="all").to_string())


# ── 3. Missing value analysis ─────────────────────────────────────────────────

def analyse_missing(df):
    """Calculate and report missing-value percentages for every affected column."""
    print("\n[3] MISSING VALUE ANALYSIS")
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    if missing_pct.empty:
        print("    No missing values found.")
    else:
        print(missing_pct.round(2).to_string())
    return missing_pct


# ── 4. Clean (EDA version) ───────────────────────────────────────────────────

def clean(df):
    """
    Apply threshold-based missing-value decisions:
      deck        (~77%)  : drop column — far too high to impute reliably
      age         (~20%)  : impute with median — within 5-30% imputation band
      embarked    (~0.2%) : drop rows — below 5% threshold, minimal data loss
      embark_town (~0.2%) : same source as embarked, drop rows together
    """
    print("\n[4] CLEANING")

    # Very high (>50%): drop deck column
    for col in ["deck", "cabin"]:
        if col in df.columns and df[col].isnull().mean() > 0.50:
            pct = df[col].isnull().mean() * 100
            df = df.drop(columns=[col])
            print(f"    Dropped '{col}' ({pct:.1f}% missing) — too high to impute")

    # 5-30%: impute age with median
    age_median = df["age"].median()
    df["age"] = df["age"].fillna(age_median)
    print(f"    Imputed 'age' with median = {age_median:.1f}")

    # Under 5%: drop rows where embarked is missing
    before = len(df)
    df = df.dropna(subset=["embarked"])
    if "embark_town" in df.columns:
        df = df.dropna(subset=["embark_town"])
    dropped = before - len(df)
    print(f"    Dropped {dropped} row(s) with missing 'embarked' (<5% threshold)")

    print(f"    Clean shape: {df.shape}")
    return df


# ── 5. Age analysis ───────────────────────────────────────────────────────────

def analyse_age(df):
    """Histogram, box plot, and IQR outlier count for Age."""
    print("\n[5] AGE ANALYSIS")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Age Distribution", fontsize=14)

    axes[0].hist(df["age"].dropna(), bins=30, color="steelblue", edgecolor="white")
    axes[0].set_title("Histogram")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Count")

    df.boxplot(column="age", ax=axes[1])
    axes[1].set_title("Box Plot")
    axes[1].set_ylabel("Age")

    plt.tight_layout()
    _save_fig("age_distribution.png")

    q1 = df["age"].quantile(0.25)
    q3 = df["age"].quantile(0.75)
    iqr = q3 - q1
    n_outliers = int(((df["age"] < q1 - 1.5 * iqr) | (df["age"] > q3 + 1.5 * iqr)).sum())

    print(f"    Q1={q1:.2f}  Q3={q3:.2f}  IQR={iqr:.2f}")
    print(f"    Outliers outside 1.5 x IQR: {n_outliers}")


# ── 6. Fare analysis ──────────────────────────────────────────────────────────

def analyse_fare(df):
    """Histogram, box plot, IQR outlier count, and skewness conclusion for Fare."""
    print("\n[6] FARE ANALYSIS")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Fare Distribution", fontsize=14)

    axes[0].hist(df["fare"], bins=40, color="salmon", edgecolor="white")
    axes[0].set_title("Histogram")
    axes[0].set_xlabel("Fare (GBP)")
    axes[0].set_ylabel("Count")

    df.boxplot(column="fare", ax=axes[1])
    axes[1].set_title("Box Plot")
    axes[1].set_ylabel("Fare (GBP)")

    plt.tight_layout()
    _save_fig("fare_distribution.png")

    mean_fare   = df["fare"].mean()
    median_fare = df["fare"].median()
    mode_fare   = df["fare"].mode()[0]
    skewness    = df["fare"].skew()

    q1 = df["fare"].quantile(0.25)
    q3 = df["fare"].quantile(0.75)
    iqr = q3 - q1
    n_outliers = int(((df["fare"] < q1 - 1.5 * iqr) | (df["fare"] > q3 + 1.5 * iqr)).sum())

    print(f"    Mean     = {mean_fare:.4f}")
    print(f"    Median   = {median_fare:.4f}")
    print(f"    Mode     = {mode_fare:.4f}")
    print(f"    Skewness = {skewness:.4f}")
    print(f"    Outliers outside 1.5 x IQR: {n_outliers}")

    direction = "right-skewed (positive)" if mean_fare > median_fare else "left-skewed (negative)"
    print(f"    Skewness conclusion: Mean ({mean_fare:.2f}) > Median ({median_fare:.2f}) > Mode ({mode_fare:.2f})")
    print(f"    => Distribution is {direction}.")
    print("       Most passengers paid low fares; a small number of first-class passengers")
    print("       paid very high fares, pulling the mean well above the median.")


# ── 7. Survival analysis ─────────────────────────────────────────────────────

def analyse_survival(df):
    """Compute survival rates by sex, pclass, and sex+pclass using boolean masking."""
    print("\n[7] SURVIVAL ANALYSIS")

    # By sex — boolean masking
    female_rate = df[df["sex"] == "female"]["survived"].mean()
    male_rate   = df[df["sex"] == "male"]["survived"].mean()
    print("    Survival by sex (boolean masking):")
    print(f"      Female : {female_rate:.4f}  ({female_rate * 100:.1f}%)")
    print(f"      Male   : {male_rate:.4f}  ({male_rate * 100:.1f}%)")

    # By pclass
    print("\n    Survival by pclass:")
    for pclass in [1, 2, 3]:
        rate = df[df["pclass"] == pclass]["survived"].mean()
        print(f"      Class {pclass}: {rate:.4f}  ({rate * 100:.1f}%)")

    # By sex + pclass — combined boolean masking with &
    print("\n    Survival by sex + pclass (boolean masking with &):")
    for sex in ["female", "male"]:
        for pclass in [1, 2, 3]:
            mask = (df["sex"] == sex) & (df["pclass"] == pclass)
            rate = df[mask]["survived"].mean()
            print(f"      {sex.title()} Class {pclass}: {rate:.4f}  ({rate * 100:.1f}%)")


# ── 8. Correlation matrix ─────────────────────────────────────────────────────

def plot_correlation(df):
    """
    Plot correlation heatmap for the exact six required columns:
    survived, pclass, age, sibsp, parch, fare.
    adult_male and alone are explicitly excluded.
    """
    print("\n[8] CORRELATION MATRIX")

    cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr = df[cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, linewidths=0.5)
    plt.title("Correlation Matrix — survived, pclass, age, sibsp, parch, fare\n"
              "(adult_male and alone excluded as required)")
    plt.tight_layout()
    _save_fig("correlation_matrix.png")

    # Find top 2 off-diagonal pairs by absolute correlation value
    corr_abs = corr.abs().copy()
    arr = corr_abs.to_numpy().copy()
    np.fill_diagonal(arr, 0)
    corr_abs = pd.DataFrame(arr, index=corr_abs.index, columns=corr_abs.columns)
    flat = corr_abs.unstack().sort_values(ascending=False)

    seen, pairs = set(), []
    for (a, b), v in flat.items():
        key = frozenset({a, b})
        if key not in seen and a != b:
            seen.add(key)
            pairs.append(((a, b), corr.loc[a, b]))
        if len(pairs) == 2:
            break

    print("    Two strongest off-diagonal correlations (by absolute value):")
    for (a, b), v in pairs:
        direction = "positive" if v > 0 else "negative"
        print(f"      {a} <-> {b}: {v:.4f} ({direction})")

    return corr, pairs


# ── 9. Multivariate data story ────────────────────────────────────────────────

def data_story(df):
    """
    Four multivariate charts, each followed by a printed 2-4 sentence interpretation.
    These charts are distinct from the earlier single-variable analyses.
    """
    print("\n[9] DATA STORY — 4 multivariate charts")

    # Chart 1: Survival rate by sex and passenger class
    pivot = df.groupby(["pclass", "sex"])["survived"].mean().unstack()
    pivot.plot(kind="bar", rot=0, figsize=(9, 5), colormap="Set1")
    plt.title("Survival Rate by Passenger Class and Sex")
    plt.xlabel("Passenger Class")
    plt.ylabel("Survival Rate")
    plt.legend(title="Sex")
    plt.tight_layout()
    _save_fig("story1_survival_sex_class.png")
    print("""
  [Chart 1 — Survival Rate by Sex and Class]
  Female passengers survived at significantly higher rates than males in every class.
  First-class females approached a ~97% survival rate while third-class males fell
  below 15%, showing the combined effect of the "women and children first" protocol
  and the access advantage of upper-deck cabin placement for wealthier passengers.""")

    # Chart 2: Age distribution by survival status and sex (violin)
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="survived", y="age", hue="sex", palette="muted")
    plt.title("Age Distribution by Survival Status and Sex")
    plt.xlabel("Survived  (0 = No, 1 = Yes)")
    plt.ylabel("Age")
    plt.tight_layout()
    _save_fig("story2_age_survival_sex.png")
    print("""
  [Chart 2 — Age by Survival and Sex]
  Among survivors, the age distribution for females is broad, indicating women
  of all ages benefited from evacuation priority. A small bump at younger ages
  in the survivor group for both sexes reflects the prioritisation of children.
  Non-surviving males skew toward younger adults, consistent with young men
  being in the lower-class cabins with limited lifeboat access.""")

    # Chart 3: Fare distribution by class and survival
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="pclass", y="fare", hue="survived", palette="Set2")
    plt.title("Fare by Passenger Class and Survival Status")
    plt.xlabel("Passenger Class")
    plt.ylabel("Fare (GBP)")
    plt.tight_layout()
    _save_fig("story3_fare_class_survival.png")
    print("""
  [Chart 3 — Fare by Class and Survival]
  First-class fares are dramatically higher and more variable than second or
  third class, reflecting luxury pricing. Within each class, survivors tend to
  have paid slightly higher fares, possibly because more expensive cabins were
  positioned closer to lifeboats on upper decks. Third-class fares are tightly
  clustered at low values, reflecting little pricing variation in steerage.""")

    # Chart 4: Embarkation port vs survival count
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="embarked", hue="survived",
                  palette="Set1", order=["S", "C", "Q"])
    plt.title("Passenger Count and Survival by Embarkation Port")
    plt.xlabel("Port  (S=Southampton, C=Cherbourg, Q=Queenstown)")
    plt.ylabel("Count")
    plt.tight_layout()
    _save_fig("story4_embarkation_survival.png")
    print("""
  [Chart 4 — Embarkation Port and Survival]
  Southampton contributed the largest number of passengers and consequently the
  most absolute survivors, though its survival proportion is lower than Cherbourg.
  Cherbourg passengers had the highest survival rate relative to their count,
  likely because Cherbourg attracted disproportionately more first-class travellers.
  Queenstown passengers were predominantly third-class emigrants, resulting in
  a lower overall survival proportion for that boarding port.""")


# ── 10. Standardisation sanity check ─────────────────────────────────────────

def standardization_check(df):
    """
    EDA-only z-score sanity check for age and fare on the full dataset.
    These standardised values are NOT used anywhere in the modelling pipeline.
    The modelling pipeline fits StandardScaler only on training data.
    """
    print("\n[10] STANDARDISATION SANITY CHECK (EDA only — not used in modelling)")
    for col in ["age", "fare"]:
        mu  = df[col].mean()
        std = df[col].std()
        z   = (df[col] - mu) / std
        print(f"    {col}:")
        print(f"      Before: mean={mu:.4f}, std={std:.4f}")
        print(f"      After : mean={z.mean():.6f} (~0),  std={z.std():.6f} (~1)")
    print("    => Standardisation working correctly.")
    print("    (These z-scores are discarded; modelling uses train-only StandardScaler.)")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_eda():
    """Run the complete EDA pipeline and return the cleaned DataFrame."""
    _ensure_dirs()

    df = load_titanic()
    profile(df)
    analyse_missing(df)
    df = clean(df)
    analyse_age(df)
    analyse_fare(df)
    analyse_survival(df)
    corr, top_pairs = plot_correlation(df)
    data_story(df)
    standardization_check(df)

    print("\n[EDA COMPLETE] Charts saved to analytics/outputs/charts/")
    return df
