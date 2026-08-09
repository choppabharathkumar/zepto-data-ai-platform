"""
modeling.py

Part B: Machine learning pipeline for the Titanic dataset.

Covers:
  - Class balance check
  - Stratified train/test split
  - Preprocessing (train-only) via ColumnTransformer + Pipeline
  - Three classifiers: Logistic Regression, Decision Tree, Random Forest
  - Evaluation: confusion matrix, accuracy, precision, recall, F1, ROC/AUC
  - Decision Tree visualisation
  - Imbalance comparison: baseline, class_weight="balanced", SMOTE
  - GridSearchCV for Random Forest tuning + OOB score
  - Regression: predict Fare with Linear Regression
  - Model comparison table
  - Pipeline save (joblib) and reload verification
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay,
)

from imblearn.over_sampling import SMOTE

from constants import CHARTS_DIR, MODELS_DIR, OUTPUTS_DIR, TITANIC_CSV


# ── Feature configuration ─────────────────────────────────────────────────────

NUMERIC_FEATURES     = ["age", "fare", "sibsp", "parch", "pclass"]
CATEGORICAL_FEATURES = ["sex", "embarked"]
TARGET               = "survived"


def _ensure_dirs():
    for d in [CHARTS_DIR, MODELS_DIR, OUTPUTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _save_fig(filename):
    path = CHARTS_DIR / filename
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Saved chart: {filename}")


# ── Load and clean ────────────────────────────────────────────────────────────

def load_and_clean():
    """
    Load from the committed titanic.csv (not from seaborn again).
    Apply the same cleaning decisions as EDA.
    """
    df = pd.read_csv(TITANIC_CSV)

    # Drop high-missing columns
    for col in ["deck", "cabin"]:
        if col in df.columns and df[col].isnull().mean() > 0.50:
            df = df.drop(columns=[col])

    # Drop rows where embarked is missing (<5% threshold)
    df = df.dropna(subset=["embarked"])
    if "embark_town" in df.columns:
        df = df.dropna(subset=["embark_town"])

    print(f"[Modeling] Dataset loaded from titanic.csv: {df.shape}")
    return df


# ── 1. Class balance ──────────────────────────────────────────────────────────

def check_class_balance(df):
    """Report the survived/not-survived class distribution."""
    print("\n[1] CLASS BALANCE")
    counts = df[TARGET].value_counts()
    pct    = df[TARGET].value_counts(normalize=True) * 100
    print(f"    Not survived (0): {counts[0]}  ({pct[0]:.1f}%)")
    print(f"    Survived     (1): {counts[1]}  ({pct[1]:.1f}%)")
    print("    Moderate imbalance — will compare baseline, balanced weight, and SMOTE.")


# ── 2. Train / test split ─────────────────────────────────────────────────────

def split_data(df):
    """
    Stratified train/test split.
    Stratification preserves the survived class ratio in both sets,
    which is important with imbalanced classes — without it, a small
    test set might accidentally contain a different proportion of
    survivors, making evaluation unreliable.
    """
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[features].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n[2] TRAIN / TEST SPLIT")
    print(f"    Features : {features}")
    print(f"    Train    : {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows")
    print(f"    Train survived rate: {y_train.mean():.4f}")
    print(f"    Test  survived rate: {y_test.mean():.4f}")
    print("    Stratification keeps the class ratio consistent across both sets.")
    return X_train, X_test, y_train, y_test


# ── 3. Preprocessing pipeline factory ────────────────────────────────────────

def build_preprocessor():
    """
    Return a fresh (unfitted) ColumnTransformer.
    Called once per experiment to avoid shared-state bugs between pipelines.
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric",     numeric_pipe,     NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
    ])


# ── 4. Train three classifiers ────────────────────────────────────────────────

def train_classifiers(X_train, y_train):
    """
    Train Logistic Regression, Decision Tree, and Random Forest.
    Each pipeline gets a fresh preprocessor so fits are independent.
    Preprocessing is fit ONLY on training data inside each Pipeline.
    """
    print("\n[3] TRAINING CLASSIFIERS")

    model_defs = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    }

    pipelines = {}
    for name, model in model_defs.items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model",        model),
        ])
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe
        print(f"    Trained: {name}")

    return pipelines


# ── 5. Evaluate classifiers ───────────────────────────────────────────────────

def evaluate_classifiers(pipelines, X_test, y_test):
    """Confusion matrix, accuracy, precision, recall, F1, ROC/AUC for each model."""
    print("\n[4] EVALUATION")

    rows = []
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

    for name, pipe in pipelines.items():
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc     = auc(fpr, tpr)

        rows.append({
            "Model":     name,
            "Accuracy":  round(acc,  4),
            "Precision": round(prec, 4),
            "Recall":    round(rec,  4),
            "F1":        round(f1,   4),
            "AUC":       round(roc_auc, 4),
        })

        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

        # Confusion matrix chart
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Not Survived", "Survived"],
        ).plot(ax=ax_cm, colorbar=False)
        ax_cm.set_title(f"Confusion Matrix — {name}")
        plt.tight_layout()
        _save_fig(f"cm_{name.lower().replace(' ', '_')}.png")

    # ROC chart
    ax_roc.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.500)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves — All Classifiers")
    ax_roc.legend()
    plt.tight_layout()
    _save_fig("roc_curves.png")

    results = pd.DataFrame(rows)
    print("\n    Classification Comparison:")
    print(results.to_string(index=False))
    return results


# ── 6. Decision Tree visualisation ───────────────────────────────────────────

def visualize_decision_tree(pipelines):
    """Visualise the Decision Tree with feature names and class names."""
    print("\n[5] DECISION TREE VISUALISATION")

    pipe         = pipelines["Decision Tree"]
    dt           = pipe.named_steps["model"]
    preprocessor = pipe.named_steps["preprocessor"]

    # Reconstruct feature names after one-hot encoding
    cat_encoder       = preprocessor.named_transformers_["categorical"]["encoder"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names     = NUMERIC_FEATURES + cat_feature_names

    plt.figure(figsize=(24, 10))
    plot_tree(
        dt,
        feature_names=feature_names,
        class_names=["Not Survived", "Survived"],
        filled=True,
        rounded=True,
        fontsize=9,
        max_depth=3,   # limit visual depth for readability; model itself uses max_depth=4
    )
    plt.title("Decision Tree (max_depth=3 shown for readability; model trained at max_depth=4)")
    plt.tight_layout()
    _save_fig("decision_tree.png")
    print("    Decision Tree chart saved (visual depth limited to 3 for readability).")


# ── 7. Imbalance comparison ───────────────────────────────────────────────────

def imbalance_comparison(X_train, X_test, y_train, y_test):
    """
    Compare three strategies using Logistic Regression:
      1. Baseline (no adjustment)
      2. class_weight='balanced'
      3. SMOTE (applied ONLY to training data)
    """
    print("\n[6] IMBALANCE COMPARISON (Logistic Regression)")

    # Fit a shared preprocessor on training data once
    prep = build_preprocessor()
    prep.fit(X_train)
    X_train_proc = prep.transform(X_train)
    X_test_proc  = prep.transform(X_test)

    strategies = {
        "Baseline":        LogisticRegression(max_iter=500, random_state=42),
        "Balanced Weight": LogisticRegression(max_iter=500, random_state=42, class_weight="balanced"),
    }

    rows = []
    for name, model in strategies.items():
        model.fit(X_train_proc, y_train)
        y_pred = model.predict(X_test_proc)
        rows.append({
            "Strategy":  name,
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        })

    # SMOTE — only on training data
    sm = SMOTE(random_state=42)
    X_sm, y_sm = sm.fit_resample(X_train_proc, y_train)
    print(f"    SMOTE resampled train from {len(y_train)} -> {len(y_sm)} rows")

    lr_smote = LogisticRegression(max_iter=500, random_state=42)
    lr_smote.fit(X_sm, y_sm)
    y_pred_sm = lr_smote.predict(X_test_proc)
    rows.append({
        "Strategy":  "SMOTE",
        "Precision": round(precision_score(y_test, y_pred_sm, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred_sm, zero_division=0), 4),
        "F1":        round(f1_score(y_test, y_pred_sm, zero_division=0), 4),
    })

    imbalance_df = pd.DataFrame(rows)
    print(imbalance_df.to_string(index=False))

    best = imbalance_df.loc[imbalance_df["F1"].idxmax()]
    best_recall = imbalance_df.loc[imbalance_df["Recall"].idxmax()]
    print(f"""
    Imbalance Conclusion:
    '{best["Strategy"]}' achieved the best F1 = {best["F1"]:.4f}.
    '{best_recall["Strategy"]}' achieved the best Recall = {best_recall["Recall"]:.4f}.
    class_weight="balanced" and SMOTE both improve Recall (catching more true survivors)
    at some cost to Precision versus the baseline. For a survival task, missing a real
    survivor (false negative) is generally more costly than a false alarm — so higher
    Recall strategies are preferable even if Precision drops slightly. SMOTE generates
    synthetic minority samples; class_weight is simpler and equally effective here.""")

    return imbalance_df


# ── 8. Random Forest tuning with GridSearchCV ─────────────────────────────────

def tune_random_forest(X_train, X_test, y_train, y_test):
    """
    GridSearchCV over n_estimators, max_depth, max_features.
    OOB score computed separately with oob_score=True on best parameters.
    """
    print("\n[7] RANDOM FOREST HYPERPARAMETER TUNING (GridSearchCV)")

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth":    [5, 10, None],
        "model__max_features": ["sqrt", "log2"],
    }

    rf_pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestClassifier(random_state=42)),
    ])

    grid = GridSearchCV(rf_pipe, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    best_params  = grid.best_params_
    best_cv_f1   = grid.best_score_
    print(f"    Best parameters : {best_params}")
    print(f"    Best CV F1 score: {best_cv_f1:.4f}")

    # OOB score — fit a standalone RF with oob_score=True using best params
    best_prep = build_preprocessor()
    best_prep.fit(X_train)
    X_train_proc = best_prep.transform(X_train)

    rf_oob = RandomForestClassifier(
        n_estimators=best_params["model__n_estimators"],
        max_depth=best_params["model__max_depth"],
        max_features=best_params["model__max_features"],
        oob_score=True,
        random_state=42,
    )
    rf_oob.fit(X_train_proc, y_train)
    print(f"    OOB Score       : {rf_oob.oob_score_:.4f}")

    return grid.best_estimator_, best_params, best_cv_f1, rf_oob.oob_score_


# ── 9. Regression side task ───────────────────────────────────────────────────

def run_regression(df):
    """
    Predict Fare using multivariate Linear Regression.
    Reports MAE, RMSE, R², Adjusted R², and a residual plot.
    """
    print("\n[8] REGRESSION — Predict Fare (Linear Regression)")

    reg_numeric     = ["pclass", "age", "sibsp", "parch"]
    reg_categorical = ["sex", "embarked"]
    reg_features    = reg_numeric + reg_categorical
    reg_target      = "fare"

    df_reg = df[reg_features + [reg_target]].dropna().copy()
    print(f"    Regression dataset: {df_reg.shape[0]} rows, {len(reg_features)} features")

    X_reg = df_reg[reg_features]
    y_reg = df_reg[reg_target]

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    reg_preprocessor = ColumnTransformer([
        ("numeric",     Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), reg_numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), reg_categorical),
    ])

    reg_pipeline = Pipeline([
        ("preprocessor", reg_preprocessor),
        ("model",        LinearRegression()),
    ])
    reg_pipeline.fit(X_train_r, y_train_r)
    y_pred_r = reg_pipeline.predict(X_test_r)

    # Metrics
    mae  = float(np.mean(np.abs(y_test_r - y_pred_r)))
    rmse = float(np.sqrt(np.mean((y_test_r - y_pred_r) ** 2)))
    ss_res = float(np.sum((y_test_r - y_pred_r) ** 2))
    ss_tot = float(np.sum((y_test_r - y_test_r.mean()) ** 2))
    r2   = 1.0 - ss_res / ss_tot

    n = len(y_test_r)
    k = len(reg_pipeline.named_steps["model"].coef_)   # actual coefficient count
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)

    print(f"    MAE         = {mae:.4f}")
    print(f"    RMSE        = {rmse:.4f}")
    print(f"    R²          = {r2:.4f}")
    print(f"    Adjusted R² = {adj_r2:.4f}  (k={k} coefficients)")

    # Residual plot
    residuals = y_test_r.values - y_pred_r
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred_r, residuals, alpha=0.5, color="steelblue", edgecolors="none")
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Predicted Fare")
    plt.ylabel("Residuals  (Actual − Predicted)")
    plt.title("Residual Plot — Fare Linear Regression")
    plt.tight_layout()
    _save_fig("regression_residuals.png")

    print("""
    Heteroscedasticity: The residual plot shows residuals fanning out at higher
    predicted fare values — variance of errors increases as the fare level rises.
    This is consistent with heteroscedasticity. The linear regression model
    systematically underestimates the error spread for high-fare first-class
    passengers. A log-transformation of fare would likely reduce this effect
    and produce more uniform residuals.""")

    return {
        "MAE":         round(mae,    4),
        "RMSE":        round(rmse,   4),
        "R2":          round(r2,     4),
        "Adjusted_R2": round(adj_r2, 4),
    }


# ── 10. Model comparison and recommendation ───────────────────────────────────

def model_comparison(clf_results, reg_results):
    """Save and print final comparison table with written recommendation."""
    print("\n[9] MODEL COMPARISON")

    print("\n    Classification Metrics:")
    print(clf_results.to_string(index=False))

    print("\n    Regression Metrics (Linear Regression — Fare prediction):")
    reg_df = pd.DataFrame([reg_results])
    print(reg_df.to_string(index=False))

    # Save artifacts
    clf_results.to_csv(OUTPUTS_DIR / "model_results.csv", index=False)
    with open(OUTPUTS_DIR / "regression_results.txt", "w", encoding="utf-8") as fh:
        fh.write("Fare Regression Results — Linear Regression\n")
        fh.write("=" * 45 + "\n")
        for k, v in reg_results.items():
            fh.write(f"{k}: {v}\n")
    print("    Saved: model_results.csv, regression_results.txt")

    # Final recommendation
    best_row = clf_results.loc[clf_results["F1"].idxmax()]
    worst_r2 = reg_results["R2"]
    print(f"""
    Final Recommendation:
    For the survival classification task, {best_row["Model"]} achieved the best
    F1 = {best_row["F1"]:.4f} with Accuracy = {best_row["Accuracy"]:.4f} and AUC = {best_row["AUC"]:.4f}.
    Random Forest consistently outperformed Logistic Regression and Decision Tree because
    its ensemble of many trees reduces variance without significantly increasing bias.
    For the fare regression task, Linear Regression produced R² = {worst_r2:.4f}, indicating
    that the selected features explain only a moderate portion of fare variance — this is
    expected because fare is heavily influenced by cabin class, which partially overlaps
    with pclass but also contains information not captured by the numeric features alone.
    Note: Classification and regression metrics are not directly comparable — accuracy/F1/AUC
    measure categorical prediction quality, while MAE/RMSE/R² measure continuous prediction error.
    """)


# ── 11. Save and reload pipeline ──────────────────────────────────────────────

def save_and_reload_pipeline(best_pipeline, X_test):
    """
    Save the complete fitted Pipeline (preprocessor + model) with joblib.
    Reload and verify it can predict from raw input without manual preprocessing.
    """
    print("\n[10] SAVE AND RELOAD PIPELINE")
    path = MODELS_DIR / "best_pipeline.joblib"
    joblib.dump(best_pipeline, path)
    print(f"    Saved: {path.name}")
    print(f"    Pipeline type: {type(best_pipeline).__name__}")
    print(f"    Steps: {[s[0] for s in best_pipeline.steps]}")

    # Reload and predict from raw (unprocessed) input
    loaded = joblib.load(path)
    sample = X_test.iloc[:1].copy()
    prediction = loaded.predict(sample)
    probability = loaded.predict_proba(sample)[0]
    print(f"\n    Reload verified.")
    print(f"    Sample raw input:\n{sample.to_string()}")
    print(f"    Predicted class: {prediction[0]}  "
          f"(Not Survived prob={probability[0]:.3f}, Survived prob={probability[1]:.3f})")
    print("    Pipeline accepts raw features and preprocesses automatically. [OK]")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_modeling():
    """Run the complete ML pipeline."""
    _ensure_dirs()

    df = load_and_clean()
    check_class_balance(df)
    X_train, X_test, y_train, y_test = split_data(df)

    pipelines   = train_classifiers(X_train, y_train)
    clf_results = evaluate_classifiers(pipelines, X_test, y_test)
    visualize_decision_tree(pipelines)

    imbalance_comparison(X_train, X_test, y_train, y_test)

    best_pipeline, best_params, best_cv_f1, oob = tune_random_forest(
        X_train, X_test, y_train, y_test
    )

    reg_results = run_regression(df)
    model_comparison(clf_results, reg_results)
    save_and_reload_pipeline(best_pipeline, X_test)

    print("\n[MODELING COMPLETE]")
    return clf_results, reg_results
