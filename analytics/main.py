"""
main.py — Analytics Module Entry Point

Run from the repository root:
    python analytics/main.py

Or from inside analytics/:
    python main.py

Produces:
  analytics/titanic.csv
  analytics/outputs/charts/     — all PNG charts
  analytics/outputs/model_results.csv
  analytics/outputs/regression_results.txt
  analytics/models/best_pipeline.joblib
"""

from eda import run_eda
from modeling import run_modeling


def main():
    print("=" * 62)
    print("  ZEPTO DATA & AI PLATFORM — Analytics Module")
    print("=" * 62)

    print("\n-- PART A: Profiling, EDA, Data Story -----------------------")
    run_eda()

    print("\n-- PART B: Machine Learning Pipeline ------------------------")
    run_modeling()

    print("\n" + "=" * 62)
    print("  Analytics complete.")
    print("  Outputs: analytics/outputs/  |  Model: analytics/models/")
    print("=" * 62)


if __name__ == "__main__":
    main()
