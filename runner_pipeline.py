"""
Running — Trail Stability Prediction Pipeline
Author: Katie Hsu

Dataset covers both quantitative biomechanics (cadence, ground contact,
elevation gain) and qualitative runner self-report data (perceived comfort,
descent confidence, traction feedback, recommendation intent) to predict
trail shoe Stability_Rating, which informs product fit, feel, and ride decisions.
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = "running_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File paths — keep CSVs in the same folder as this script
SHOE_SPECS_PATH   = "shoe_specs.csv"
RUNNER_TESTS_PATH = "runner_tests.csv"

# Step 1: Load both dataset CSVs from folder
def load_raw_data():
    """Load the pre-existing messy CSVs from disk."""
    shoe_specs_df   = pd.read_csv(SHOE_SPECS_PATH)
    runner_tests_df = pd.read_csv(RUNNER_TESTS_PATH)

    print(f"✔  shoe_specs.csv loaded   → {shoe_specs_df.shape[0]} rows, "
          f"{shoe_specs_df.shape[1]} cols")
    print(f"✔  runner_tests.csv loaded → {runner_tests_df.shape[0]} rows, "
          f"{runner_tests_df.shape[1]} cols")
    return shoe_specs_df, runner_tests_df


# Step 2: Use sqlite INNER JOIN to load both csvs into dataset.

def sql_join(shoe_specs_df, runner_tests_df):
    """Load both tables into SQLite and JOIN on Shoe_ID."""
    conn = sqlite3.connect(":memory:")
    shoe_specs_df.to_sql("shoe_specs",    conn, if_exists="replace", index=False)
    runner_tests_df.to_sql("runner_tests", conn, if_exists="replace", index=False)

    query = """
        SELECT
            rt.Runner_ID,
            rt.Shoe_ID,
            ss.Shoe_Model,
            rt.Trail_Condition,
            rt.Cadence_spm,
            rt.Ground_Contact_ms,
            rt.Session_Duration_min,
            rt.Elevation_Gain_ft,
            ss.Stack_Height_mm,
            ss.Midsole_Firmness,
            ss.Lug_Depth_mm,
            ss.Heel_Toe_Drop_mm,
            ss.Outsole_Hardness,
            rt.Perceived_Comfort,
            rt.Confidence_On_Descent,
            rt.Traction_Feedback,
            rt.Would_Recommend,
            rt.Stability_Rating
        FROM runner_tests AS rt
        INNER JOIN shoe_specs AS ss
            ON rt.Shoe_ID = ss.Shoe_ID
    """

    joined_df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"✔  SQL JOIN complete → {len(joined_df)} rows, {joined_df.shape[1]} cols")
    return joined_df


# Step 3: Clean the data

NUMERIC_COLS = [
    "Stack_Height_mm", "Midsole_Firmness", "Lug_Depth_mm",
    "Heel_Toe_Drop_mm", "Outsole_Hardness",
    "Cadence_spm", "Ground_Contact_ms",
    "Session_Duration_min", "Elevation_Gain_ft",
    "Perceived_Comfort", "Confidence_On_Descent",
    "Stability_Rating",
]

def clean_data(df):
    """
    Cleaning steps:
      1. Standardise Trail_Condition labels
      2. IQR outlier capping on all numeric columns
      3. Median imputation for missing numerics
      4. Binary encode Traction_Feedback and Would_Recommend (qualitative)
      5. Binary encode surface condition flags
      6. Drop rows with missing target (Stability_Rating)
    """
    raw_rows = len(df)

    # --- 3a. Standardise Trail_Condition ---
    df["Trail_Condition"] = (
        df["Trail_Condition"]
        .astype(str).str.strip().str.capitalize()
    )
    known_conditions = {"Dry", "Wet", "Muddy", "Rocky"}
    df["Trail_Condition"] = df["Trail_Condition"].where(
        df["Trail_Condition"].isin(known_conditions), other=np.nan
    )
    df["Trail_Condition"].fillna(df["Trail_Condition"].mode()[0], inplace=True)

    # --- 3b. IQR outlier capping ---
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 3*IQR, upper=Q3 + 3*IQR)

    # --- 3c. Median imputation ---
    for col in NUMERIC_COLS:
        df[col].fillna(df[col].median(), inplace=True)

    # --- 3d. Binary encode Traction_Feedback (qualitative) ---
    traction_map = {
        "Excellent": 1, "Good": 1,
        "Slipped Once": 0, "Felt Unstable": 0,
    }
    df["Traction_Feedback"] = (
        df["Traction_Feedback"]
        .astype(str).str.strip().str.title()
        .map(traction_map)
    )
    df["Traction_Feedback"].fillna(df["Traction_Feedback"].median(), inplace=True)

    # --- 3e. Binary encode Would_Recommend (qualitative) ---
    recommend_map = {"Yes": 1, "Maybe": 0.5, "No": 0}
    df["Would_Recommend"] = (
        df["Would_Recommend"]
        .astype(str).str.strip().str.title()
        .map(recommend_map)
    )
    df["Would_Recommend"].fillna(df["Would_Recommend"].median(), inplace=True)

    # --- 3f. Surface condition binary flags ---
    df["Is_Wet"]   = df["Trail_Condition"].isin(["Wet", "Muddy"]).astype(int)
    df["Is_Muddy"] = (df["Trail_Condition"] == "Muddy").astype(int)
    df["Is_Rocky"] = (df["Trail_Condition"] == "Rocky").astype(int)

    # --- 3g. Drop rows where target is still missing ---
    df.dropna(subset=["Stability_Rating"], inplace=True)

    print(f"✔  Cleaning done → {raw_rows} → {len(df)} rows kept "
          f"({raw_rows - len(df)} dropped)")
    return df


# Step 4: Machine Learning using Random Forest

FEATURE_COLS = [
    # Shoe specs — quantitative
    "Stack_Height_mm", "Midsole_Firmness", "Lug_Depth_mm",
    "Heel_Toe_Drop_mm", "Outsole_Hardness",
    # Biomechanics — quantitative
    "Cadence_spm", "Ground_Contact_ms",
    "Session_Duration_min", "Elevation_Gain_ft",
    # Runner self-report — qualitative (encoded)
    "Perceived_Comfort", "Confidence_On_Descent",
    "Traction_Feedback", "Would_Recommend",
    # Surface flags
    "Is_Wet", "Is_Muddy", "Is_Rocky",
]
TARGET_COL = "Stability_Rating"

def train_model(df):
    """Train a RandomForestRegressor to predict Stability_Rating."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"✔  Model trained  →  MAE: {mae:.3f}  |  R²: {r2:.3f}")
    return rf


# Step 5: Interactive Visualization

RUN_BLUE   = "#003087"
RUN_TEAL   = "#00AEEF"
RUN_ORANGE = "#F47920"
RUN_LIGHT  = "#E8F4FD"

FEATURE_LABELS = {
    "Stack_Height_mm":       "Stack Height (mm)",
    "Midsole_Firmness":      "Midsole Firmness",
    "Lug_Depth_mm":          "Lug Depth (mm)",
    "Heel_Toe_Drop_mm":      "Heel-Toe Drop (mm)",
    "Outsole_Hardness":      "Outsole Hardness",
    "Cadence_spm":           "Cadence (spm)",
    "Ground_Contact_ms":     "Ground Contact (ms)",
    "Session_Duration_min":  "Session Duration (min)",
    "Elevation_Gain_ft":     "Elevation Gain (ft)",
    "Perceived_Comfort":     "★ Perceived Comfort",
    "Confidence_On_Descent": "★ Confidence on Descent",
    "Traction_Feedback":     "★ Traction Feedback",
    "Would_Recommend":       "★ Would Recommend",
    "Is_Wet":                "Surface: Wet",
    "Is_Muddy":              "Surface: Muddy",
    "Is_Rocky":              "Surface: Rocky",
}

QUAL_FEATURES = {
    "Perceived_Comfort", "Confidence_On_Descent",
    "Traction_Feedback", "Would_Recommend"
}

def plot_feature_importance(model, output_path):
    """
    Horizontal bar chart distinguishing quantitative vs qualitative features.
    Qualitative features marked with ★ and coloured in light teal.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)
    sorted_features   = [FEATURE_COLS[i] for i in indices]
    sorted_importance = importances[indices]
    labels = [FEATURE_LABELS.get(f, f) for f in sorted_features]

    colors = []
    for i, feat in enumerate(sorted_features):
        if i == len(sorted_features) - 1:
            colors.append(RUN_ORANGE)
        elif feat in QUAL_FEATURES:
            colors.append("#7BC8F6")
        elif sorted_importance[i] > np.median(sorted_importance):
            colors.append(RUN_TEAL)
        else:
            colors.append(RUN_BLUE)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(RUN_LIGHT)
    ax.set_facecolor(RUN_LIGHT)

    bars = ax.barh(labels, sorted_importance, color=colors,
                   edgecolor="white", linewidth=0.8, height=0.62)

    for bar, val in zip(bars, sorted_importance):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left",
                fontsize=9, color=RUN_BLUE, fontweight="bold")

    ax.set_xlabel("Mean Decrease in Impurity (Importance)",
                  fontsize=11, color=RUN_BLUE, labelpad=8)
    ax.set_title(
        "Feature Importance for Trail Stability Prediction\n"
        "Quantitative + Qualitative Runner Data  ·  RandomForestRegressor",
        fontsize=13, color=RUN_BLUE, fontweight="bold", pad=14
    )
    ax.tick_params(axis="y", labelsize=9.5, colors=RUN_BLUE)
    ax.tick_params(axis="x", labelsize=9,   colors=RUN_BLUE)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#BDD7EE")
    ax.set_xlim(0, sorted_importance.max() * 1.30)

    legend_handles = [
        mpatches.Patch(color=RUN_ORANGE, label="Top Feature"),
        mpatches.Patch(color=RUN_TEAL,   label="High Importance — Quantitative"),
        mpatches.Patch(color="#7BC8F6",      label="★ Qualitative — Runner Self-Report"),
        mpatches.Patch(color=RUN_BLUE,    label="Lower Importance"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=8.5, framealpha=0.6, edgecolor="white")

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"✔  Feature importance chart saved → {output_path}")


# Step 6: Export dataset for further data visualization in Power BI

def export_for_powerbi(df, model):
    """Append model predictions and export clean dataset for Power BI."""
    export_df = df.copy()
    export_df["Predicted_Stability"] = model.predict(df[FEATURE_COLS]).round(2)

    export_path = f"{OUTPUT_DIR}/rundata_powerbi_ready.csv"
    export_df.to_csv(export_path, index=False)
    print(f"✔  Power BI export saved → {export_path}")
    return export_df


# Main Pipeline

def main():
    print("\n" + "="*62)
    print("  RUNNING Data — TRAIL STABILITY PIPELINE")
    print("  Quantitative + Qualitative Runner Data")
    print("="*62 + "\n")

    print("[ STEP 1 ] Loading raw CSVs from disk...")
    shoe_specs_df, runner_tests_df = load_raw_data()

    print("\n[ STEP 2 ] SQL JOIN via SQLite...")
    joined_df = sql_join(shoe_specs_df, runner_tests_df)

    print("\n[ STEP 3 ] Cleaning data...")
    clean_df = clean_data(joined_df)

    print("\n[ STEP 4 ] Training RandomForestRegressor...")
    model = train_model(clean_df)

    print("\n[ STEP 5 ] Plotting feature importance...")
    chart_path = f"{OUTPUT_DIR}/feature_importance.png"
    plot_feature_importance(model, chart_path)

    print("\n[ STEP 6 ] Exporting Power BI-ready CSV...")
    final_df = export_for_powerbi(clean_df, model)

    print("\n" + "="*62)
    print("  PIPELINE COMPLETE  →  ./RUN_output/")
    print("="*62 + "\n")

    preview_cols = [
        "Runner_ID", "Shoe_Model", "Trail_Condition",
        "Cadence_spm", "Perceived_Comfort", "Confidence_On_Descent",
        "Traction_Feedback", "Stability_Rating", "Predicted_Stability"
    ]
    print("Final dataset preview (5 rows):")
    print(final_df[preview_cols].head().to_string(index=False))


if __name__ == "__main__":
    main()
