"""
Extracts new features from parquet files and saves them for dataset creation. 
Also includes a debug step to compare expected files from Excel with actual files in the folder, and reports any mismatches.

"""


import pandas as pd
import numpy as np
from pathlib import Path

ANN_PATH = r"your\path\to\annotation\excel"  # specify the path to the annotation Excel file
INPUT_DIR = r"your\path\to\input\parquet"  # specify the path to the input parquet folder
OUTPUT_DIR = r"your\path\to\output\dir"  # specify the path to the output directory

SHEET_NAME = "FullDataset" # change if you want to use a different sheet in the annotation Excel file
DATASET_NAME = None # for example "testing" to filter to a specific dataset, or None for all datasets


def add_chase_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["rel_speed"] = np.abs(
        df["Movement_mouse_1_centroid"] - df["Movement_mouse_2_centroid"]
    )

    df["nose1_to_body2"] = np.minimum(
        df["M1_Nose_to_M2_lat_left"],
        df["M1_Nose_to_M2_lat_right"]
    )

    df["nose2_to_body1"] = np.minimum(
        df["M2_Nose_to_M1_lat_left"],
        df["M2_Nose_to_M1_lat_right"]
    )

    for i in [1, 2]:
        dx = df[f"Center_{i}_x"].diff().fillna(0)
        dy = df[f"Center_{i}_y"].diff().fillna(0)

        if i == 1:
            rx = df["Center_2_x"] - df["Center_1_x"]
            ry = df["Center_2_y"] - df["Center_1_y"]
        else:
            rx = df["Center_1_x"] - df["Center_2_x"]
            ry = df["Center_1_y"] - df["Center_2_y"]

        num = dx * rx + dy * ry
        denom = np.sqrt(dx**2 + dy**2) * np.sqrt(rx**2 + ry**2) + 1e-6

        df[f"pursuit_cos_{i}"] = num / denom

    return df


def run_feature_engineering():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    ann = pd.read_excel(ANN_PATH, sheet_name=SHEET_NAME)

    if DATASET_NAME and "dataset" in ann.columns:
        ann = ann[ann["dataset"] == DATASET_NAME]

    if ann.empty:
        print("No annotation data found.")
        return

    # get expected files from Excel file
    unique_pairs = ann[["video", "pair"]].drop_duplicates()

    expected_files = {
        f"{row.pair}_{row.video}_simba_features_calculated.parquet"
        for _, row in unique_pairs.iterrows()
    }

    # get actual files in the input directory
    input_files = {p.name for p in input_dir.glob("*.parquet")}

    # compare and report mismatches
    missing_files = expected_files - input_files
    extra_files = input_files - expected_files
    matching_files = expected_files & input_files

    print("\n FILE MISMATCH REPORT")
    print(f"Files in folder: {len(input_files)}")
    print(f"Expected from Excel: {len(expected_files)}")
    print(f"Matching files: {len(matching_files)}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Unused files: {len(extra_files)}")

    if missing_files:
        print("\n Missing files (in Excel but not folder):")
        for f in sorted(missing_files):
            print(f"  {f}")

    if extra_files:
        print("\n Unused files (in folder but not Excel):")
        for f in sorted(extra_files):
            print(f"  {f}")

    # save debug report
    debug_df = pd.DataFrame({
        "missing_files": pd.Series(list(missing_files)),
        "unused_files": pd.Series(list(extra_files)),
    })

    debug_path = output_dir / "DEBUG_file_mismatches.csv"
    debug_df.to_csv(debug_path, index=False)
    print(f"\nDebug report saved: {debug_path}")

    print("\nPROCESSING MATCHING FILES")

    processed = 0

    for fname in sorted(matching_files):
        fpath = input_dir / fname

        print(f"\nProcessing: {fname}")

        try:
            df = pd.read_parquet(fpath)
            df.columns = df.columns.str.strip()

            df_enriched = add_chase_features(df)

            out_file = output_dir / f"feature_enriched_{fname.replace('_simba_features_calculated', '')}"
            df_enriched.to_parquet(out_file, engine="pyarrow")

            print(f"Saved: {out_file.name} | rows={len(df_enriched)}")
            processed += 1

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    print("\nSUMMARY")
    print(f"Processed files: {processed}")
    print(f"Skipped (missing): {len(missing_files)}")
    print(f"Unused in folder: {len(extra_files)}")


if __name__ == "__main__":
    run_feature_engineering()