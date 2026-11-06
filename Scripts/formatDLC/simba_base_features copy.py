import os
import pandas as pd
import numpy as np

INPUT_DIR = r"your\path\to\input\parquet"  # specify the input parquet folder path, should be the pairwise parquet folder
SAVE_DIR = r"your\path\to\output\parquet"  # specify the output parquet folder path
PX_PER_MM = 0.59

ANIMALS = ["1", "2"]
BODYPARTS = ["Ear_left","Ear_right","Nose", "Center", "Lat_left", "Lat_right","Tail_base","Tail_end"]  
COORDS = ["x", "y", "p"]

os.makedirs(SAVE_DIR, exist_ok=True)

def add_center_coordinates(features, M1_cent_x, M1_cent_y, M2_cent_x, M2_cent_y):
    features["Center_1_x"] = M1_cent_x
    features["Center_1_y"] = M1_cent_y
    features["Center_2_x"] = M2_cent_x
    features["Center_2_y"] = M2_cent_y
    return features

def euclidean_distance(x1, x2, y1, y2, px_per_mm):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / px_per_mm

def movement(x, y, px_per_mm):
    dx = x.diff().fillna(0)
    dy = y.diff().fillna(0)
    return np.sqrt(dx**2 + dy**2) / px_per_mm

def get_bp(df, animal, bp):
    return df[f"{bp}_{animal}_x"], df[f"{bp}_{animal}_y"]

for file_name in os.listdir(INPUT_DIR):
    if not file_name.endswith(".parquet"):
        continue

    file_path = os.path.join(INPUT_DIR, file_name)
    print(f"\nProcessing: {file_name}")

    df = pd.read_parquet(file_path)

    expected_cols = []
    for animal in ANIMALS:
        for bp in BODYPARTS:
            for coord in COORDS:
                expected_cols.append(f"{bp}_{animal}_{coord}")

    if len(expected_cols) != len(df.columns):
        print(f"Skipping {file_name}: column mismatch")
        continue

    df.columns = expected_cols

    # EXTRACT BODY PARTS
    M1_nose_x, M1_nose_y = get_bp(df, "1", "Nose")
    M2_nose_x, M2_nose_y = get_bp(df, "2", "Nose")

    M1_cent_x, M1_cent_y = get_bp(df, "1", "Center")
    M2_cent_x, M2_cent_y = get_bp(df, "2", "Center")

    M2_latL_x, M2_latL_y = get_bp(df, "2", "Lat_left")
    M2_latR_x, M2_latR_y = get_bp(df, "2", "Lat_right")

    M1_latL_x, M1_latL_y = get_bp(df, "1", "Lat_left")
    M1_latR_x, M1_latR_y = get_bp(df, "1", "Lat_right")

    # COMPUTE FEATURES
    features = pd.DataFrame(index=df.index)

    features["Centroid_distance"] = euclidean_distance(
        M1_cent_x, M2_cent_x, M1_cent_y, M2_cent_y, PX_PER_MM
    )

    features["Nose_to_nose_distance"] = euclidean_distance(
        M1_nose_x, M2_nose_x, M1_nose_y, M2_nose_y, PX_PER_MM
    )

    features["M1_Nose_to_M2_lat_left"] = euclidean_distance(
        M1_nose_x, M2_latL_x, M1_nose_y, M2_latL_y, PX_PER_MM
    )

    features["M1_Nose_to_M2_lat_right"] = euclidean_distance(
        M1_nose_x, M2_latR_x, M1_nose_y, M2_latR_y, PX_PER_MM
    )

    features["M2_Nose_to_M1_lat_left"] = euclidean_distance(
        M2_nose_x, M1_latL_x, M2_nose_y, M1_latL_y, PX_PER_MM
    )

    features["M2_Nose_to_M1_lat_right"] = euclidean_distance(
        M2_nose_x, M1_latR_x, M2_nose_y, M1_latR_y, PX_PER_MM
    )

    features["Movement_mouse_1_centroid"] = movement(
        M1_cent_x, M1_cent_y, PX_PER_MM
    )

    features["Movement_mouse_2_centroid"] = movement(
        M2_cent_x, M2_cent_y, PX_PER_MM
    )
    features = add_center_coordinates(
        features, M1_cent_x, M1_cent_y, M2_cent_x, M2_cent_y
    )
    base_name = os.path.splitext(file_name)[0]
    out_path = os.path.join(
        SAVE_DIR, f"{base_name}_simba_features_calculated.parquet"
    )

    features.to_parquet(out_path, index=True)

    print(f"Saved: {out_path}")