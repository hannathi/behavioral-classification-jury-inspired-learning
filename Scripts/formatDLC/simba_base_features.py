#%%
import pandas as pd
import numpy as np

#%%
# CONFIG
PARQUET_PATH = r"your\path\to\input\parquet"  # specify the input parquet file path, should be a pairwise parquet file with features extracted for two mice
PX_PER_MM = 0.59 

ANIMALS = ["1", "2"]
BODYPARTS = ["Ear_left","Ear_right","Nose", "Center", "Lat_left", "Lat_right","Tail_base","Tail_end"]  
COORDS = ["x", "y", "p"]
#%%
# HELPERS
def euclidean_distance(x1, x2, y1, y2, px_per_mm):
    '''Compute Euclidean distance in millimeters between two body-parts.'''
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / px_per_mm

def movement(x, y, px_per_mm):
    dx = x.diff().fillna(0)
    dy = y.diff().fillna(0)
    return np.sqrt(dx**2 + dy**2) / px_per_mm

def get_bp(df, animal, bp):
    """Helper to extract x,y columns"""
    return df[f"{bp}_{animal}_x"], df[f"{bp}_{animal}_y"]
#%%
# LOAD DATA
df = pd.read_parquet(PARQUET_PATH)  
print(df.shape)
print(df)
'''
# If DLC multiindex → flatten
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ["_".join(col).strip() for col in df.columns.values]
'''
# RENAME COLUMNS BASED ON ORDER
expected_cols = []
for animal in ANIMALS:
    for bp in BODYPARTS:
        for coord in COORDS:
            expected_cols.append(f"{bp}_{animal}_{coord}")

# Check length match
if len(expected_cols) != len(df.columns):
    raise ValueError(
        f"Column mismatch! Expected {len(expected_cols)}, got {len(df.columns)}"
    )

df.columns = expected_cols
#%%
# EXTRACT BODY PARTS
M1_nose_x, M1_nose_y = get_bp(df, "1", "Nose")
M2_nose_x, M2_nose_y = get_bp(df, "2", "Nose")

M1_cent_x, M1_cent_y = get_bp(df, "1", "Center")
M2_cent_x, M2_cent_y = get_bp(df, "2", "Center")

M2_latL_x, M2_latL_y = get_bp(df, "2", "Lat_left")
M2_latR_x, M2_latR_y = get_bp(df, "2", "Lat_right")

M1_latL_x, M1_latL_y = get_bp(df, "1", "Lat_left")
M1_latR_x, M1_latR_y = get_bp(df, "1", "Lat_right")


#%%
# COMPUTE FEATURES
features = pd.DataFrame(index=df.index)

# Distances
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

# Movement (centroid-based)
features["Movement_mouse_1_centroid"] = movement(
    M1_cent_x, M1_cent_y, PX_PER_MM
)

features["Movement_mouse_2_centroid"] = movement(
    M2_cent_x, M2_cent_y, PX_PER_MM
)

# optional: save shifted data (xx_shifted = xx.shift(1).add_suffix("_shifted").fillna(0)), then calculate movement by movement = distance between bp_shifted and bp

#%%
SIMBA_BASE_FEATURES = [
    "Centroid_distance",
    "Nose_to_nose_distance",

    "M1_Nose_to_M2_lat_left",
    "M1_Nose_to_M2_lat_right",
    "M2_Nose_to_M1_lat_left",
    "M2_Nose_to_M1_lat_right",

    "Movement_mouse_1_centroid",
    "Movement_mouse_2_centroid",
]
'''
Definitions
Centroid_distance	distance between the centroid of the CD1 and centroid of the C57
Nose_to_nose_distance	distance between the nose of the CD1 and nose of the C57
M1_Nose_to_M2_lat_left	distance between the nose of the CD1 and lateral left side of the C57
M1_Nose_to_M2_lat_right	distance between the nose of the CD1 and lateral right side of the C57
M2_Nose_to_M1_lat_left	distance between the nose of the C57 and lateral left side of the CD1
M2_Nose_to_M1_lat_right	distance between the nose of the C57 and lateral right side of the CD1
Movement_mouse_1_centroid	movements of CD1 centroid from previous frame
Movement_mouse_2_centroid	movements of C57 centroid from previous frame
'''
#%%
import os

# OUTPUT FOLDER
SAVE_DIR = r"your\path\to\output\parquet"  # specify the output parquet folder path
os.makedirs(SAVE_DIR, exist_ok=True)

# create output filename based on input file
base_name = os.path.basename(PARQUET_PATH).replace(".parquet", "")
out_path = os.path.join(SAVE_DIR, f"{base_name}_simba_features_calculated.parquet")

features_to_save = features.copy()

# save
features_to_save.to_parquet(out_path, index=True)

print(f"Saved features to: {out_path}")
print(features_to_save.shape)

# %%
