"""
This script creates pairwise parquet files and optionally duplicates videos.
Designed to run directly with hardcoded paths.
"""

import os
import shutil
import pandas as pd


# input and output paths for videos (mp4) and parquet files
INPUT_VIDEOS = r"your\path\to\input\videos"  # specify the input video folder path
OUTPUT_VIDEOS = r"your\path\to\output\videos"  # specify the output video folder path

INPUT_PARQUET = r"your\path\to\input\parquet"  # specify the input parquet folder path
OUTPUT_PARQUET = r"your\path\to\output\parquet"  # specify the output parquet folder path

CREATE_VIDEOS = True   # set True if you want video duplication


def duplicate_video(input_folder, output_folder):
    pairs = ['BG', 'BY', 'RB', 'RG', 'RY', 'GY'] # define all pairwise combinations of mice

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            print(f"Copying {filename}:")
            src_file = os.path.join(input_folder, filename)

            for pair in pairs:
                dest_file = os.path.join(output_folder, f"{pair}_{filename}")
                shutil.copy2(src_file, dest_file)
                print(f"  -> {pair}")

            print(f"{filename} copied.\n")



def subset_df2(df, mouse=None, bp=None, coord=None, index_list=None):
    """Subset parquet files without relying on column names."""
    n_cols = len(df.columns)

    mice = ['Red', 'Blue', 'Green', 'Yellow']
    body_parts = [
        '1_earL', '2_earR', '3_nose', '4_centre',
        '5_centreL', '6_centreR', '7_tailBase', '8_tailTip'
    ]
    coords = ['x', 'y', 'likelihood']

    mask = pd.Series(False, index=range(n_cols))

    for i in range(n_cols):
        m = mice[i // 24]
        b = body_parts[(i % 24) // 3]
        c = coords[(i % 24) % 3]

        cond = True
        if mouse is not None:
            cond &= m in mouse
        if bp is not None:
            cond &= b in bp
        if coord is not None:
            cond &= c in coord

        mask[i] = cond

    subset = df.iloc[:, mask.values]

    if index_list is not None:
        subset = subset.loc[index_list]

    return subset




def parquet_by_pair(input_folder, output_folder):
    mouse_dict = {'B': 'Blue', 'G': 'Green', 'R': 'Red', 'Y': 'Yellow'}
    pairs = ['BG', 'BY', 'RB', 'RG', 'RY', 'GY']

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.parquet'):
            print(f'Processing {filename}')
            src_file = os.path.join(input_folder, filename)

            df_all = pd.read_parquet(src_file)

            for pair in pairs:
                mouse_list = [mouse_dict[i] for i in pair]

                df_pair = subset_df2(df_all, mouse=mouse_list)

                dest_file = os.path.join(output_folder, f'{pair}_{filename}')
                df_pair.to_parquet(dest_file)

                print(f"  saved {pair}")

            print(f'{filename} completed.\n')


def main():
    print("STARTING PAIRWISE PARQUET CREATION AND VIDEO DUPLICATION")

    if CREATE_VIDEOS:
        duplicate_video(INPUT_VIDEOS, OUTPUT_VIDEOS)

    parquet_by_pair(INPUT_PARQUET, OUTPUT_PARQUET)

    print("Done.")


if __name__ == "__main__":
    main()