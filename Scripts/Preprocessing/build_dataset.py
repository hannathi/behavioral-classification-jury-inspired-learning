import os
import pandas as pd
import numpy as np
from pathlib import Path


ANN_PATH = r"your\path\to\annotation\excel"  # specify the path to the annotation Excel file

DATACREATION_SET = r"your\path\to\data\creation\set"  # specify the path to the data creation set

OUTPUT_DIR = r"your\path\to\output\dir"  # specify the path to the output directory

SHEET_NAME = "FullDataset" # change if you want to use a different sheet in the annotation Excel file
DATASET_NAME = None  # for example "testing" to filter to a specific dataset, or None for all datasets

RANDOM_SEED = 42
MIN_NEG_PER_VIDEO = 50



# HELPERS
def sample_negative_events_in_mask(frames, any_annotated_mask, df, n_neg, rng, mask, min_len=12):
    negatives = []
    idx_pool = np.where(mask)[0]

    if len(idx_pool) == 0:
        return negatives

    feature_df = df.select_dtypes(include=[np.number])
    max_attempts = n_neg * 5
    attempts = 0

    while len(negatives) < n_neg and attempts < max_attempts:
        attempts += 1
        i = rng.choice(idx_pool)
        dur = rng.integers(min_len, min_len * 4)
        j = min(i + dur, len(df) - 1)

        if not mask[i:j + 1].all() or any_annotated_mask[i:j + 1].any():
            continue

        feats = feature_df.iloc[i:j + 1].mean(axis=0).copy()
        negatives.append((feats, i, j))

    if len(negatives) < n_neg:
        print(f"Warning: Only generated {len(negatives)}/{n_neg} negatives")

    return negatives


def build_near_far_masks(frames, annotated_intervals, buffer_frames=50, near_band=500):
    n = len(frames)
    any_annotated_mask = np.zeros(n, dtype=bool)

    for start, end in annotated_intervals:
        any_annotated_mask[start:end + 1] = True

    near_mask = np.zeros(n, dtype=bool)
    buffer_mask = np.zeros(n, dtype=bool)

    for start, end in annotated_intervals:
        near_region = (
            ((frames >= start - near_band) & (frames < start - buffer_frames)) |
            ((frames > end + buffer_frames) & (frames <= end + near_band))
        )
        near_mask |= near_region

        buffer_region = (frames >= start - buffer_frames) & (frames <= end + buffer_frames)
        buffer_mask |= buffer_region

    near_mask &= ~any_annotated_mask
    near_mask &= ~buffer_mask
    far_mask = (~near_mask) & (~any_annotated_mask)

    return near_mask, far_mask



# main function
def build_dataset(
    ann_path,
    enriched_dir,
    save_dir,
    dataset_name=None,
    sheet_name="Sheet1",
    test_frac=0.2
):

    rng = np.random.default_rng(RANDOM_SEED)
    ann = pd.read_excel(ann_path, sheet_name=sheet_name)

    if dataset_name:
        ann = ann[ann["dataset"] == dataset_name].copy()


    # train/test split at video level
    videos = ann[["video", "pair"]].drop_duplicates().reset_index(drop=True)
    n_test = max(1, int(len(videos) * test_frac))

    test_idx = rng.choice(len(videos), size=n_test, replace=False)

    videos["dataset_split"] = "training"
    videos.loc[test_idx, "dataset_split"] = "testing"

    ann = ann.merge(videos, on=["video", "pair"], how="left")

    meta_cols = [
        "video", "pair", "start", "end",
        "event_number", "annotator_id", "event_label",
        "group_id", "is_annotated", "dataset"
    ]

    event_rows = []
    synthetic_counter = -1

    for (video, pair), group_ann in ann.groupby(["video", "pair"]):

        fname = f"feature_enriched_{pair}_{video}.parquet"
        fpath = Path(enriched_dir) / fname

        if not fpath.exists():
            print(f"Skipping missing file: {fname}")
            continue

        df = pd.read_parquet(fpath)
        frames = np.arange(len(df))
        feature_df = df.select_dtypes(include=[np.number])

        annotated_intervals = []

        for (event_number, annotator_id), ev_group in group_ann.groupby(
            ["event_number", "annotator_id"]
        ):
            start_idx = int(ev_group["start"].min())
            end_idx = int(ev_group["end"].max())

            feats = feature_df.iloc[start_idx:end_idx + 1].mean(axis=0).copy()

            row = feats.to_dict()
            row.update({
                "video": video,
                "pair": pair,
                "start": start_idx,
                "end": end_idx,
                "event_number": event_number,
                "annotator_id": annotator_id,
                "event_label": ev_group["event_label"].iloc[0],
                "is_annotated": 1,
                "group_id": f"{video}_{pair}",
                "dataset": ev_group["dataset_split"].iloc[0]
            })

            event_rows.append(row)
            annotated_intervals.append((start_idx, end_idx))

        # negative sampling
        near_mask, far_mask = build_near_far_masks(frames, annotated_intervals)

        n_total_neg = MIN_NEG_PER_VIDEO
        n_neg_near = max(1, int(n_total_neg * 0.05))
        n_neg_far = n_total_neg - n_neg_near

        any_annotated_mask = np.zeros(len(df), dtype=bool)
        for start, end in annotated_intervals:
            any_annotated_mask[start:end + 1] = True

        negatives_near = sample_negative_events_in_mask(
            frames, any_annotated_mask, df, n_neg_near, rng, near_mask
        )

        negatives_far = sample_negative_events_in_mask(
            frames, any_annotated_mask, df, n_neg_far, rng, far_mask
        )

        for neg_feats, start_idx, end_idx in negatives_near + negatives_far:

            row = neg_feats.to_dict()
            row.update({
                "video": video,
                "pair": pair,
                "start": start_idx,
                "end": end_idx,

                # global unique ID
                "event_number": synthetic_counter, # negative events get unique negative IDs counting down from -1

                "annotator_id": "none", # negatives are not tied to a specific annotator
                "event_label": 0, # negatives get label 0 (no event)
                "is_annotated": 0, # negatives = 0 
                "group_id": f"{video}_{pair}",
                "dataset": group_ann["dataset_split"].iloc[0]
            })

            event_rows.append(row)

            synthetic_counter -= 1

    # finalize dataset
    dataset = pd.DataFrame(event_rows)

    for col in meta_cols:
        if col not in dataset.columns:
            dataset[col] = np.nan

    feature_cols = [c for c in dataset.columns if c not in meta_cols]
    dataset = dataset[meta_cols + feature_cols]

    os.makedirs(save_dir, exist_ok=True)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")

    out_parquet = Path(save_dir) / f"final_training_set_{timestamp}.parquet"
    out_excel = out_parquet.with_suffix(".xlsx")

    dataset.to_parquet(out_parquet, index=False)
    dataset.to_excel(out_excel, index=False)

    print("\nDataset created!")
    print(f"Saved: {out_parquet}")
    print(f"Rows: {len(dataset)}")
    print(dataset["is_annotated"].value_counts())
    print(dataset["dataset"].value_counts())

    return dataset

if __name__ == "__main__":
    build_dataset(
        ANN_PATH,
        DATACREATION_SET,
        OUTPUT_DIR,
        dataset_name=DATASET_NAME,
        sheet_name=SHEET_NAME
    )