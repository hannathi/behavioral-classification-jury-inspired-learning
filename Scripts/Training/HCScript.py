import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import optuna
import os
import joblib
import json
from datetime import datetime

DATA_PATH = r"your\path\to\input\parquet"  # specify the path to the input parquet file (should contain the full dataset with annotations and features)
DB_FOLDER = r"your\path\to\output\dir"  # specify the path to the output directory  
PARAMS_PATH = os.path.join(DB_FOLDER, "best_params_hc.json")
DB_PATH = f"sqlite:///{os.path.join(DB_FOLDER, 'optuna_study.db')}"
RANDOMSEED = 42
RUN_OPTUNA = True # set to False to skip hyperparameter optimization and use previously saved best parameters
N_TRIALS = 500  

FEATURES = [
    "Centroid_distance", "Nose_to_nose_distance", 
    "M1_Nose_to_M2_lat_left", "M1_Nose_to_M2_lat_right", 
    "M2_Nose_to_M1_lat_left", "M2_Nose_to_M1_lat_right", 
    "Movement_mouse_1_centroid", "Movement_mouse_2_centroid", 
    "Center_1_x", "Center_1_y", "Center_2_x", "Center_2_y", 
    "rel_speed", "nose1_to_body2", "nose2_to_body1", 
    "pursuit_cos_1", "pursuit_cos_2"
]

# Majoirty vote logic for label assignment, with random tie-breaking
def build_high_confidence_labels(df):
    df = df.copy()
    df["target"] = np.nan
    
    df.loc[(df["is_annotated"] == 0) | (df["event_number"] < 0), "target"] = 0
    
    # Extract real annotations
    ann = df[(df["is_annotated"] == 1) & (df["event_number"] >= 0)].copy()
    if len(ann) == 0: 
        return df
        
    # Count votes per event and label
    counts = ann.groupby(["event_number", "event_label"]).size().reset_index(name="n_votes")
    
    # Shuffle counts to ensure random tie-breaking
    counts_shuffled = counts.sample(frac=1, random_state=RANDOMSEED).reset_index(drop=True)
    
    # Because we shuffled first, tied vote counts will have a random order.
    final_labels = (
        counts_shuffled.sort_values(by="n_votes", ascending=False)
        .drop_duplicates(subset=["event_number"])
    )
    
    # map labels
    label_map = final_labels.set_index("event_number")["event_label"]
    df.loc[df["target"].isna(), "target"] = df["event_number"].map(label_map)
    
    return df.dropna(subset=["target"])

def aggregate_rows(df):
    df_agg = df.drop_duplicates(subset=["event_number"]).copy()
    cols_to_drop = ["annotator_id", "event_label", "is_annotated"]
    df_agg = df_agg.drop(columns=[c for c in cols_to_drop if c in df_agg.columns])
    return df_agg.reset_index(drop=True)

# Optuna study
def objective(trial, X, y, groups):

    params = {
        "verbosity": 0,
        "objective": "multi:softprob",
        "num_class": 3,
        "tree_method": "hist",
        "random_state": RANDOMSEED,
        "n_estimators": 500, 
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.75, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "gamma": trial.suggest_float("gamma", 0.01, 10.0, log=True),
    }

    sgkf = StratifiedGroupKFold(n_splits=5)
    cv_scores = []

    for train_idx, val_idx in sgkf.split(X, y, groups=groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        weights = compute_sample_weight(class_weight='balanced', y=y_train)
        model = xgb.XGBClassifier(**params)
        
        model.fit(X_train, y_train, sample_weight=weights)
        preds = model.predict(X_val)
        score = f1_score(y_val, preds, average='macro')
        cv_scores.append(score)
        
        trial.report(score, step=len(cv_scores))
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(cv_scores)


# main training function

def run_train_model(full_df):
    train_df = full_df[full_df["dataset"] == "training"].copy()
    test_df_raw = full_df[full_df["dataset"] == "testing"].copy()
    
    df_hc_train = build_high_confidence_labels(train_df)
    df_agg_train = aggregate_rows(df_hc_train)
    X = df_agg_train[FEATURES]
    y = df_agg_train["target"]
    groups = df_agg_train["group_id"]

    if RUN_OPTUNA:
        print(f"\n Starting HC Optuna ({N_TRIALS} trials) ")
        study = optuna.create_study(
            study_name="hc_model_optimization", 
            storage=DB_PATH, 
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
            load_if_exists=True
        )
        study.optimize(lambda trial: objective(trial, X, y, groups), n_trials=N_TRIALS)
        best_params = study.best_params
    else:
        try:
            study = optuna.load_study(study_name="hc_model_optimization", storage=DB_PATH)
            best_params = study.best_params
            print(f"Loaded best params from DB: {best_params}")
        except:
            print("No DB found, using defaults.")
            best_params = {"n_estimators": 500, "learning_rate": 0.1, "max_depth": 5}

    print(f"\n--- Training Final HC Model with params: {best_params} ---")
    final_weights = compute_sample_weight(class_weight='balanced', y=y)
    final_model = xgb.XGBClassifier(**best_params, tree_method="hist", random_state=RANDOMSEED)
    final_model.fit(X, y, sample_weight=final_weights)

    # overfitting check on training set
    print("\n" + "="*20 + " OVERFITTING CHECK: TRAINING SET " + "="*20)
    train_preds = final_model.predict(X)
    print("\n[TRAINING SET PERFORMANCE]")
    print(classification_report(y, train_preds, target_names=["No Event", "Follow", "Chase"]))
    train_f1 = f1_score(y, train_preds, average='macro')

    # evaluation on test set
    test_hc = build_high_confidence_labels(test_df_raw)
    test_final = aggregate_rows(test_hc)
    X_test = test_final[FEATURES]
    y_test = test_final["target"]
    
    preds_test = final_model.predict(X_test)
    test_f1 = f1_score(y_test, preds_test, average='macro')
    
    print("\n" + "="*20 + " FINAL EVALUATION: TEST SET " + "="*20)
    print("\n[TEST SET PERFORMANCE]")
    print(classification_report(y_test, preds_test, target_names=["No Event", "Follow", "Chase"]))
    
    print("\nConfusion Matrix (Test):")
    print(pd.DataFrame(confusion_matrix(y_test, preds_test), 
                       index=['True 0', 'True 1', 'True 2'], 
                       columns=['Pred 0', 'Pred 1', 'Pred 2']))
    
    # summary
    print(f"\nFinal Overfitting Gap (Macro F1): {train_f1 - test_f1:.4f}")

    # save model and params
    joblib.dump(final_model, os.path.join(DB_FOLDER, "hc_xgb_model.joblib"))
    with open(os.path.join(DB_FOLDER, "features.json"), 'w') as f:
        json.dump(FEATURES, f)

if __name__ == "__main__": 
    df = pd.read_parquet(DATA_PATH)
    run_train_model(df)     
