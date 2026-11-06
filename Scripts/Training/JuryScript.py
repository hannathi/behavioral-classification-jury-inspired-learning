"""
This is the model utilizing Jury learning
"""
import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
import itertools
import optuna
import numpy as np
import json
import joblib
from datetime import datetime

DATA_PATH = r"your\path\to\input\parquet"  # specify the path to the input parquet file (should contain the full dataset with annotations and features)
DB_FOLDER = r"your\path\to\output\dir"  # specify the path to the output directory
PARAMS_PATH = os.path.join(DB_FOLDER, "best_params_jury.json")
DB_PATH = f"sqlite:///{os.path.join(DB_FOLDER, 'jury_optuna_study.db')}"
MODEL_DIR = os.path.join(DB_FOLDER, "jury_models")

RANDOMSEED = 42
RUN_OPTUNA = False # set to False to skip hyperparameter optimization and use previously saved best parameters
N_TRIALS = 500

FEATURES = [
    'Centroid_distance', 'Nose_to_nose_distance', 'M1_Nose_to_M2_lat_left',
    'M1_Nose_to_M2_lat_right', 'M2_Nose_to_M1_lat_left', 'M2_Nose_to_M1_lat_right',
    'Movement_mouse_1_centroid', 'Movement_mouse_2_centroid', 'Center_1_x',
    'Center_1_y', 'Center_2_x', 'Center_2_y', 'rel_speed', 'nose1_to_body2',
    'nose2_to_body1', 'pursuit_cos_1', 'pursuit_cos_2'
]

def save_best_params(params):
    os.makedirs(os.path.dirname(PARAMS_PATH), exist_ok=True)
    with open(PARAMS_PATH, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Best parameters saved to {PARAMS_PATH}")

def save_jury_models(jury_models):
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    for annotator, model in jury_models.items():
        path = os.path.join(MODEL_DIR, f"juror_{annotator}.joblib")
        joblib.dump(model, path)
    
    print(f"Saved {len(jury_models)} juror models to {MODEL_DIR}")

def save_probabilities(df, probs, filename):
    prob_df = pd.DataFrame(probs, columns=["p_no_event", "p_follow", "p_chase"])
    out = pd.concat([df.reset_index(drop=True), prob_df], axis=1)
    
    path = os.path.join(DB_FOLDER, filename)
    out.to_parquet(path, index=False)
    
    print(f"Saved probabilities to {path}")

def load_best_params_from_db():
    try:
        study = optuna.load_study(study_name="jury_model_optimization", storage=DB_PATH)
        
        return study.best_params
    except Exception:
        return None

def build_high_confidence_labels(df):
    """Majority vote logic for evaluation labels."""
    df = df.copy()
    df["target"] = np.nan
    df.loc[(df["is_annotated"] == 0) | (df["event_number"] < 0), "target"] = 0
    ann = df[(df["is_annotated"] == 1) & (df["event_number"] >= 0)].copy()
    if len(ann) == 0: return df
    
    counts = ann.groupby(["event_number", "event_label"]).size().reset_index(name="n_votes")
    final_labels = counts.sort_values(by=["n_votes", "event_label"], ascending=[False, True]).drop_duplicates(subset=["event_number"])
    label_map = final_labels.set_index("event_number")["event_label"]
    df.loc[df["target"].isna(), "target"] = df["event_number"].map(label_map)
    return df.dropna(subset=["target"])

def evaluate_individual_jurors(jury_models, test_df):
    print("\n   Individual Juror Evaluation (WITH shared negatives) ")
    
    negative_pool = test_df[(test_df["is_annotated"] == 0) | (test_df["event_number"] < 0)].copy()
    
    for annotator, model in jury_models.items():
        
        # positive samples for this annotator
        ant_pos = test_df[test_df["annotator_id"] == annotator].copy()
        
        if ant_pos.empty:
            continue
        
        relevant_groups = ant_pos["group_id"].unique()
        
        # add corresponding negatives for this annotator's groups
        ant_neg = negative_pool[negative_pool["group_id"].isin(relevant_groups)].copy()
        
        juror_df = pd.concat([ant_pos, ant_neg], axis=0).dropna(subset=["event_label"])
        
        if len(juror_df) < 30:
            print(f"\n   Juror: {annotator} (SKIPPED - too few samples: {len(juror_df)}) ")
            continue
        
        X = juror_df[FEATURES]
        y_true = juror_df["event_label"].astype(int)
        
        y_pred = model.predict(X)
        
        print(f"\n   Juror: {annotator} ")
        print(f"Samples: {len(juror_df)}")
        print("Class distribution:")
        print(y_true.value_counts())
        
        print(classification_report(
            y_true,
            y_pred,
            target_names=["No Event", "Follow", "Chase"],
            zero_division=0
        ))
        
        cm = confusion_matrix(y_true, y_pred)
        print(pd.DataFrame(cm, index=['True 0', 'True 1', 'True 2'], columns=['Pred 0', 'Pred 1', 'Pred 2']))

def aggregate_rows(df):
    df_agg = df.drop_duplicates(subset=["event_number"]).copy()
    cols_to_drop = ["annotator_id", "event_label", "is_annotated"]
    df_agg = df_agg.drop(columns=[c for c in cols_to_drop if c in df_agg.columns])
    return df_agg.reset_index(drop=True)

def train_jury_models(train_df, best_params=None, silent=True):
    """Trains individual XGB models for each juror."""
    if best_params is None:
        best_params = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5, "tree_method": "hist"}
    negative_pool = train_df[(train_df["is_annotated"] == 0) | (train_df["event_number"] < 0)].copy()
    annotators = [a for a in train_df["annotator_id"].unique() if pd.notna(a) and a != "none"]
    
    jury_models = {}
    for annotator in sorted(annotators):
        ant_pos = train_df[train_df["annotator_id"] == annotator].copy()
        relevant_groups = ant_pos["group_id"].unique()
        ant_neg = negative_pool[negative_pool["group_id"].isin(relevant_groups)].copy()
        
        ant_train_set = pd.concat([ant_pos, ant_neg], axis=0).dropna(subset=["event_label"])
        X = ant_train_set[FEATURES]
        y = ant_train_set["event_label"].astype(int)
        
        weights = compute_sample_weight(class_weight='balanced', y=y)
        
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            random_state=RANDOMSEED,
            **best_params
        )
        model.fit(X, y, sample_weight=weights)
        jury_models[annotator] = model
        if not silent: print(f"Juror {annotator} trained.")

    return jury_models

def run_jury_inference(jury_models, test_df, features, return_individual=False):
    X_test = test_df[features]
    
    individual_probs = {
        name: model.predict_proba(X_test)
        for name, model in jury_models.items()
    }

    annotator_names = list(jury_models.keys())
    jury_combinations = list(itertools.combinations(annotator_names, 3))
    
    all_jury_means = []
    for jury in jury_combinations:
        probs_stack = np.stack([individual_probs[name] for name in jury])
        all_jury_means.append(np.mean(probs_stack, axis=0))

    final_mean_probs = np.mean(all_jury_means, axis=0)
    final_preds = np.argmax(final_mean_probs, axis=1)

    if return_individual:
        return final_preds, final_mean_probs, individual_probs
    
    return final_preds, final_mean_probs

def objective(trial, train_df):
    params = {
        "verbosity": 0,
        "tree_method": "hist",
        "n_estimators": 500, 
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.75, 0.95), 
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0), 
        "gamma": trial.suggest_float("gamma", 0.01, 10.0, log=True), 
    }

    sgkf = StratifiedGroupKFold(n_splits=5)
    df_for_split = build_high_confidence_labels(train_df)
    df_for_split = aggregate_rows(df_for_split)
    
    cv_scores = []
    
    for fold, (t_idx, v_idx) in enumerate(sgkf.split(df_for_split, df_for_split["target"], groups=df_for_split["group_id"])):
        train_groups = df_for_split.iloc[t_idx]["group_id"].unique()
        val_groups = df_for_split.iloc[v_idx]["group_id"].unique()
        
        fold_train_df = train_df[train_df["group_id"].isin(train_groups)]
        fold_val_df = df_for_split[df_for_split["group_id"].isin(val_groups)]
        
        temp_jury = train_jury_models(fold_train_df, best_params=params, silent=True)
        preds, _ = run_jury_inference(temp_jury, fold_val_df, FEATURES)
        
        score = f1_score(fold_val_df["target"], preds, average='macro')
        cv_scores.append(score)
        
        trial.report(score, step=fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(cv_scores)

def run_train_model(full_df):
    train_df = full_df[full_df["dataset"] == "training"].copy()
    test_df_raw = full_df[full_df["dataset"] == "testing"].copy()

    test_df_hc = build_high_confidence_labels(test_df_raw)
    test_df_final = aggregate_rows(test_df_hc)

    if RUN_OPTUNA:
        print(f"\n Starting Optuna Study ({N_TRIALS} trials) ")
        study = optuna.create_study(
            study_name="jury_model_optimization", 
            storage=DB_PATH, 
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
            load_if_exists=True
        )
        study.optimize(lambda trial: objective(trial, train_df), n_trials=N_TRIALS)
        best_hyperparams = study.best_params
        save_best_params(best_hyperparams)
    else:
        best_hyperparams = load_best_params_from_db()
        if best_hyperparams is None:
            print("No DB found, using defaults.")
            best_hyperparams = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5}
    print(f"Loaded best params: {best_hyperparams}")

    print(f"\n   Training Final Jury Models with params: {best_hyperparams} ")
    final_jury = train_jury_models(train_df, best_params=best_hyperparams, silent=False)
    save_jury_models(final_jury)

    print("\n OVERFITTING CHECK: TRAINING SET ")
    train_df_hc = build_high_confidence_labels(train_df)
    train_df_final = aggregate_rows(train_df_hc)
    
    train_preds, _ = run_jury_inference(final_jury, train_df_final, FEATURES)
    
    print("\n[TRAINING SET PERFORMANCE]")
    print(classification_report(train_df_final["target"], train_preds, target_names=["No Event", "Follow", "Chase"]))
    train_f1 = f1_score(train_df_final["target"], train_preds, average='macro')

    # test set evaluation
    print("\n  FINAL EVALUATION: TEST SET ")
    final_preds, final_probs, individual_probs = run_jury_inference(
        final_jury,
        test_df_final,
        FEATURES,
        return_individual=True
    )
    
    print("\n[TEST SET PERFORMANCE]")
    print(classification_report(test_df_final["target"], final_preds, target_names=["No Event", "Follow", "Chase"]))
    test_f1 = f1_score(test_df_final["target"], final_preds, average='macro')

    # summary
    print(f"\nFinal Overfitting Gap (Macro F1): {train_f1 - test_f1:.4f}")

    print("\n   Final Consensus Evaluation on Test Set ")
    final_preds, final_probs, individual_probs = run_jury_inference(
        final_jury,
        test_df_final,
        FEATURES,
        return_individual=True
    )
    save_probabilities(test_df_final, final_probs, "jury_consensus_probs.parquet")
    for annotator, probs in individual_probs.items():
        save_probabilities(
            test_df_final,
            probs,
            f"juror_{annotator}_probs.parquet"
        )
    
    print("\n" + "-"*40)
    print(classification_report(test_df_final["target"], final_preds, target_names=["No Event", "Follow", "Chase"]))
    
    cm = confusion_matrix(test_df_final["target"], final_preds)
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=['True 0', 'True 1', 'True 2'], columns=['Pred 0', 'Pred 1', 'Pred 2']))
    evaluate_individual_jurors(final_jury, test_df_raw)

if __name__ == "__main__": 
    print("Running Jury inspired learning script")
    df = pd.read_parquet(DATA_PATH)
    run_train_model(df)