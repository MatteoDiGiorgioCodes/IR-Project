import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, ndcg_score
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_svmlight_file
import os


def qid_to_group(qid_array):
    _, counts = np.unique(qid_array, return_counts=True)
    return counts.tolist()

def ndcg_by_group(y_true, y_pred, group, k=10):
    start = 0
    scores = []
    for g in group:
        if g < 2:
            start += g
            continue  # Salta gruppi troppo piccoli
        y_t = y_true[start:start+g]
        y_p = y_pred[start:start+g]
        scores.append(ndcg_score([y_t], [y_p], k=k))
        start += g
    return np.mean(scores)

def load_mslr_dataset(path):

    X_train, y_train, qid_train = load_svmlight_file(f"{path}/train.txt", query_id=True)
    X_val, y_val, qid_val = load_svmlight_file(f"{path}/vali.txt", query_id=True)
    X_test, y_test, qid_test = load_svmlight_file(f"{path}/test.txt", query_id=True)

    return X_train, y_train, qid_to_group(qid_train), X_val, y_val, qid_to_group(qid_val), X_test, y_test, qid_to_group(qid_test)

def load_kdd2010_dataset(path):
        
    # Carica i dati in formato sparse
    X_train_full, y_train_full = load_svmlight_file(f"{path}/kddb")
    X_test, y_test = load_svmlight_file(f"{path}/kddb.t", n_features=X_train_full.shape[1])  # Forza lo stesso numero di feature

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def load_flight_delay_dataset(path):

    train_file = f"{path}/train-10m.csv"
    valid_file = f"{path}/valid.csv"
    test_file = f"{path}/test.csv"
    
    def preprocess(df, fit_columns=None):
        df = df.copy()
        y = df["dep_delayed_15min"].apply(lambda x: 1 if x == "Y" else 0)
        X = df.drop(columns=["dep_delayed_15min"])
        
        '''
        # Encode le feature categoriche
        for col in X.select_dtypes(include='object').columns:
            X[col] = LabelEncoder().fit_transform(X[col])
        '''
        X = pd.get_dummies(X)

        if fit_columns is not None:
            X = X.reindex(columns=fit_columns, fill_value=0)
            

        return X, y

    train_df = pd.read_csv(train_file)
    valid_df = pd.read_csv(valid_file)
    test_df = pd.read_csv(test_file)

    X_train, y_train = preprocess(train_df)
    X_valid, y_valid = preprocess(valid_df, X_train.columns)
    X_test, y_test = preprocess(test_df, X_train.columns)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def confront_function(x_tr, x_vl, x_ts, y_tr, y_vl, y_ts, my_task, my_base_params, my_xgb_model_hist, my_xgb_model_exa, a, b, nbr, group_tr=None, group_vl=None, group_ts=None, is_kdd=False):
    # ===== CONFIGURAZIONE =====
    task = my_task  # "classification" o "ranking"
    X_train, X_val, X_test, y_train, y_val, y_test = x_tr, x_vl, x_ts, y_tr, y_vl, y_ts

    # Convertiamo i dati in formato LightGBM
    if task == "ranking":
        train_data = lgb.Dataset(X_train, label=y_train, group=group_tr)
        validation_data = lgb.Dataset(X_val, label=y_val, group=group_vl)
        test_data = lgb.Dataset(X_test, label=y_test, group=group_ts)
    else:
        train_data = lgb.Dataset(X_train, label=y_train)
        validation_data = lgb.Dataset(X_val, label=y_val)
        test_data = lgb.Dataset(X_test, label=y_test)

    # ===== PARAMETRI BASE LIGHTGBM =====
    base_params = my_base_params

    # ===== XGBOOST HIST=====
    if not is_kdd:
        print("\nTraining XGBoost hist...")
        xgb_model_hist = my_xgb_model_hist

        start_time = time.time()

        if task == "ranking":
            xgb_model_hist.fit(
                X_train,
                y_train,
                group=group_tr,
                eval_set=[(X_val, y_val)],
                eval_group=[group_vl],
                verbose=nbr/10
            )
        else:
            xgb_model_hist.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=nbr/10
            )
        xgb_time = (time.time() - start_time) / xgb_model_hist.best_iteration
        y_pred_xgb = xgb_model_hist.predict_proba(X_test, iteration_range=(0, xgb_model_hist.best_iteration))[:, 1] if task == "classification" else xgb_model_hist.predict(X_test, iteration_range=(0, xgb_model_hist.best_iteration))
        xgb_score = roc_auc_score(y_test, y_pred_xgb) if task == "classification" else ndcg_by_group(y_test, y_pred_xgb, group_ts, k=10)
        print(f"XGBoost hist-> AUC/NDCG: {xgb_score:.4f} | Time: {xgb_time:.2f}s")

    # ===== XGBOOST EXA=====
    print("\nTraining XGBoost exa...")
    xgb_model_exa = my_xgb_model_exa

    start_time = time.time()

    if task == "ranking":
        xgb_model_exa.fit(
            X_train,
            y_train,
            group=group_tr,
            eval_set=[(X_val, y_val)],
            eval_group=[group_vl],
            verbose=nbr/10
        )
    else:
        xgb_model_exa.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=nbr/10
        )
    xgb_time = (time.time() - start_time) /xgb_model_exa.best_iteration
    y_pred_xgb = xgb_model_exa.predict_proba(X_test, iteration_range=(0, xgb_model_exa.best_iteration))[:, 1] if task == "classification" else xgb_model_exa.predict(X_test, iteration_range=(0, xgb_model_exa.best_iteration))
    xgb_score = roc_auc_score(y_test, y_pred_xgb) if task == "classification" else ndcg_by_group(y_test, y_pred_xgb, group_ts, k=10)
    print(f"XGBoost exa-> AUC/NDCG: {xgb_score:.4f} | Time: {xgb_time:.2f}s")

    # ===== LIGHTGBM BASELINE (NO GOSS, NO EFB) =====
    print("\nTraining LightGBM Baseline...")
    baseline_params = base_params.copy()
    baseline_params.update({"extra_trees": False})  # Disabilitiamo ottimizzazioni speciali

    start_time = time.time()
    model_baseline = lgb.train(baseline_params, train_data, valid_sets=[validation_data], num_boost_round=nbr, callbacks=[lgb.early_stopping(stopping_rounds=int(nbr * 0.05)),])
    baseline_time = (time.time() - start_time)/model_baseline.best_iteration
    y_pred_baseline = model_baseline.predict(X_test, num_iteration=model_baseline.best_iteration)
    baseline_score = roc_auc_score(y_test, y_pred_baseline) if task == "classification" else ndcg_by_group(y_test, y_pred_baseline, group_ts, k=10)
    print(f"LightGBM Baseline -> AUC/NDCG: {baseline_score:.4f} | Time: {baseline_time:.2f}s")

    # ===== LIGHTGBM SOLO EFB =====
    print("\nTraining LightGBM con solo EFB...")
    efb_params = baseline_params.copy()
    efb_params.update({"enable_bundle": True})  # Abilitiamo Exclusive Feature Bundling

    start_time = time.time()
    model_efb = lgb.train(efb_params, train_data, valid_sets=[validation_data], num_boost_round=nbr, callbacks=[lgb.early_stopping(stopping_rounds=int(nbr * 0.05)),])
    efb_time = (time.time() - start_time)/model_efb.best_iteration
    y_pred_efb = model_efb.predict(X_test, num_iteration=model_efb.best_iteration)
    efb_score = roc_auc_score(y_test, y_pred_efb) if task == "classification" else ndcg_by_group(y_test, y_pred_efb, group_ts, k=10)
    print(f"LightGBM + EFB -> AUC/NDCG: {efb_score:.4f} | Time: {efb_time:.2f}s")

    # ===== LIGHTGBM COMPLETO (GOSS + EFB) =====
    print("\nTraining LightGBM Completo (GOSS + EFB)...")
    full_params = efb_params.copy()
    full_params.update({"top_rate": a, "other_rate": b})  # Parametri GOSS dal paper

    start_time = time.time()
    model_full = lgb.train(full_params, train_data, valid_sets=[validation_data], num_boost_round=nbr, callbacks=[lgb.early_stopping(stopping_rounds=int(nbr * 0.05)),])
    full_time = (time.time() - start_time)/model_full.best_iteration
    y_pred_full = model_full.predict(X_test, num_iteration=model_full.best_iteration)
    full_score = roc_auc_score(y_test, y_pred_full) if task == "classification" else ndcg_by_group(y_test, y_pred_full, group_ts, k=10)
    print(f"LightGBM Completo -> AUC/NDCG: {full_score:.4f} | Time: {full_time:.2f}s")


#Wiring per i vari dataset 

current_dir = os.path.dirname(os.path.abspath(__file__))

'''
#wiring per LETOR dataset
print(f"LETOR DATASET")
path = os.path.join(current_dir, "MSLR-WEB30K")

X_train, y_train, group_train, X_val, y_val, group_val, X_test, y_test, group_test = load_mslr_dataset(path)

task = "ranking"

base_params = {
    "boosting_type": "gbdt",
    "objective": "binary" if task == "classification" else "lambdarank",
    "metric": "auc" if task == "classification" else "ndcg",

    "learning_rate": 0.05,
    "min_child_weight": 100,

    "num_leaves": 255,
    
}

xgb_model_hist = xgb.XGBRanker(
    objective="binary:logistic" if task == "classification" else "rank:pairwise",
    eval_metric="auc" if task == "classification" else "ndcg",
    tree_method="hist",
    n_estimators=1000,
    early_stopping_rounds=50,

    learning_rate=0.05,
    min_child_weight=100,

    grow_policy='lossguide',
    max_depth=0,
    max_leaves=255,
)

xgb_model_exa = xgb.XGBRanker(
    objective="binary:logistic" if task == "classification" else "rank:pairwise",
    eval_metric="auc" if task == "classification" else "ndcg",
    tree_method="hist",
    n_estimators=1000,
    early_stopping_rounds=50,

    learning_rate=0.05,
    min_child_weight=100,

    max_depth=16
)

confront_function(X_train, X_val, X_test, y_train, y_val, y_test, task, base_params, xgb_model_hist, xgb_model_exa, 0.1, 0.1, 1000, group_train, group_val, group_test)


#wiring per kdd 2010 dataset
print(f"KDD2010 DATASET")
path = os.path.join(current_dir, "Kdd 2010")

X_train, y_train, X_val, y_val, X_test, y_test = load_kdd2010_dataset(path)

task = "classification"

base_params = {
    "boosting_type": "gbdt",
    "objective": "binary" if task == "classification" else "lambdarank",
    "metric": "auc" if task == "classification" else "ndcg",

    "learning_rate": 0.1,
    "min_child_weight": 3000,

    "num_leaves": 255,
    "min_child_samples":1000
    
}

xgb_model_hist = xgb.XGBClassifier(
    objective="binary:logistic" if task == "classification" else "rank:pairwise",
    eval_metric="auc" if task == "classification" else "ndcg",
    tree_method="hist",
    n_estimators=100,
    early_stopping_rounds=5,

    learning_rate=0.1,
    min_child_weight=3000,

    grow_policy='lossguide',
    max_depth=0,
    max_leaves=255,
    #min_child_data=1000 parametro specificato nella documentazione ma non presente su XGBoost
)

xgb_model_exa = xgb.XGBClassifier(
    objective="binary:logistic" if task == "classification" else "rank:pairwise",
    eval_metric="auc" if task == "classification" else "ndcg",
    tree_method="hist",
    n_estimators=100,
    early_stopping_rounds=5,

    learning_rate=0.1,
    min_child_weight=3000,

    max_depth=50, 
    min_split_loss=150
)

confront_function(X_train, X_val, X_test, y_train, y_val, y_test, task, base_params, xgb_model_hist, xgb_model_exa, 0.05, 0.05, 100, True)

'''

#wiring per flight delay dataset
print(f"FLIGHT DELAY DATASET")
path = os.path.join(current_dir, "Flight delay dataset")

X_train, y_train, X_val, y_val, X_test, y_test = load_flight_delay_dataset(path)

task = "classification"

base_params = {
    "boosting_type": "gbdt",
    "objective": "binary" if task == "classification" else "lambdarank",
    "metric": "auc" if task == "classification" else "ndcg",

    "learning_rate": 0.1,
    "min_child_weight": 100,

    "num_leaves": 255,
    
}

xgb_model_hist = xgb.XGBClassifier(
    objective="binary:logistic" if task == "classification" else "rank:pairwise",
    eval_metric="auc" if task == "classification" else "ndcg",
    tree_method="hist",
    n_estimators=1000,
    early_stopping_rounds=50,

    learning_rate=0.1,
    min_child_weight=100,

    grow_policy='lossguide',
    max_depth=0,
    max_leaves=255
)

xgb_model_exa = xgb.XGBClassifier(
    objective="binary:logistic" if task == "classification" else "rank:pairwise",
    eval_metric="auc" if task == "classification" else "ndcg",
    tree_method="hist",
    n_estimators=1000,
    early_stopping_rounds=50,

    learning_rate=0.1,
    min_child_weight=100,

    max_depth=12, 
    min_split_loss=60
)

confront_function(X_train, X_val, X_test, y_train, y_val, y_test, task, base_params, xgb_model_hist, xgb_model_exa, 0.1, 0.1, 1000)