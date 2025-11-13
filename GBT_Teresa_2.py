import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG (edit these)
# ---------------------------
output_dir = 'ML-TERESA-RAWAN-VERO/1107_tests/timestamps/models'
train_sensor_dir = "ML-TERESA-RAWAN-VERO/sensor_train"  # contains *_CapacitanceTable.csv
train_mocap_dir = "ML-TERESA-RAWAN-VERO/mocap_train"    # contains *_resamp.csv
test_sensor_dir = "ML-TERESA-RAWAN-VERO/sensor_test"
test_mocap_dir = "ML-TERESA-RAWAN-VERO/mocap_test"

# If True, per-pair X/y row counts must match exactly (recommended for accuracy).
STRICT_ROW_MATCH = False

# Outlier clipping in feature/target space (robust to outliers)
HANDLE_OUTLIERS = True
LOWER_Q, UPPER_Q = 0.01, 0.99

# XGBoost params
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 8,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'alpha': 0.1,
    'lambda': 1.0,
    'random_state': 42,
    'tree_method': 'hist',
    'eval_metric': 'rmse',
}
NUM_ROUNDS = 5000
EARLY_STOP = 200

# ---------------------------
# Helpers
# ---------------------------

class ProgressBarCallback(xgb.callback.TrainingCallback):
    def __init__(self, total_rounds):
        self.total_rounds = total_rounds
        self.pbar = tqdm(total=total_rounds, desc="Training Progress")

    def after_iteration(self, model, epoch, evals_log):
        try:
            if epoch % 200 == 0 and "eval" in evals_log and "rmse" in evals_log["eval"]:
                eval_error = evals_log["eval"]["rmse"][-1]
                print(f"Iter {epoch}, Eval RMSE: {eval_error}")
        except Exception:
            pass
        self.pbar.update(1)
        if hasattr(model, 'best_iteration') and model.best_iteration == epoch:
            self.pbar.close()
            return True
        if epoch + 1 == self.total_rounds:
            self.pbar.close()
        return False

def clean_value(v):
    s = str(v).strip()
    parts = s.split('.')
    if len(parts) > 2:
        s = '.'.join(parts[:-1])
    return s

# _TS_NAME_RE = re.compile(r'(?:^|_)(time|timestamp|time_ms|datetime|date|ms|frame)(?:_|$)', re.IGNORECASE)

def _detect_header(file_path, sample_rows=5):
    # autodetect delimiter and header presence
    peek = pd.read_csv(file_path, nrows=sample_rows, header=None, dtype=str, sep=None, engine='python')
    first_row = peek.iloc[0].astype(str)
    if first_row.apply(lambda s: bool(re.search(r'[A-Za-z]', s))).any():
        return 0
    return None

def _drop_timestamp_like_columns(df):
    cols_to_drop = []
    for c in df.columns:
        name = str(c)
        if _TS_NAME_RE.search(name.strip()):
            cols_to_drop.append(c)
    for c in df.columns:
        if c in cols_to_drop:
            continue
        col = df[c].astype(str)
        digit_ratio = np.mean(col.str.fullmatch(r'-?\d{10,}').fillna(False))
        iso_ratio = np.mean(col.str.contains(r'\d{4}-\d{2}-\d{2}', na=False))
        if digit_ratio > 0.9 or iso_ratio > 0.6:
            cols_to_drop.append(c)
    if cols_to_drop:
        # print(f"Dropping timestamp-like columns: {list(cols_to_drop)}")
        df = df.drop(columns=cols_to_drop)
    return df

def add_lags(X, lags=10):
    # X: [T, F] -> [T, F*(lags+1)]
    T, F = X.shape
    out = [X]
    pad = np.repeat(X[:1], lags, axis=0)
    Xp = np.vstack([pad, X])  # prefix padding
    for k in range(1, lags+1):
        out.append(Xp[lags-k: lags-k+T])
    return np.hstack(out)

LAGS = 10

def _winsorize_df(df, lower_q=0.01, upper_q=0.99):
    q_low = df.quantile(lower_q, axis=0, numeric_only=True)
    q_high = df.quantile(upper_q, axis=0, numeric_only=True)
    return df.clip(lower=q_low, upper=q_high, axis=1)

def load_and_clean_csv(file_path, handle_outliers=True, lower_q=0.01, upper_q=0.99, drop_first_n_cols=0):
    header = _detect_header(file_path)
    df = pd.read_csv(file_path, header=header, dtype=str, sep=None, engine='python')

    if drop_first_n_cols > 0:
        df = df.iloc[:, drop_first_n_cols:]

    df = df.map(clean_value)
    # df = _drop_timestamp_like_columns(df)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    empty_cols = [c for c in df.columns if df[c].notna().sum() == 0]
    if empty_cols:
        print(f"Dropping empty columns: {empty_cols}")
        df = df.drop(columns=empty_cols)

    before = len(df)
    df = df.dropna(how='all')
    if len(df) != before:
        print(f"Dropped {before - len(df)} completely empty rows.")

    medians = df.median(numeric_only=True)
    df = df.fillna(medians)

    if handle_outliers and not df.empty:
        df = _winsorize_df(df, lower_q=lower_q, upper_q=upper_q)

    return df.to_numpy(dtype=float)

# ---------------------------
# Directory loaders & strict pairing
# ---------------------------
SENSOR_SUFFIX = "_CapacitanceTable"
MOCAP_SUFFIX = "_resamp"
CSV_EXTS = {".csv", ".CSV", ".tsv", ".TSV", ".txt", ".TXT"}

def _basename_without_suffix(p: Path, suffix: str):
    stem = p.stem
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem

def _collect_files(dir_path: str):
    d = Path(dir_path)
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(f"Directory not found or not a directory: {dir_path}")
    files = [p for p in d.iterdir() if p.is_file() and p.suffix in CSV_EXTS]
    if not files:
        raise RuntimeError(f"No data files (csv/tsv/txt) found in: {dir_path}")
    return files

def _build_keymap(files, suffix_expected):
    m = {}
    for f in files:
        key = _basename_without_suffix(f, suffix_expected)
        m[key] = f
    return m

def match_pairs_strict(sensor_dir: str, mocap_dir: str):
    sensor_files = _collect_files(sensor_dir)
    mocap_files = _collect_files(mocap_dir)

    sensor_files = [p for p in sensor_files if p.stem.endswith(SENSOR_SUFFIX)]
    mocap_files = [p for p in mocap_files if p.stem.endswith(MOCAP_SUFFIX)]

    if not sensor_files:
        raise RuntimeError(f"No sensor files ending with '{SENSOR_SUFFIX}' found in {sensor_dir}")
    if not mocap_files:
        raise RuntimeError(f"No mocap files ending with '{MOCAP_SUFFIX}' found in {mocap_dir}")

    smap = _build_keymap(sensor_files, SENSOR_SUFFIX)
    mmap = _build_keymap(mocap_files, MOCAP_SUFFIX)

    s_keys = set(smap.keys())
    m_keys = set(mmap.keys())

    only_in_sensor = sorted(s_keys - m_keys)
    only_in_mocap = sorted(m_keys - s_keys)
    if only_in_sensor or only_in_mocap:
        msg = []
        if only_in_sensor:
            msg.append(f"Unmatched sensor keys: {only_in_sensor[:8]}{' ...' if len(only_in_sensor)>8 else ''}")
        if only_in_mocap:
            msg.append(f"Unmatched mocap keys: {only_in_mocap[:8]}{' ...' if len(only_in_mocap)>8 else ''}")
        raise RuntimeError("Sensor/Mocap filename mismatch within split. " + " | ".join(msg))

    keys_sorted = sorted(s_keys)
    return [(smap[k], mmap[k], k) for k in keys_sorted]

def align_X_y(X, y, key, strict_row_match=True):
    """
    Align X and y arrays by trimming the longer one to match the shorter if not strict.
    If strict, raise error on mismatch.
    """
    if X.shape[0] != y.shape[0]:
        if strict_row_match:
            raise RuntimeError(f"[{key}] Row mismatch: X={X.shape[0]} vs y={y.shape[0]} (strict mode).")
        else:
            mn = min(X.shape[0], y.shape[0])
            print(f"[WARN] [{key}] Row mismatch -> trimming both to {mn}")
            X = X[:mn]
            y = y[:mn]
    return X, y

def load_dataset_from_dirs(sensor_dir: str, mocap_dir: str,
                           handle_outliers=True, lower_q=0.01, upper_q=0.99,
                           strict_row_match=True):
    pairs = match_pairs_strict(sensor_dir, mocap_dir)
    X_list, y_list, keys = [], [], []
    print(f"[INFO] {len(pairs)} matched pairs found in\n  Sensor: {sensor_dir}\n  Mocap : {mocap_dir}")

    for s_path, m_path, key in pairs:
        X = load_and_clean_csv(str(s_path), handle_outliers=handle_outliers, lower_q=lower_q, upper_q=upper_q, drop_first_n_cols=0)
        y = load_and_clean_csv(str(m_path), handle_outliers=handle_outliers, lower_q=lower_q, upper_q=upper_q, drop_first_n_cols=4)

        # Align X and y first to ensure same row count before masking
        X, y = align_X_y(X, y, key, strict_row_match=strict_row_match)

        # Safety on NaNs in y (should be none after cleaning)
        mask = np.all(~np.isnan(y), axis=1)
        if mask.sum() != y.shape[0]:
            dropped = int(y.shape[0] - mask.sum())
            print(f"[{key}] Dropped {dropped} rows with NaN in y.")
        X = X[mask]
        y = y[mask]

        X_list.append(X)
        y_list.append(y)
        keys.append(key)

    X_all = np.vstack(X_list)
    y_all = np.vstack(y_list)
    return X_all, y_all, keys

# ---------------------------
# Training / Evaluation
# ---------------------------
def choose_device_param():
    device = 'cuda'
    try:
        _ = xgb.Booster(params={'device': device})
    except Exception:
        device = 'cpu'
        print("[INFO] CUDA not available; using CPU.")
    return device

def train_and_evaluate():
    os.makedirs(output_dir, exist_ok=True)

    # Load train/test (strict pairing)
    X_train, y_train_full, train_keys = load_dataset_from_dirs(
        train_sensor_dir, train_mocap_dir,
        handle_outliers=HANDLE_OUTLIERS, lower_q=LOWER_Q, upper_q=UPPER_Q,
        strict_row_match=STRICT_ROW_MATCH
    )
    X_test, y_test_full, test_keys = load_dataset_from_dirs(
        test_sensor_dir, test_mocap_dir,
        handle_outliers=HANDLE_OUTLIERS, lower_q=LOWER_Q, upper_q=UPPER_Q,
        strict_row_match=STRICT_ROW_MATCH
    )

    print(f"Shapes -> X_train: {X_train.shape}, y_train: {y_train_full.shape}, "
          f"X_test: {X_test.shape}, y_test: {y_test_full.shape}")
    
    # # ADD LAG TEST
    # X_train = add_lags(X_train, lags=LAGS)
    # X_test  = add_lags(X_test,  lags=LAGS)

    # Scale
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train_full)
    y_test_scaled = scaler_y.transform(y_test_full)

    joblib.dump(scaler_X, os.path.join(output_dir, 'scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(output_dir, 'scaler_y.pkl'))

    # add the unscaled scalers?

    # print(f"[Saved] scalers to {output_dir}/scaler_X.pkl and scaler_y.pkl")

    # Print variance stats
    print("Train y var (scaled):", np.var(y_train_scaled))
    print("Test y var (scaled):", np.var(y_test_scaled))
    print("Train y var (unscaled):", np.var(y_train_full))
    print("Test y var (unscaled):", np.var(y_test_full))

    # Train per-output dimension
    device_param = choose_device_param()
    params = dict(XGB_PARAMS)
    params['device'] = device_param

    train_matrix = xgb.DMatrix(X_train_scaled)
    test_matrix = xgb.DMatrix(X_test_scaled)

    y_train_pred_list, y_test_pred_list = [], []
    individual_train_r2_scaled, individual_test_r2_scaled = [], []
    models = []

    for i in range(y_train_full.shape[1]):
        y_tr = y_train_scaled[:, i]
        y_te = y_test_scaled[:, i]
        train_matrix.set_label(y_tr)
        test_matrix.set_label(y_te)

        model = xgb.train(
            params,
            train_matrix,
            num_boost_round=NUM_ROUNDS,
            evals=[(test_matrix, "eval")],
            early_stopping_rounds=EARLY_STOP,
            verbose_eval=False,
            # callbacks=[ProgressBarCallback(NUM_ROUNDS)],
        )

        model_path = os.path.join(output_dir, f'xgboost_model_dim_{i+1}.json')
        model.save_model(model_path)
        models.append(model)
        print(f"[Saved] model for dim {i+1} -> {model_path}")

        y_tr_pred = model.predict(train_matrix)
        y_te_pred = model.predict(test_matrix)
        y_train_pred_list.append(y_tr_pred)
        y_test_pred_list.append(y_te_pred)

        tr_r2 = r2_score(y_tr, y_tr_pred)
        te_r2 = r2_score(y_te, y_te_pred)
        individual_train_r2_scaled.append(tr_r2)
        individual_test_r2_scaled.append(te_r2)
        print(f"Dim {i+1:02d}  R2 train (scaled): {tr_r2:.4f} | R2 test (scaled): {te_r2:.4f}")

    y_train_pred_scaled = np.column_stack(y_train_pred_list)
    y_test_pred_scaled = np.column_stack(y_test_pred_list)
    overall_train_r2_scaled = r2_score(y_train_scaled, y_train_pred_scaled, multioutput='variance_weighted')
    overall_test_r2_scaled = r2_score(y_test_scaled, y_test_pred_scaled, multioutput='variance_weighted')

    # Inverse transform for unscaled evaluation
    y_train_pred_unscaled = scaler_y.inverse_transform(y_train_pred_scaled)
    y_test_pred_unscaled = scaler_y.inverse_transform(y_test_pred_scaled)

    # R2 on unscaled
    overall_train_r2_unscaled = r2_score(y_train_full, y_train_pred_unscaled, multioutput='variance_weighted')
    overall_test_r2_unscaled = r2_score(y_test_full, y_test_pred_unscaled, multioutput='variance_weighted')

    individual_train_r2_unscaled = [r2_score(y_train_full[:, i], y_train_pred_unscaled[:, i]) for i in range(y_train_full.shape[1])]
    individual_test_r2_unscaled = [r2_score(y_test_full[:, i], y_test_pred_unscaled[:, i]) for i in range(y_test_full.shape[1])]

    print("\nScaled Individual Training R2 per dim:", individual_train_r2_scaled)
    print("Scaled Individual Testing R2 per dim:", individual_test_r2_scaled)
    print("Scaled Overall Training R2:", overall_train_r2_scaled)
    print("Scaled Overall Testing R2:", overall_test_r2_scaled)

    print("\nUnscaled Individual Training R2 per dim:", individual_train_r2_unscaled)
    print("Unscaled Individual Testing R2 per dim:", individual_test_r2_unscaled)
    print("Unscaled Overall Training R2:", overall_train_r2_unscaled)
    print("Unscaled Overall Testing R2:", overall_test_r2_unscaled)

    # Save metadata for reproducibility (use unscaled R2)
    meta = {
        "train_sensor_dir": train_sensor_dir,
        "train_mocap_dir": train_mocap_dir,
        "test_sensor_dir": test_sensor_dir,
        "test_mocap_dir": test_mocap_dir,
        "train_keys": train_keys,
        "test_keys": test_keys,
        "X_shape_train": list(X_train.shape),
        "y_shape_train": list(y_train_full.shape),
        "X_shape_test": list(X_test.shape),
        "y_shape_test": list(y_test_full.shape),
        "strict_row_match": STRICT_ROW_MATCH,
        "handle_outliers": HANDLE_OUTLIERS,
        "lower_q": LOWER_Q,
        "upper_q": UPPER_Q,
        "xgb_params": params,
        "num_rounds": NUM_ROUNDS,
        "early_stop": EARLY_STOP,
        "overall_train_r2": overall_train_r2_unscaled,
        "overall_test_r2": overall_test_r2_unscaled,
        "individual_train_r2": individual_train_r2_unscaled,
        "individual_test_r2": individual_test_r2_unscaled,
        "note": "R2 computed on unscaled data for fair variance comparison"
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Saved] metadata -> {os.path.join(output_dir, 'run_metadata.json')}")

    # ---------------------------
    # Plotting Section
    # ---------------------------
    plots_dir = os.path.join(output_dir, 'test_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    num_dims = y_test_full.shape[1]
    print(f"[INFO] Plotting {num_dims} dimensions.")
    
    cols = 4
    rows = (num_dims + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows), sharex=True)
    if rows == 1:
        axes = axes.ravel()
    else:
        axes = axes.ravel()
    
    for dim in range(num_dims):
        ax = axes[dim]
        time_steps = np.arange(len(y_test_full))
        ax.plot(time_steps, y_test_full[:, dim], label='True', alpha=0.7, color='blue')
        ax.plot(time_steps, y_test_pred_unscaled[:, dim], label='Predicted', alpha=0.7, color='red')
        r2_val = r2_score(y_test_full[:, dim], y_test_pred_unscaled[:, dim])
        ax.set_title(f'Dimension {dim+1} (R² = {r2_val:.3f})')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    for idx in range(num_dims, len(axes)):
        axes[idx].set_visible(False)
    
    plt.xlabel('Time Step')
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'all_dims_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] Combined plot: {plot_path}")

    return {
        "overall_train_r2": overall_train_r2_unscaled,
        "overall_test_r2": overall_test_r2_unscaled,
        "individual_train_r2": individual_train_r2_unscaled,
        "individual_test_r2": individual_test_r2_unscaled,
    }

# ---------------------------
# Inference (file or directory)
# ---------------------------
def predict_new_file(sensor_file_path, models_dir, lower_q=LOWER_Q, upper_q=UPPER_Q):
    scaler_X = joblib.load(os.path.join(models_dir, 'scaler_X.pkl'))
    scaler_y = joblib.load(os.path.join(models_dir, 'scaler_y.pkl'))

    X_new = load_and_clean_csv(sensor_file_path, handle_outliers=True, lower_q=lower_q, upper_q=upper_q, drop_first_n_cols=0)
    if np.isnan(X_new).any():
        col_medians = np.nanmedian(X_new, axis=0)
        inds = np.where(np.isnan(X_new))
        X_new[inds] = np.take(col_medians, inds[1])

    X_new_scaled = scaler_X.transform(X_new)
    X_new_matrix = xgb.DMatrix(X_new_scaled)

    # Infer number of dimensions from saved models
    model_files = sorted(
        f for f in os.listdir(models_dir) if re.fullmatch(r'xgboost_model_dim_\d+\.json', f)
    )
    if not model_files:
        raise RuntimeError("No saved models found.")
    predictions_scaled = []
    for mf in model_files:
        booster = xgb.Booster()
        booster.load_model(os.path.join(models_dir, mf))
        pred = booster.predict(X_new_matrix)
        predictions_scaled.append(pred)
    predictions_scaled = np.column_stack(predictions_scaled)
    return scaler_y.inverse_transform(predictions_scaled)

def predict_new_directory(sensor_dir, models_dir, lower_q=LOWER_Q, upper_q=UPPER_Q):
    d = Path(sensor_dir)
    files = [p for p in d.iterdir() if p.is_file() and p.suffix in CSV_EXTS and p.stem.endswith(SENSOR_SUFFIX)]
    if not files:
        raise RuntimeError(f"No sensor files ending with '{SENSOR_SUFFIX}' found in {sensor_dir}")
    results = {}
    for p in sorted(files, key=lambda x: _basename_without_suffix(x, SENSOR_SUFFIX)):
        key = _basename_without_suffix(p, SENSOR_SUFFIX)
        results[key] = predict_new_file(str(p), models_dir, lower_q=lower_q, upper_q=upper_q)
    return results

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    train_and_evaluate()