"""
test.py
Feature testing engine.
Builds on the volare LightGBM pipeline to evaluate whether a candidate feature
improves on the current active feature set against the rolling vol baseline.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from lightgbm import LGBMRegressor, early_stopping, log_evaluation, record_evaluation

from src.agent.schemas import TestResults
from src.agent.tool import ToolResult

# ---------------------------------------------------------------------------
# Safe code execution
# ---------------------------------------------------------------------------

def validate_code(code: str) -> None:
    """Basic safety check on LLM-generated code."""
    forbidden = [
        "import os", "import sys", "import subprocess",
        "import shutil", "import socket", "import requests",
        "open(", "exec(", "eval(", "__import__",
        "globals()", "locals()", "compile(",
        "os.path", "os.system", "pathlib"
    ]
    for pattern in forbidden:
        if pattern in code:
            raise ValueError(f"Forbidden pattern in generated code: '{pattern}'")


def execute_feature_code(code: str, df: pd.DataFrame) -> pd.Series:
    """Execute LLM-generated feature code in a restricted namespace."""
    validate_code(code)

    namespace = {"numpy": np, "pandas": pd, "np": np, "pd": pd}
    exec(code, namespace)

    if "compute_feature" not in namespace:
        raise ValueError("Generated code does not define a 'compute_feature' function")

    result = namespace["compute_feature"](df)

    if not isinstance(result, pd.Series):
        raise ValueError(f"compute_feature must return a pandas Series, got {type(result)}")

    if len(result) != len(df):
        raise ValueError(
            f"compute_feature returned Series of length {len(result)}, expected {len(df)}"
        )

    return result


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    active_features: List[dict],
    candidate_feature: dict,
    horizon_seconds: int,
    eps: float = 1e-8
) -> pd.DataFrame:
    """
    Constructs the full feature matrix for LightGBM training.

    Builds:
    - Baseline rolling vol features (multi-window, matching volare pipeline)
    - Active promoted features (from code in active_features)
    - Candidate feature (the new one being tested)
    - Target: rolling_log_future_vol

    Returns a DataFrame with all features and target, NaNs dropped.
    """
    df = df.copy()

    time_res = (df.index[1] - df.index[0]).total_seconds()
    H = int(horizon_seconds / time_res)

    # Log returns
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # --- Baseline: naive 1-hour rolling vol only ---
    baseline_window = max(int(horizon_seconds / time_res), 2)
    baseline_col = f"rolling_vol_{baseline_window}"
    df[baseline_col] = df["log_return"].rolling(baseline_window).std()

    # --- Active promoted features ---
    for feat in active_features:
        try:
            series = execute_feature_code(feat["code"], df)
            nan_pct = series.isna().mean() * 100
            if nan_pct > 10.0:
                print(f"Warning: active feature '{feat['name']}' has {nan_pct:.1f}% NaN — skipping")
            else:
                df[feat["name"]] = series
        except Exception as e:
            print(f"Warning: failed to compute active feature '{feat['name']}': {e}")

    # --- Candidate feature ---
    candidate_series = execute_feature_code(candidate_feature["code"], df)
    nan_pct = candidate_series.isna().mean() * 100
    if nan_pct > 10.0:
        raise ValueError(
            f"Candidate feature '{candidate_feature['name']}' produces {nan_pct:.1f}% NaN "
            f"(limit: 10%). Likely cause: rolling window without min_periods=1, window too large, "
            f"or chained rolling operations. Check the generated code."
        )
    candidate_col = candidate_feature["name"]
    df[candidate_col] = candidate_series

    # --- Target: log future realised vol ---
    r = df["log_return"].values
    r2 = np.where(np.isnan(r), 0.0, r ** 2)
    csum = np.cumsum(np.insert(r2, 0, 0))
    rms = np.sqrt((csum[H:] - csum[:-H]) / H)
    future_vol = np.full(len(df), np.nan)
    future_vol[:len(rms)] = rms
    df["rolling_future_vol"] = future_vol
    df["rolling_log_future_vol"] = np.log(df["rolling_future_vol"] + eps)

    return df


def get_feature_cols(df: pd.DataFrame, candidate_name: str) -> List[str]:
    """Return all feature column names — baseline + active + candidate."""
    exclude = {
        "open", "high", "low", "close", "symbol",
        "log_return", "rolling_future_vol", "rolling_log_future_vol"
    }
    return [c for c in df.columns if c not in exclude]


# ---------------------------------------------------------------------------
# Walk-forward validation using LightGBM
# ---------------------------------------------------------------------------

def walk_forward_validate_lgbm(
    df_features: pd.DataFrame,
    candidate_name: str,
    horizon_seconds: int,
    n_folds: int = 5,
    min_train_size: float = 0.5,
    eps: float = 1e-8
) -> ToolResult:
    """
    Walk-forward validation using LightGBM.

    Two models are compared per fold:
    - baseline_model: trained on baseline rolling vol features only
    - full_model: trained on baseline + active + candidate features

    Improvement is measured as RMSE reduction of full vs baseline,
    both evaluated against the rolling vol naive baseline prediction.
    """
    target_col = "rolling_log_future_vol"
    time_res = (df_features.index[1] - df_features.index[0]).total_seconds()
    baseline_window = max(int(horizon_seconds / time_res), 2)
    baseline_col = f"rolling_vol_{baseline_window}"

    all_feature_cols = get_feature_cols(df_features, candidate_name)
    baseline_feature_cols = [baseline_col]
    full_feature_cols = all_feature_cols

    # Deduplicate: baseline_col is already in full_feature_cols; adding it again creates a 2-column selection
    combined_cols = list(dict.fromkeys(full_feature_cols + [target_col, baseline_col]))
    combined = df_features[combined_cols].dropna()

    # --- NaN diagnostics ---
    try:
        _cols_to_check = full_feature_cols + [target_col, baseline_col]
        _total_rows = len(df_features)
        _nan_report = []
        for _col in _cols_to_check:
            if _col not in df_features.columns:
                _nan_report.append(f"  MISSING column: {_col}")
                continue
            _nan_count = int(df_features[_col].isna().sum())
            if _total_rows > 0:
                _nan_pct = _nan_count / _total_rows * 100
                if _nan_pct > 10.0:
                    _nan_report.append(
                        f"  HIGH NaN — {_col}: {_nan_count}/{_total_rows} rows ({_nan_pct:.1f}%)"
                    )
        if len(combined) < 1000 or _nan_report:
            print(
                f"[NaN Diagnostic] {candidate_name} — {_total_rows} total rows, "
                f"{len(combined)} survived dropna "
                f"({len(combined)/_total_rows*100:.1f}% retained)"
                if _total_rows > 0
                else f"[NaN Diagnostic] {candidate_name} — 0 total rows"
            )
            if _nan_report:
                print("  Columns with >10% NaN (likely root cause):")
                for _line in _nan_report:
                    print(_line)
            elif len(combined) < 1000:
                print("  No single column had >10% NaN — NaN pattern is from column interaction")
    except Exception:
        pass  # diagnostic must never mask the real error

    if len(combined) < 1000:
        return {"error": "Insufficient data after NaN removal"}

    n = len(combined)
    min_train = int(n * min_train_size)
    fold_size = (n - min_train) // n_folds

    fold_results = []

    lgbm_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "verbose": -1,
        "n_jobs": -1
    }

    for i in range(n_folds):
        train_end = min_train + i * fold_size
        test_start = train_end
        test_end = test_start + fold_size

        if test_end > n:
            break

        train = combined.iloc[:train_end]
        test = combined.iloc[test_start:test_end]

        # Split train into train/val (90/10)
        val_idx = int(len(train) * 0.9)
        tr, val = train.iloc[:val_idx], train.iloc[val_idx:]

        y_test = test[target_col].values
        baseline_pred = np.log(test[baseline_col].values + eps)

        rmse_naive = np.sqrt(np.mean((baseline_pred - y_test) ** 2))

        # --- Baseline model (no candidate feature) ---
        try:
            base_cols_available = [
                c for c in baseline_feature_cols if c in combined.columns
            ]
            m_base = LGBMRegressor(**lgbm_params)
            m_base.fit(
                tr[base_cols_available], tr[target_col],
                eval_set=[(val[base_cols_available], val[target_col])],
                eval_metric="rmse",
                callbacks=[early_stopping(50, verbose=False), log_evaluation(-1)]
            )
            pred_base = m_base.predict(test[base_cols_available])
            rmse_base_model = np.sqrt(np.mean((pred_base - y_test) ** 2))
        except Exception as e:
            rmse_base_model = rmse_naive
            print(f"Baseline model fold {i+1} failed: {e}")

        # --- Full model (with candidate feature) ---
        try:
            m_full = LGBMRegressor(**lgbm_params)
            m_full.fit(
                tr[full_feature_cols], tr[target_col],
                eval_set=[(val[full_feature_cols], val[target_col])],
                eval_metric="rmse",
                callbacks=[early_stopping(50, verbose=False), log_evaluation(-1)]
            )
            pred_full = m_full.predict(test[full_feature_cols])
            rmse_full_model = np.sqrt(np.mean((pred_full - y_test) ** 2))

            # Feature importance for candidate
            importances = dict(zip(full_feature_cols, m_full.feature_importances_))
            candidate_importance = importances.get(candidate_name, 0)
            total_importance = sum(importances.values()) + eps
            candidate_importance_pct = candidate_importance / total_importance * 100

        except Exception as e:
            rmse_full_model = rmse_base_model
            candidate_importance_pct = 0.0
            print(f"Full model fold {i+1} failed: {e}")

        # Improvement: full model vs baseline model
        improvement_vs_base = (
            (rmse_base_model - rmse_full_model) / rmse_base_model * 100
        )
        # Improvement: full model vs naive rolling vol
        improvement_vs_naive = (
            (rmse_naive - rmse_full_model) / rmse_naive * 100
        )

        fold_results.append({
            "fold": i + 1,
            "rmse_naive_baseline": round(float(rmse_naive), 6),
            "rmse_lgbm_without_candidate": round(float(rmse_base_model), 6),
            "rmse_lgbm_with_candidate": round(float(rmse_full_model), 6),
            "improvement_vs_lgbm_baseline_pct": round(float(improvement_vs_base), 3),
            "improvement_vs_naive_baseline_pct": round(float(improvement_vs_naive), 3),
            "candidate_importance_pct": round(float(candidate_importance_pct), 2)
        })

    if not fold_results:
        return {"error": "No folds completed"}

    improvements = [f["improvement_vs_lgbm_baseline_pct"] for f in fold_results]
    importances = [f["candidate_importance_pct"] for f in fold_results]

    monotonic_decay = all(
        improvements[i] >= improvements[i + 1]
        for i in range(len(improvements) - 1)
    )

    mean_imp = np.mean(importances)
    importance_drift_pct = (
        (max(importances) - min(importances)) / mean_imp * 100
        if mean_imp > 0 else 0.0
    )

    return {
        "folds": fold_results,
        "overall_improvement_vs_lgbm_pct": round(float(np.mean(improvements)), 3),
        "overall_improvement_vs_naive_pct": round(
            float(np.mean([f["improvement_vs_naive_baseline_pct"] for f in fold_results])), 3
        ),
        "mean_candidate_importance_pct": round(float(mean_imp), 2),
        "importance_drift_pct": round(float(importance_drift_pct), 1),
        "monotonic_decay": monotonic_decay,
        "n_folds_completed": len(fold_results)
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_feature_test(
    feature: dict,
    context: dict,
    data: Dict[str, pd.DataFrame],
    active_features: List[dict] = None
) -> ToolResult:
    """
    Main entry point. Tests candidate feature across all pairs using LightGBM.
    Compares full model (baseline features + active + candidate) vs
    baseline model (baseline features only).

    Returns a ToolResult with value: TestResults on success.
    """
    if active_features is None:
        active_features = []

    pairs = context["data"]["pairs"]
    horizon = context["data"]["horizon_seconds"]
    code = feature.get("code")

    if not code:
        return ToolResult(ok=False,
        error='[test] No code provided in feature')
    per_pair_results = {}
    errors = []

    for pair in pairs:
        if pair not in data:
            errors.append(f"No data found for {pair}")
            continue

        df = data[pair].copy()

        # Ensure timestamp is index
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")

        try:
            df_features = build_feature_matrix(
                df, active_features, feature, horizon
            )
            results = walk_forward_validate_lgbm(
                df_features, feature["name"], horizon
            )
            per_pair_results[pair] = results
        except ValueError as e:
            errors.append(f"{pair} — validation error: {str(e)}")
        except Exception as e:
            errors.append(f"{pair} — error: {str(e)}")

    lgbm_improvements = [
        per_pair_results[p]["overall_improvement_vs_lgbm_pct"]
        for p in per_pair_results
        if "overall_improvement_vs_lgbm_pct" in per_pair_results.get(p, {})
    ]
    naive_improvements = [
        per_pair_results[p]["overall_improvement_vs_naive_pct"]
        for p in per_pair_results
        if "overall_improvement_vs_naive_pct" in per_pair_results.get(p, {})
    ]
    drifts = [
        per_pair_results[p]["importance_drift_pct"]
        for p in per_pair_results
        if "importance_drift_pct" in per_pair_results.get(p, {})
    ]
    importances = [
        per_pair_results[p]["mean_candidate_importance_pct"]
        for p in per_pair_results
        if "mean_candidate_importance_pct" in per_pair_results.get(p, {})
    ]

    def safe_mean(values):
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else None

    try:
        lgbm_mean = safe_mean(lgbm_improvements)
        naive_mean = safe_mean(naive_improvements)
        importance_mean = safe_mean(importances)
        drift_mean = safe_mean(drifts)
        results = {
            "per_pair": per_pair_results,
            "aggregate": {
                "mean_improvement_vs_lgbm_pct": round(lgbm_mean, 3) if lgbm_mean else None,
                "mean_improvement_vs_naive_pct": round(naive_mean, 3) if naive_mean else None,
                "mean_candidate_importance_pct": round(importance_mean, 2) if importance_mean else None,
                "mean_importance_drift_pct": round(drift_mean, 1) if drift_mean else None,
                "any_monotonic_decay": any(
                    per_pair_results[p].get("monotonic_decay", False)
                    for p in per_pair_results
                ),
                "pairs_tested": list(per_pair_results.keys()),
                "errors": errors
            }
        }

        validated = TestResults.model_validate(results)

        return ToolResult(ok=True, value=validated)

    except Exception as e:
        return ToolResult(ok=False, error=f"[test] {type(e).__name__}: {str(e)}")