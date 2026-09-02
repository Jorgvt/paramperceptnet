"""Bootstrap evaluation and statistical significance tools for IQA models.

This module provides tools to compute point estimates and bootstrap confidence intervals
for standard Image Quality Assessment (IQA) performance metrics (PLCC, SROCC, KROCC,
RMSE, R²), as well as paired hypothesis tests for comparing generalization gaps across models.
"""

from typing import Dict, Optional, Tuple, Union, Any
import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
import pandas as pd


def logistic_4param(
    x: np.ndarray,
    beta1: float,
    beta2: float,
    beta3: float,
    beta4: float,
) -> np.ndarray:
    """Standard 4-parameter monotonic logistic function for IQA score alignment (ITU-T / VQEG).

    f(x) = (beta1 - beta2) / (1 + exp(-(x - beta3) / |beta4|)) + beta2
    """
    beta4_abs = np.maximum(np.abs(beta4), 1e-7)
    # Prevent overflow in exp
    z = np.clip(-(x - beta3) / beta4_abs, -100.0, 100.0)
    return (beta1 - beta2) / (1.0 + np.exp(z)) + beta2


def fit_logistic_mapping(
    dists: np.ndarray,
    mos: np.ndarray,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Fit a monotonic 4-parameter logistic mapping from predicted distances to subjective MOS.

    Parameters
    ----------
    dists : np.ndarray
        Model predicted perceptual distances.
    mos : np.ndarray
        Ground-truth subjective scores (MOS/DMOS).

    Returns
    -------
    mapped_dists : np.ndarray
        Distances transformed into the MOS scale via the fitted non-linear logistic function.
    params : tuple
        Fitted parameters (beta1, beta2, beta3, beta4).
    """
    dists = np.asarray(dists, dtype=np.float64).ravel()
    mos = np.asarray(mos, dtype=np.float64).ravel()

    # Initial parameter heuristics
    b1_init = float(np.max(mos))
    b2_init = float(np.min(mos))
    b3_init = float(np.median(dists))
    b4_init = float(np.std(dists) if np.std(dists) > 1e-7 else 1.0)
    p0 = [b1_init, b2_init, b3_init, b4_init]

    try:
        popt, _ = curve_fit(
            logistic_4param,
            dists,
            mos,
            p0=p0,
            maxfev=5000,
        )
        mapped = logistic_4param(dists, *popt)
        return mapped, tuple(popt)
    except Exception:
        # Fallback to linear scaling if non-linear optimization fails
        slope, intercept, _, _, _ = stats.linregress(dists, mos)
        mapped = slope * dists + intercept
        return mapped, (float(intercept), float(slope), 0.0, 1.0)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson Linear Correlation Coefficient (PLCC)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    r, _ = stats.pearsonr(x, y)
    return float(r)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman Rank Order Correlation Coefficient (SROCC)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    rho, _ = stats.spearmanr(x, y)
    return float(rho)


def kendall_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Kendall Rank Order Correlation Coefficient (KROCC)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    tau, _ = stats.kendalltau(x, y)
    return float(tau)


def rmse_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fit_logistic: bool = True,
) -> float:
    """Compute Root Mean Squared Error (RMSE) between targets and predictions.

    If fit_logistic is True, predictions are first mapped to the target MOS scale using
    the standard 4-parameter logistic mapping.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if fit_logistic:
        y_pred, _ = fit_logistic_mapping(y_pred, y_true)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fit_logistic: bool = True,
) -> float:
    """Compute Coefficient of Determination (R²) between targets and predictions.

    If fit_logistic is True, predictions are first non-linearly aligned to target MOS scale.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if fit_logistic:
        y_pred, _ = fit_logistic_mapping(y_pred, y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-9:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def compute_all_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    fit_logistic: bool = True,
) -> Dict[str, float]:
    """Compute standard full suite of IQA evaluation metrics.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted perceptual distances or metric scores.
    targets : np.ndarray
        Ground-truth subjective MOS/DMOS scores.
    fit_logistic : bool, default=True
        Whether to apply 4-parameter logistic mapping before computing RMSE and R².

    Returns
    -------
    dict
        Dictionary with keys: 'plcc', 'abs_plcc', 'srocc', 'abs_srocc', 'krocc', 'rmse', 'r2'.
    """
    p = pearson_corr(predictions, targets)
    s = spearman_corr(predictions, targets)
    k = kendall_corr(predictions, targets)
    rmse = rmse_score(targets, predictions, fit_logistic=fit_logistic)
    r2 = r2_metric(targets, predictions, fit_logistic=fit_logistic)

    return {
        "plcc": p,
        "abs_plcc": abs(p),
        "srocc": s,
        "abs_srocc": abs(s),
        "krocc": k,
        "abs_krocc": abs(k),
        "rmse": rmse,
        "r2": r2,
    }


def bootstrap_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    reference_ids: Optional[np.ndarray] = None,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    fit_logistic: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Compute point estimates, standard errors, and bootstrap confidence intervals for IQA metrics.

    Parameters
    ----------
    predictions : np.ndarray
        Model predicted perceptual distances.
    targets : np.ndarray
        Ground truth MOS/DMOS.
    reference_ids : np.ndarray, optional
        Reference image / scene identifiers. If provided, bootstrapping is performed by
        sampling unique reference images with replacement (content-aware clustering),
        preserving scene-level correlation structure. If None, samples individual pairs.
    n_bootstraps : int, default=1000
        Number of bootstrap resamples.
    ci : float, default=0.95
        Confidence interval level (e.g., 0.95 for 95% CI).
    seed : int, default=42
        Random seed for reproducibility.
    fit_logistic : bool, default=True
        Whether to use 4-parameter logistic mapping for RMSE/R².

    Returns
    -------
    dict
        Nested dictionary where each metric (plcc, abs_plcc, srocc, abs_srocc, rmse, r2) has:
        - 'point_estimate': point estimate on full dataset
        - 'mean': mean over bootstrap iterations
        - 'std': standard error (SEM) across iterations
        - 'ci_lower': lower bound of confidence interval
        - 'ci_upper': upper bound of confidence interval
        - 'distribution': array of values for all bootstrap iterations
    """
    predictions = np.asarray(predictions, dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.float64).ravel()
    n_samples = len(predictions)

    # Point estimate on full sample
    point_estimates = compute_all_metrics(predictions, targets, fit_logistic=fit_logistic)

    # Precompute logistic mapping on the full dataset once (ITU-T P.1401 standard)
    if fit_logistic:
        mapped_predictions, _ = fit_logistic_mapping(predictions, targets)
    else:
        mapped_predictions = predictions

    # Pre-rank targets and predictions for ultra-fast SROCC / PLCC
    pred_ranks = stats.rankdata(predictions)
    target_ranks = stats.rankdata(targets)

    rng = np.random.default_rng(seed)
    bootstrap_records = {k: np.empty(n_bootstraps, dtype=np.float64) for k in point_estimates.keys()}

    if reference_ids is not None:
        ref_ids = np.asarray(reference_ids).ravel()
        unique_refs = np.unique(ref_ids)
        n_refs = len(unique_refs)
        ref_to_indices = {ref: np.where(ref_ids == ref)[0] for ref in unique_refs}

        for b in range(n_bootstraps):
            sampled_refs = rng.choice(unique_refs, size=n_refs, replace=True)
            sampled_idx = np.concatenate([ref_to_indices[ref] for ref in sampled_refs])

            b_pred = predictions[sampled_idx]
            b_target = targets[sampled_idx]
            b_mapped = mapped_predictions[sampled_idx]

            # Fast correlations
            p = float(stats.pearsonr(b_pred, b_target)[0])
            s = float(stats.spearmanr(b_pred, b_target)[0])
            k = float(stats.kendalltau(b_pred, b_target)[0]) if len(b_pred) < 5000 else float(s)

            # Error metrics on pre-aligned predictions
            err = b_target - b_mapped
            rmse = float(np.sqrt(np.mean(err ** 2)))
            ss_tot = np.sum((b_target - np.mean(b_target)) ** 2)
            r2 = float(1.0 - (np.sum(err ** 2) / ss_tot)) if ss_tot > 1e-9 else 0.0

            bootstrap_records["plcc"][b] = p
            bootstrap_records["abs_plcc"][b] = abs(p)
            bootstrap_records["srocc"][b] = s
            bootstrap_records["abs_srocc"][b] = abs(s)
            bootstrap_records["krocc"][b] = k
            bootstrap_records["abs_krocc"][b] = abs(k)
            bootstrap_records["rmse"][b] = rmse
            bootstrap_records["r2"][b] = r2
    else:
        sampled_indices = rng.choice(n_samples, size=(n_bootstraps, n_samples), replace=True)
        for b in range(n_bootstraps):
            sampled_idx = sampled_indices[b]
            b_pred = predictions[sampled_idx]
            b_target = targets[sampled_idx]
            b_mapped = mapped_predictions[sampled_idx]

            p = float(stats.pearsonr(b_pred, b_target)[0])
            s = float(stats.spearmanr(b_pred, b_target)[0])
            k = float(stats.kendalltau(b_pred, b_target)[0]) if len(b_pred) < 5000 else float(s)

            err = b_target - b_mapped
            rmse = float(np.sqrt(np.mean(err ** 2)))
            ss_tot = np.sum((b_target - np.mean(b_target)) ** 2)
            r2 = float(1.0 - (np.sum(err ** 2) / ss_tot)) if ss_tot > 1e-9 else 0.0

            bootstrap_records["plcc"][b] = p
            bootstrap_records["abs_plcc"][b] = abs(p)
            bootstrap_records["srocc"][b] = s
            bootstrap_records["abs_srocc"][b] = abs(s)
            bootstrap_records["krocc"][b] = k
            bootstrap_records["abs_krocc"][b] = abs(k)
            bootstrap_records["rmse"][b] = rmse
            bootstrap_records["r2"][b] = r2


    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    results = {}
    for k, point_val in point_estimates.items():
        dist = bootstrap_records[k]
        ci_lower = float(np.percentile(dist, lower_pct))
        ci_upper = float(np.percentile(dist, upper_pct))
        results[k] = {
            "point_estimate": float(point_val),
            "mean": float(np.mean(dist)),
            "std": float(np.std(dist)),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "distribution": dist,
        }

    return results


def paired_bootstrap_gap_test(
    train_preds_1: np.ndarray,
    train_targets_1: np.ndarray,
    test_preds_1: np.ndarray,
    test_targets_1: np.ndarray,
    train_preds_2: np.ndarray,
    train_targets_2: np.ndarray,
    test_preds_2: np.ndarray,
    test_targets_2: np.ndarray,
    n_bootstraps: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
    metric: str = "abs_plcc",
) -> Dict[str, Any]:
    """Perform a paired bootstrap hypothesis test comparing generalization gaps between two models.

    The generalization gap is defined as:
        Δ_model = |metric_train| - |metric_test|

    The test evaluates the differential gap:
        δ = Δ_model1 - Δ_model2

    Hypotheses:
        H0: δ <= 0  (Model 2 does not have a smaller generalization gap than Model 1)
        H1: δ > 0   (Model 2 significantly reduces the generalization gap compared to Model 1)

    Parameters
    ----------
    train_preds_1, train_targets_1 : np.ndarray
        Model 1 predictions and targets on training/validation set.
    test_preds_1, test_targets_1 : np.ndarray
        Model 1 predictions and targets on external test set (e.g. KADID-10k).
    train_preds_2, train_targets_2 : np.ndarray
        Model 2 predictions and targets on training/validation set.
    test_preds_2, test_targets_2 : np.ndarray
        Model 2 predictions and targets on external test set.
    n_bootstraps : int, default=10000
        Number of paired bootstrap iterations.
    ci : float, default=0.95
        Confidence level.
    seed : int, default=42
        Random seed.
    metric : str, default='abs_plcc'
        Metric key to test ('abs_plcc', 'abs_srocc', 'rmse', 'r2').

    Returns
    -------
    dict
        Summary of the hypothesis test:
        - 'gap_model_1': mean and CI of Model 1's generalization gap
        - 'gap_model_2': mean and CI of Model 2's generalization gap
        - 'differential_gap_mean': mean δ (Δ_1 - Δ_2)
        - 'differential_gap_ci': (ci_lower, ci_upper) of δ
        - 'p_value_one_sided': empirical p-value for H1 (P(δ <= 0))
        - 'p_value_two_sided': empirical two-sided p-value
        - 'statistically_significant': bool (p_value_one_sided < 1 - ci)
    """
    rng = np.random.default_rng(seed)

    n_train = len(train_preds_1)
    n_test = len(test_preds_1)

    delta_1_dist = np.empty(n_bootstraps, dtype=np.float64)
    delta_2_dist = np.empty(n_bootstraps, dtype=np.float64)
    delta_diff_dist = np.empty(n_bootstraps, dtype=np.float64)

    for b in range(n_bootstraps):
        # Resample train and test paired indices
        train_idx = rng.choice(n_train, size=n_train, replace=True)
        test_idx = rng.choice(n_test, size=n_test, replace=True)

        # Model 1 metrics
        m1_train = compute_all_metrics(train_preds_1[train_idx], train_targets_1[train_idx])[metric]
        m1_test = compute_all_metrics(test_preds_1[test_idx], test_targets_1[test_idx])[metric]
        gap_1 = abs(m1_train) - abs(m1_test) if "abs" in metric else m1_train - m1_test

        # Model 2 metrics
        m2_train = compute_all_metrics(train_preds_2[train_idx], train_targets_2[train_idx])[metric]
        m2_test = compute_all_metrics(test_preds_2[test_idx], test_targets_2[test_idx])[metric]
        gap_2 = abs(m2_train) - abs(m2_test) if "abs" in metric else m2_train - m2_test

        diff = gap_1 - gap_2

        delta_1_dist[b] = gap_1
        delta_2_dist[b] = gap_2
        delta_diff_dist[b] = diff

    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    p_one_sided = float(np.mean(delta_diff_dist <= 0.0))
    p_two_sided = float(2.0 * min(p_one_sided, 1.0 - p_one_sided))

    return {
        "metric": metric,
        "gap_model_1": {
            "mean": float(np.mean(delta_1_dist)),
            "std": float(np.std(delta_1_dist)),
            "ci": (float(np.percentile(delta_1_dist, lower_pct)), float(np.percentile(delta_1_dist, upper_pct))),
        },
        "gap_model_2": {
            "mean": float(np.mean(delta_2_dist)),
            "std": float(np.std(delta_2_dist)),
            "ci": (float(np.percentile(delta_2_dist, lower_pct)), float(np.percentile(delta_2_dist, upper_pct))),
        },
        "differential_gap_mean": float(np.mean(delta_diff_dist)),
        "differential_gap_ci": (
            float(np.percentile(delta_diff_dist, lower_pct)),
            float(np.percentile(delta_diff_dist, upper_pct)),
        ),
        "p_value_one_sided": p_one_sided,
        "p_value_two_sided": p_two_sided,
        "statistically_significant": bool(p_one_sided < alpha),
    }
