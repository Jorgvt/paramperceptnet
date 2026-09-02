"""Analysis and statistical evaluation tools for ParamPerceptNet."""

from paramperceptnet.analysis.bootstrap import (
    bootstrap_metrics,
    paired_bootstrap_gap_test,
    fit_logistic_mapping,
    logistic_4param,
    pearson_corr,
    spearman_corr,
    kendall_corr,
    rmse_score,
    r2_metric,
    compute_all_metrics,
)

__all__ = [
    "bootstrap_metrics",
    "paired_bootstrap_gap_test",
    "fit_logistic_mapping",
    "logistic_4param",
    "pearson_corr",
    "spearman_corr",
    "kendall_corr",
    "rmse_score",
    "r2_metric",
    "compute_all_metrics",
]
