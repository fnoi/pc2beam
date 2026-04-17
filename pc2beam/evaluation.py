"""
Evaluation helpers for per-instance beam direction estimation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from .processing import project_points_to_line


def _safe_unit(vec: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return None
    return vec / norm


def _axis_angle_error_deg(pred_vec: np.ndarray, gt_vec: np.ndarray) -> Optional[float]:
    pred_u = _safe_unit(np.asarray(pred_vec, dtype=np.float64))
    gt_u = _safe_unit(np.asarray(gt_vec, dtype=np.float64))
    if pred_u is None or gt_u is None:
        return None
    cos_val = float(np.clip(np.abs(np.dot(pred_u, gt_u)), 0.0, 1.0))
    return float(np.rad2deg(np.arccos(cos_val)))


def _pairing_endpoint_error(
    pred_start: np.ndarray,
    pred_end: np.ndarray,
    gt_start: np.ndarray,
    gt_end: np.ndarray,
) -> float:
    d_direct = float(np.linalg.norm(pred_start - gt_start) + np.linalg.norm(pred_end - gt_end))
    d_swapped = float(np.linalg.norm(pred_start - gt_end) + np.linalg.norm(pred_end - gt_start))
    return min(d_direct, d_swapped) / 2.0


def evaluate_s2_against_ground_truth(
    points: np.ndarray,
    instances: np.ndarray,
    s2_features: Dict[Any, Dict[str, Any]],
    ground_truth_yaml: Union[str, Path],
) -> pd.DataFrame:
    """
    Compare per-instance predicted centerlines with IFC ground truth.
    """
    gt_data = OmegaConf.to_container(OmegaConf.load(Path(ground_truth_yaml)), resolve=True)
    if not isinstance(gt_data, dict):
        raise ValueError("Ground truth YAML must contain a dictionary of beam entries.")

    rows = []
    unique_instances = np.unique(instances)
    for instance_id in unique_instances:
        feature = s2_features.get(instance_id)
        if feature is None:
            continue

        gt_key = str(int(instance_id) + 1)
        gt_entry = gt_data.get(gt_key)
        if gt_entry is None:
            continue

        gt_start = np.asarray(gt_entry["start"], dtype=np.float64)
        gt_end = np.asarray(gt_entry["end"], dtype=np.float64)
        gt_vec = gt_end - gt_start
        gt_length = float(np.linalg.norm(gt_vec))

        pred_vec = feature.get("s2_vector")
        pred_point = feature.get("s2_point")
        status = feature.get("status", "unknown")

        angle_error_deg = None
        endpoint_error = None
        length_error = None
        pred_length = None

        if pred_vec is not None and pred_point is not None:
            instance_points = points[instances == instance_id]
            if len(instance_points) >= 2:
                pred_start, pred_end = project_points_to_line(
                    instance_points,
                    np.asarray(pred_vec, dtype=np.float64),
                    np.asarray(pred_point, dtype=np.float64),
                )
                pred_length = float(np.linalg.norm(pred_end - pred_start))
                angle_error_deg = _axis_angle_error_deg(
                    np.asarray(pred_vec, dtype=np.float64),
                    gt_vec,
                )
                endpoint_error = _pairing_endpoint_error(
                    np.asarray(pred_start, dtype=np.float64),
                    np.asarray(pred_end, dtype=np.float64),
                    gt_start,
                    gt_end,
                )
                length_error = abs(pred_length - gt_length)

        rows.append(
            {
                "instance_id": int(instance_id),
                "status": status,
                "confidence": float(feature.get("confidence", 0.0)),
                "point_count": int(feature.get("point_count", 0)),
                "plane1_inliers": int(feature.get("plane1_inliers", 0)),
                "plane2_inliers": int(feature.get("plane2_inliers", 0)),
                "plane_angle_deg": feature.get("plane_angle_deg"),
                "axis_angle_error_deg": angle_error_deg,
                "endpoint_error": endpoint_error,
                "pred_length": pred_length,
                "gt_length": gt_length,
                "length_error": length_error,
            }
        )

    return pd.DataFrame(rows)


def summarize_s2_metrics(metrics_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build aggregate statistics from per-instance metrics.
    """
    if metrics_df.empty:
        return {
            "total_instances": 0,
            "valid_predictions": 0,
            "failed_predictions": 0,
            "angle_error_mean_deg": None,
            "angle_error_median_deg": None,
            "angle_error_p95_deg": None,
            "endpoint_error_mean": None,
            "length_error_mean": None,
        }

    valid = metrics_df[metrics_df["axis_angle_error_deg"].notna()]
    angle_series = valid["axis_angle_error_deg"] if not valid.empty else pd.Series(dtype=float)
    endpoint_series = valid["endpoint_error"] if not valid.empty else pd.Series(dtype=float)
    length_series = valid["length_error"] if not valid.empty else pd.Series(dtype=float)

    return {
        "total_instances": int(len(metrics_df)),
        "valid_predictions": int(len(valid)),
        "failed_predictions": int(len(metrics_df) - len(valid)),
        "angle_error_mean_deg": float(angle_series.mean()) if not angle_series.empty else None,
        "angle_error_median_deg": float(angle_series.median()) if not angle_series.empty else None,
        "angle_error_p95_deg": float(angle_series.quantile(0.95)) if not angle_series.empty else None,
        "endpoint_error_mean": float(endpoint_series.mean()) if not endpoint_series.empty else None,
        "length_error_mean": float(length_series.mean()) if not length_series.empty else None,
    }


def summarize_catalogue_fit(catalogue_fit: Dict[Any, Dict[str, Any]]) -> Dict[str, Any]:
    if not catalogue_fit:
        return {
            "total_instances": 0,
            "fit_success": 0,
            "fit_failed": 0,
            "unique_profiles": 0,
            "top_profiles": {},
            "score_mean": None,
        }
    rows = []
    for instance_id, result in catalogue_fit.items():
        if result.get("ok"):
            fitness = result.get("fitness", {})
            rows.append(
                {
                    "instance_id": int(instance_id),
                    "ok": True,
                    "cstype": result.get("cstype"),
                    "score": fitness.get("score"),
                }
            )
        else:
            rows.append(
                {
                    "instance_id": int(instance_id),
                    "ok": False,
                    "cstype": None,
                    "score": None,
                }
            )
    df = pd.DataFrame(rows)
    ok_df = df[df["ok"]]
    top_profiles = (
        ok_df["cstype"].value_counts().head(5).to_dict()
        if not ok_df.empty and ok_df["cstype"].notna().any()
        else {}
    )
    return {
        "total_instances": int(len(df)),
        "fit_success": int(len(ok_df)),
        "fit_failed": int(len(df) - len(ok_df)),
        "unique_profiles": int(ok_df["cstype"].nunique()) if not ok_df.empty else 0,
        "top_profiles": top_profiles,
        "score_mean": float(ok_df["score"].mean()) if not ok_df.empty else None,
    }
