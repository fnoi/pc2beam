#!/usr/bin/env python3
"""
Bundled-sample E2E run through ``run_pc2beam`` with optional Plotly HTML export and ``fig.show``.

Intended as a stable dev substitute for the Jupyter demo notebook.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

from run import run_pc2beam
from pc2beam import config_io, viz
from pc2beam.evaluation import summarize_catalogue_fit
from pc2beam.ifc_io import resolve_catalogue_csv_path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_POINTS = REPO_ROOT / "data" / "test_points.txt"
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"
DEFAULT_GT = REPO_ROOT / "data" / "test_points_gt.yaml"
HELIOS_ROOT = REPO_ROOT / "output" / "helios_pc2beam"
HELIOS_POINTS_REL = Path("pc2beam_input") / "points_with_normals_instances.txt"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run pc2beam on the bundled sample (smoke / dev demo)."
    )
    p.add_argument(
        "--points",
        type=Path,
        default=None,
        help="Point cloud TXT (default: latest HELIOS run points_with_normals_instances.txt).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config (default: {DEFAULT_CONFIG})",
    )
    p.add_argument(
        "--evaluate",
        action="store_true",
        help="Run S2 evaluation against ground-truth YAML.",
    )
    p.add_argument(
        "--gt-yaml",
        type=Path,
        default=None,
        help=f"Ground truth YAML when --evaluate (default: {DEFAULT_GT})",
    )
    p.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Write per-instance metrics CSV (only with --evaluate).",
    )
    p.add_argument(
        "--legacy",
        action="store_true",
        help="Force legacy projection step (enabled by default in this demo).",
    )
    p.add_argument(
        "--catalogue",
        action="store_true",
        help="Force catalogue cross-section fitting (enabled by default in this demo).",
    )
    p.add_argument(
        "--catalogue-csv-path",
        type=Path,
        default=None,
        help=(
            "Catalogue CSV path when using --catalogue "
            "(default from config: data/European_Steel_Section_Properties.csv)."
        ),
    )
    p.add_argument(
        "--ifc-catalogue-path",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--viz",
        action="store_true",
        help="After the pipeline, export Plotly HTML under --viz-dir and call fig.show().",
    )
    p.add_argument(
        "--viz-dir",
        type=Path,
        default=None,
        help="Directory for HTML exports (default: output/demo_runs/<UTC timestamp>/).",
    )
    p.add_argument(
        "--browser-renderer",
        default="browser",
        help="Plotly renderer for fig.show() when using --viz (default: browser).",
    )
    p.add_argument(
        "--cross-sections",
        action="store_true",
        help=(
            "After the pipeline, write a 3x3 HTML figure of random beam cross-sections "
            "(requires legacy projection: --legacy or legacy_projection.enabled in config)."
        ),
    )
    p.add_argument(
        "--cross-section-seed",
        type=int,
        default=42,
        help="RNG seed for picking up to 9 instances for --cross-sections (default: 42).",
    )
    p.add_argument(
        "--catalogue-n-downsample",
        type=int,
        default=None,
        help="Override catalogue_fit.n_downsample (0 = full resolution; typical legacy value: 200).",
    )
    p.add_argument(
        "--polygon-subdivision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override catalogue_fit.polygon_subdivision (edge subdivision with edge_subdivision_lmax).",
    )
    return p.parse_args()


def _find_latest_helios_inputs() -> tuple[Path | None, Path | None]:
    if not HELIOS_ROOT.exists():
        return None, None

    run_dirs = [d for d in HELIOS_ROOT.iterdir() if d.is_dir()]
    run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

    for run_dir in run_dirs:
        points_path = run_dir / HELIOS_POINTS_REL
        if not points_path.is_file():
            continue
        gt_candidates = sorted((run_dir / "pc2beam_input").glob("*_gt.yaml"))
        gt_path = gt_candidates[0] if gt_candidates else None
        return points_path, gt_path

    return None, None


def _plotly_features(point_cloud) -> Dict[str, Any]:
    """``compute_s1`` stores the full calculate_s1 dict under ``features['s1']``."""
    raw = point_cloud.features.get("s1")
    if raw is None:
        return {}
    if isinstance(raw, dict) and "s1" in raw:
        return {"s1": raw["s1"]}
    return {"s1": np.asarray(raw)}


def _emit_viz(
    *,
    point_cloud,
    skeleton,
    viz_dir: Path,
    renderer: str,
) -> None:
    viz_dir.mkdir(parents=True, exist_ok=True)
    stem = 1

    feats = _plotly_features(point_cloud)

    if point_cloud.has_instances:
        print(f"[demo] viz: point cloud by instance -> {viz_dir / f'{stem:02d}_point_cloud_by_instance.html'}")
        fig_pc = viz.plot_point_cloud(
            point_cloud.points,
            normals=point_cloud.normals if point_cloud.has_normals else None,
            instances=point_cloud.instances,
            features=feats if feats else None,
            mode="points",
            color_by="instance",
            show_vectors=False,
            title="[demo] point cloud by instance",
        )
        viz.show_or_export_plot(
            fig_pc,
            viz_dir / f"{stem:02d}_point_cloud_by_instance.html",
            renderer=renderer,
        )
        stem += 1

        s1_arr = feats.get("s1") if feats else None
        if s1_arr is not None and getattr(s1_arr, "ndim", 0) == 2 and s1_arr.shape[1] == 3:
            print(f"[demo] viz: s1 supernormals -> {viz_dir / f'{stem:02d}_s1_supernormals.html'}")
            fig_s1 = viz.plot_point_cloud(
                point_cloud.points,
                normals=point_cloud.normals if point_cloud.has_normals else None,
                instances=point_cloud.instances,
                features=feats,
                mode="supernormals",
                color_by="instance",
                show_vectors=True,
                title="[demo] s1 supernormals",
            )
            viz.show_or_export_plot(
                fig_s1,
                viz_dir / f"{stem:02d}_s1_supernormals.html",
                renderer=renderer,
            )
            stem += 1
    else:
        print("[demo] viz: skip instance / s1 point plots (no instance labels)")

    print(f"[demo] viz: skeleton -> {viz_dir / f'{stem:02d}_skeleton.html'}")
    fig_sk = viz.plot_skeleton(skeleton)
    viz.show_or_export_plot(fig_sk, viz_dir / f"{stem:02d}_skeleton.html", renderer=renderer)
    stem += 1

    if point_cloud.has_instances:
        print(f"[demo] viz: skeleton with points -> {viz_dir / f'{stem:02d}_skeleton_with_points.html'}")
        fig_combo = viz.plot_skeleton_with_points(
            point_cloud.points,
            point_cloud.instances,
            skeleton,
            title="[demo] skeleton with point cloud",
        )
        viz.show_or_export_plot(
            fig_combo,
            viz_dir / f"{stem:02d}_skeleton_with_points.html",
            renderer=renderer,
        )
    else:
        print("[demo] viz: skip skeleton_with_points (no instance labels)")


def main() -> None:
    args = _parse_args()
    args.viz = True
    args.cross_sections = True
    args.legacy = True
    args.catalogue = True
    latest_points, latest_gt = _find_latest_helios_inputs()

    if args.metrics_csv and not args.evaluate:
        raise SystemExit("--metrics-csv requires --evaluate")

    gt_path = args.gt_yaml
    if args.evaluate:
        if gt_path is None:
            gt_path = latest_gt if latest_gt is not None else DEFAULT_GT
        gt_path = Path(gt_path)
        if not gt_path.is_file():
            raise SystemExit(f"Ground truth YAML not found: {gt_path}")

    points_path = Path(args.points) if args.points is not None else (
        latest_points if latest_points is not None else DEFAULT_POINTS
    )
    if not points_path.is_file():
        raise SystemExit(f"Point cloud file not found: {points_path}")

    config_path = Path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"Config file not found: {config_path}")

    config = config_io.load_config(config_path)
    fit_cfg = config.get("catalogue_fit", {})
    cfg_catalogue_path = fit_cfg.get("catalogue_csv_path")
    if cfg_catalogue_path is None:
        cfg_catalogue_path = fit_cfg.get("ifc_catalogue_path")
    cfg_catalogue_region = str(fit_cfg.get("catalogue_region", "eur"))
    cli_catalogue_override = (
        args.catalogue_csv_path
        if args.catalogue_csv_path is not None
        else args.ifc_catalogue_path
    )
    effective_catalogue_path = resolve_catalogue_csv_path(
        catalogue_region=cfg_catalogue_region,
        override_path=cli_catalogue_override if cli_catalogue_override is not None else cfg_catalogue_path,
    )
    if effective_catalogue_path is not None and not effective_catalogue_path.is_file():
        raise SystemExit(f"Catalogue CSV file not found: {effective_catalogue_path}")

    artifact_dir: Path | None = None
    if args.viz or args.cross_sections:
        if args.viz_dir is not None:
            artifact_dir = Path(args.viz_dir)
        else:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            artifact_dir = REPO_ROOT / "output" / "demo_runs" / ts

    viz_dir = artifact_dir if args.viz else None

    print(f"[demo] points={points_path}")
    print(f"[demo] config={config_path}")
    if effective_catalogue_path is not None:
        print(f"[demo] catalogue_csv={effective_catalogue_path} region={cfg_catalogue_region}")
    if args.evaluate:
        print(f"[demo] evaluate=True gt={gt_path}")

    catalogue_fit_overrides = {}
    if args.catalogue_n_downsample is not None:
        catalogue_fit_overrides["n_downsample"] = int(args.catalogue_n_downsample)
    if args.polygon_subdivision is not None:
        catalogue_fit_overrides["polygon_subdivision"] = bool(args.polygon_subdivision)

    print("[demo] running run_pc2beam (through legacy projection stage) ...")
    result = run_pc2beam(
        input_file=str(points_path),
        config_path=str(config_path),
        entry="instance",
        ground_truth_yaml=str(gt_path) if args.evaluate else None,
        evaluate=args.evaluate,
        metrics_output_csv=str(args.metrics_csv) if args.metrics_csv else None,
        run_legacy_projection=args.legacy,
        run_catalogue_fit=False,
        catalogue_csv_path=str(effective_catalogue_path)
        if effective_catalogue_path is not None
        else None,
        catalogue_fit_overrides=catalogue_fit_overrides if catalogue_fit_overrides else None,
        visualize=True,
        run_s1=False,
    )
    print("[demo] projection stage finished")

    if args.cross_sections:
        pc = result["point_cloud"]
        legacy = pc.features.get("legacy_projection")
        if legacy is None:
            raise SystemExit(
                "[demo] --cross-sections requires legacy_projection in point cloud features. "
                "Use --legacy or set legacy_projection.enabled in config."
            )
        rng = np.random.default_rng(int(args.cross_section_seed))
        cat = pc.features.get("catalogue_fit")
        cat_arg = cat if isinstance(cat, dict) and cat else None
        ids = viz.select_cross_section_instance_ids(
            legacy,
            catalogue_fit=cat_arg,
            n=9,
            rng=rng,
        )
        if not ids:
            req = (
                "ok legacy_projection and ok catalogue_fit"
                if cat_arg
                else "ok legacy_projection"
            )
            raise SystemExit(f"[demo] --cross-sections: no instances with {req}.")
        if artifact_dir is None:
            raise SystemExit("[demo] internal error: artifact_dir unset for --cross-sections")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        out_html_prefit = artifact_dir / "cross_sections_prefit_9.html"
        print(f"[demo] cross-sections (pre-fit): {len(ids)} instance(s) -> {out_html_prefit}")
        print("[demo] cross-sections: overlaying projection intersection lines when available")
        fig_cs_prefit = viz.plot_cross_section_grid(
            pc,
            ids,
            show_projection_lines=True,
            grid_title="[demo] beam cross-sections before catalogue fit (2D legacy projection)",
        )
        viz.show_or_export_plot(
            fig_cs_prefit,
            out_html_prefit,
            renderer=args.browser_renderer,
            show=bool(args.viz),
        )

    if args.catalogue:
        pc = result["point_cloud"]
        print("[demo] running catalogue fitting ...")
        km_rs = fit_cfg.get("kmeans_random_seed")
        pc.fit_cross_sections_from_catalogue(
            catalogue_csv_path=(
                str(effective_catalogue_path) if effective_catalogue_path is not None else None
            ),
            ifc_catalogue_path=str(args.ifc_catalogue_path) if args.ifc_catalogue_path else None,
            catalogue_region=cfg_catalogue_region,
            n_pop=int(fit_cfg.get("n_pop", 100)),
            n_gen=int(fit_cfg.get("n_gen", 20)),
            show_progress=bool(fit_cfg.get("show_progress", True)),
            show_generation_progress=bool(fit_cfg.get("show_generation_progress", False)),
            progress_min_interval=float(fit_cfg.get("progress_min_interval", 0.25)),
            n_downsample=int(catalogue_fit_overrides.get("n_downsample", fit_cfg.get("n_downsample", 0))),
            polygon_subdivision=bool(
                catalogue_fit_overrides.get(
                    "polygon_subdivision",
                    fit_cfg.get("polygon_subdivision", False),
                )
            ),
            edge_subdivision_lmax=float(fit_cfg.get("edge_subdivision_lmax", 0.01)),
            use_cluster_weights=bool(fit_cfg.get("use_cluster_weights", True)),
            kmeans_random_seed=int(km_rs) if km_rs is not None else None,
            n_jobs=int(fit_cfg.get("n_jobs", -1)),
        )
        catalogue_fit_summary = summarize_catalogue_fit(pc.features.get("catalogue_fit", {}))
        print("[demo] catalogue fitting summary:")
        for key, value in catalogue_fit_summary.items():
            print(f"  - {key}: {value}")

    if args.cross_sections and args.catalogue:
        pc = result["point_cloud"]
        legacy = pc.features.get("legacy_projection")
        rng = np.random.default_rng(int(args.cross_section_seed))
        cat = pc.features.get("catalogue_fit")
        cat_arg = cat if isinstance(cat, dict) and cat else None
        ids = viz.select_cross_section_instance_ids(
            legacy,
            catalogue_fit=cat_arg,
            n=9,
            rng=rng,
        )
        if ids and artifact_dir is not None:
            out_html = artifact_dir / "cross_sections_9.html"
            print(f"[demo] cross-sections (post-fit): {len(ids)} instance(s) -> {out_html}")
            fig_cs = viz.plot_cross_section_grid(
                pc,
                ids,
                show_projection_lines=True,
                grid_title="[demo] beam cross-sections after catalogue fit (2D legacy projection)",
            )
            viz.show_or_export_plot(
                fig_cs,
                out_html,
                renderer=args.browser_renderer,
                show=bool(args.viz),
            )

    if args.viz and viz_dir is not None:
        print(f"[demo] viz output dir: {viz_dir}")
        _emit_viz(
            point_cloud=result["point_cloud"],
            skeleton=result["skeleton"],
            viz_dir=viz_dir,
            renderer=args.browser_renderer,
        )
        print("[demo] viz complete")


if __name__ == "__main__":
    main()
