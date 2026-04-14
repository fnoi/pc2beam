"""
Optional LAS / LAZ reading for simulated scan visualization.

With ``helios_one_obj_per_instance`` exports, HELIOS++ usually writes a per-hit
``hitObjectId`` extra dimension equal to the **0-based scene part index**. Map to
IFC ``instance_id`` using ``meshed_instance_ids_in_part_order[hitObjectId]`` from
``scene_sidecar.yaml``, or :func:`pc2beam.ifc_mesh_export.instance_id_for_las_hit_object_id`.

Single merged OBJ scenes load as one part, so ``hitObjectId`` may be constant (often 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


@dataclass
class LasScan:
    """Arrays from a LAS/LAZ point record; optional channels may be missing."""

    points: np.ndarray
    intensity: Optional[np.ndarray] = None
    classification: Optional[np.ndarray] = None
    rgb: Optional[np.ndarray] = None
    hit_object_id: Optional[np.ndarray] = None


def _hit_object_id_array(las) -> Optional[np.ndarray]:
    """Best-effort read of HELIOS++ ``hitObjectId`` (laspy dimension naming varies)."""
    for name in ("hit_object_id", "HitObjectId", "hitObjectId"):
        if hasattr(las, name):
            return np.asarray(getattr(las, name))
    try:
        dim = las.point_format.extra_dimensions
        for d in dim:
            dn = getattr(d, "name", "") or ""
            if dn in ("hit_object_id", "HitObjectId", "hitObjectId"):
                return np.asarray(las[dn])
    except Exception:
        pass
    return None


def read_las_scan(path: Union[str, Path]) -> LasScan:
    """
    Load points and optional intensity, classification, RGB, and hit_object_id from LAS/LAZ.

    RGB is returned as (N, 3) uint16 when all of red/green/blue exist on the point format.
    """
    try:
        import laspy
    except ImportError as e:
        raise ImportError("Install laspy to read LAS files: pip install laspy") from e

    path = Path(path)
    las = laspy.read(str(path))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    pts = np.column_stack([x, y, z])

    intens = getattr(las, "intensity", None)
    if intens is not None:
        intens = np.asarray(intens)

    classification = getattr(las, "classification", None)
    if classification is not None:
        classification = np.asarray(classification)

    rgb: Optional[np.ndarray] = None
    if all(hasattr(las, c) for c in ("red", "green", "blue")):
        rgb = np.column_stack(
            [
                np.asarray(las.red),
                np.asarray(las.green),
                np.asarray(las.blue),
            ]
        )

    hit_oid = _hit_object_id_array(las)

    return LasScan(
        points=pts,
        intensity=intens,
        classification=classification,
        rgb=rgb,
        hit_object_id=hit_oid,
    )


def read_las_xyz_intensity(path: Union[str, Path]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load points and intensity from a LAS/LAZ file using ``laspy`` when installed.

    Returns
    -------
    points
        (N, 3) float64 array of X, Y, Z.
    intensity
        (N,) array if present, else ``None``.
    """
    scan = read_las_scan(path)
    return scan.points, scan.intensity


def las_dimension_names(path: Union[str, Path]) -> List[str]:
    """Point format dimension names (useful when checking HELIOS++ LAS outputs)."""
    try:
        import laspy
    except ImportError as e:
        raise ImportError("Install laspy to read LAS files: pip install laspy") from e

    las = laspy.read(str(Path(path)))
    pf = las.point_format
    names = getattr(pf, "dimension_names", None)
    if names is not None:
        return list(names)
    dims = getattr(pf, "dimensions", None)
    if dims is not None:
        return [getattr(d, "name", str(d)) for d in dims]
    return []


def find_first_las(directory: Union[str, Path]) -> Optional[Path]:
    """Return the first ``*.las`` file in a directory tree (shallow walk)."""
    directory = Path(directory)
    if not directory.is_dir():
        return None
    for p in sorted(directory.rglob("*.las")):
        return p
    return None
