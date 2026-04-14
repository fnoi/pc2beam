"""
Generate per-beam ground-truth YAML from IFC geometry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import ifcopenshell.geom as geom
import numpy as np
from omegaconf import OmegaConf

from pc2beam.ifc_io import beam_profile_and_area, iter_beam_records, open_ifc, type_relation_info


def _geom_settings() -> geom.settings:
    settings = geom.settings()
    settings.set("use-world-coords", True)
    return settings


def _beam_vertices(settings: geom.settings, beam) -> Optional[np.ndarray]:
    try:
        shape = geom.create_shape(settings, beam)
    except Exception:
        return None
    verts_flat = np.array(shape.geometry.verts, dtype=np.float64)
    if verts_flat.size == 0:
        return None
    return verts_flat.reshape(-1, 3)


def _principal_endpoints(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = vertices.mean(axis=0)
    centered = vertices - center
    cov = centered.T @ centered
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    axis = eig_vecs[:, int(np.argmax(eig_vals))]
    axis /= max(np.linalg.norm(axis), 1e-12)

    projections = centered @ axis
    p0 = center + axis * float(np.min(projections))
    p1 = center + axis * float(np.max(projections))

    # Fix axis sign ambiguity for deterministic start/end ordering.
    if tuple(p1.tolist()) < tuple(p0.tolist()):
        p0, p1 = p1, p0
    return p0, center, p1


def _round3(point: np.ndarray) -> Tuple[float, float, float]:
    return (round(float(point[0]), 6), round(float(point[1]), 6), round(float(point[2]), 6))


def _beam_type_label(beam, profile_name: str) -> str:
    tinfo = type_relation_info(beam)
    for key in ("beam_type_tag", "element_type_name", "predefined_type"):
        value = tinfo.get(key)
        if value:
            return str(value)
    if profile_name and profile_name != "unknown_profile":
        return str(profile_name)
    return beam.is_a()


def build_ifc_beam_ground_truth(ifc_path: Union[str, Path]) -> Dict[str, Dict[str, Union[str, float]]]:
    """
    Build per-beam GT fields using world-space beam geometry.

    Returns dict keyed as "1", "2", ... with fields:
    start, end, beam_type.
    """
    ifc_path = Path(ifc_path)
    records = iter_beam_records(ifc_path)
    rec_by_gid = {r.global_id: r for r in records}
    ifc_file = open_ifc(ifc_path)
    settings = _geom_settings()

    out: Dict[str, Dict[str, Union[str, float]]] = {}
    beam_index = 0

    for beam in sorted(ifc_file.by_type("IfcBeam"), key=lambda b: (b.id(), getattr(b, "GlobalId", "") or "")):
        vertices = _beam_vertices(settings, beam)
        if vertices is None:
            continue
        beam_index += 1
        key = str(beam_index)

        gid = getattr(beam, "GlobalId", "") or ""
        rec = rec_by_gid.get(gid)
        profile_name = rec.profile_name if rec is not None else beam_profile_and_area(beam)[0]
        beam_type = _beam_type_label(beam, profile_name)

        start, _, end = _principal_endpoints(vertices)
        sx, sy, sz = _round3(start)
        ex, ey, ez = _round3(end)

        out[key] = {
            "start": [sx, sy, sz],
            "end": [ex, ey, ez],
            "beam_type": beam_type,
        }

    return out


def write_ifc_beam_ground_truth_yaml(
    ifc_path: Union[str, Path],
    output_yaml: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Write `<ifc_stem>_gt.yaml` next to IFC by default and return file path.
    """
    ifc_path = Path(ifc_path)
    output_yaml = Path(output_yaml) if output_yaml is not None else ifc_path.with_name(f"{ifc_path.stem}_gt.yaml")
    data = build_ifc_beam_ground_truth(ifc_path)
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(data), output_yaml)
    return output_yaml.resolve()
