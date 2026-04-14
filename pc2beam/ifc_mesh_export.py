"""
Tessellate IFC geometry for HELIOS++: Wavefront OBJ/MTL plus YAML sidecar.

**Merged mode** (``helios_one_obj_per_instance=False``): one ``scene.obj`` with
per-instance ``usemtl`` groups (HELIOS often reports a single ``hitObjectId``).

**Split mode** (default in :func:`pc2beam.helios_pipeline.run_ifc_helios_pipeline`):
one minimal OBJ per meshed IFC instance and a multi-``<part>`` scene; LAS
``hitObjectId`` is typically the 0-based part index—map to ``instance_id`` via
``meshed_instance_ids_in_part_order`` or :func:`instance_id_for_las_hit_object_id`.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import ifcopenshell.geom as geom
import numpy as np
from omegaconf import DictConfig, OmegaConf

from pc2beam.ifc_io import (
    BeamRecord,
    beam_material_name,
    beam_profile_and_area,
    beam_sidecar_fields,
    iter_beam_records,
    iter_eligible_scene_products,
    non_beam_sidecar_fields,
    open_ifc,
    sanitize_mtl_key,
)


def _kd_rgb_for_label(label: str) -> Tuple[float, float, float]:
    """Stable pseudo-random diffuse color in (0.15, 0.85) for an MTL name."""
    h = hashlib.sha256(label.encode("utf-8")).digest()
    r = 0.15 + (h[0] / 255.0) * 0.7
    g = 0.15 + (h[1] / 255.0) * 0.7
    b = 0.15 + (h[2] / 255.0) * 0.7
    return (r, g, b)


def _iter_triangles(faces: List[int], num_vertices: int) -> List[Tuple[int, int, int]]:
    """
    Expand ifcopenshell face buffer into triangle index triples (0-based).

    Newer IfcOpenShell builds use a flat ``i,j,k`` list; older builds prefix each polygon
    with its vertex count.
    """
    if not faces:
        return []
    if len(faces) % 3 == 0 and max(faces) < num_vertices:
        return [(faces[i], faces[i + 1], faces[i + 2]) for i in range(0, len(faces), 3)]

    tris: List[Tuple[int, int, int]] = []
    i = 0
    n = len(faces)
    while i < n:
        cnt = faces[i]
        if cnt < 3 or i + cnt >= n:
            break
        idx = faces[i + 1 : i + 1 + cnt]
        if cnt == 3:
            tris.append((idx[0], idx[1], idx[2]))
        else:
            for k in range(1, cnt - 1):
                tris.append((idx[0], idx[k], idx[k + 1]))
        i += 1 + cnt
    return tris


def tessellate_product(settings: geom.settings, product) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Return (vertices, triangle_indices) for one IfcProduct, or None if no mesh.

    vertices: (N, 3) float64; faces: (M, 3) int64
    """
    try:
        shape = geom.create_shape(settings, product)
    except Exception:
        return None
    g = shape.geometry
    verts_flat = np.array(g.verts, dtype=np.float64)
    if verts_flat.size == 0:
        return None
    v = verts_flat.reshape(-1, 3)
    tris = _iter_triangles(list(g.faces), len(v))
    if not tris:
        return None
    f = np.asarray(tris, dtype=np.int64)
    return v, f


def tessellate_beam(settings: geom.settings, beam) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Backward-compatible alias for :func:`tessellate_product`."""
    return tessellate_product(settings, beam)


def _instance_mtl_width(n: int) -> int:
    return max(5, len(str(max(n, 1))))


def _geom_settings(
    deflection_tolerance: Optional[float],
    angular_tolerance: Optional[float],
) -> geom.settings:
    settings = geom.settings()
    settings.set("use-world-coords", True)
    if deflection_tolerance is not None:
        settings.set("mesher-linear-deflection", float(deflection_tolerance))
    if angular_tolerance is not None:
        settings.set("mesher-angular-deflection", math.radians(float(angular_tolerance)))
    return settings


def _write_minimal_instance_obj(
    path: Union[str, Path],
    v_arr: np.ndarray,
    f_arr: np.ndarray,
    *,
    source_name: str = "pc2beam",
) -> None:
    """Write a standalone Wavefront OBJ (vertices + triangular faces only) for one instance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {source_name} instance mesh\n")
        for x, y, z in v_arr:
            fh.write(f"v {float(x):.6f} {float(y):.6f} {float(z):.6f}\n")
        for a, b, c in f_arr:
            fh.write(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")


def export_ifc_scene_obj_mtl(
    ifc_path: Union[str, Path],
    obj_path: Union[str, Path],
    mtl_path: Union[str, Path],
    *,
    include_all_geometries: bool = True,
    helios_one_obj_per_instance: bool = False,
    deflection_tolerance: Optional[float] = None,
    angular_tolerance: Optional[float] = None,
) -> DictConfig:
    """
    Tessellate IFC products for HELIOS++.

    If ``helios_one_obj_per_instance`` is False (default for this parameter: see pipeline),
    writes one merged ``scene.obj`` / ``scene.mtl`` with one material/group per instance.

    If True, writes **one OBJ file per meshed IFC instance** under ``<obj_dir>/instances/``
    (``obj_dir`` is ``obj_path.parent``). Unmeshed instances get no file and no scene part.
    HELIOS++ typically sets LAS ``hitObjectId`` to the **0-based index** among scene
    ``<part>`` entries; use sidecar ``meshed_instance_ids_in_part_order`` to map that
    index to ``instance_id`` / IFC metadata.

    Parameters
    ----------
    include_all_geometries
        If True (default), export all eligible ``IfcProduct`` (see
        :func:`pc2beam.ifc_io.iter_eligible_scene_products`). If False, only ``IfcBeam``.
    helios_one_obj_per_instance
        If True, split meshes into separate OBJs and populate ``helios_scene_parts`` in
        the sidecar (merged OBJ/MTL are not written).
    """
    ifc_path = Path(ifc_path)
    obj_path = Path(obj_path)
    mtl_path = Path(mtl_path)

    f = open_ifc(ifc_path)
    eligible = iter_eligible_scene_products(f)
    if include_all_geometries:
        products = eligible
    else:
        products = [p for p in eligible if p.is_a("IfcBeam")]

    beam_by_gid = {r.global_id: r for r in iter_beam_records(ifc_path)}
    settings = _geom_settings(deflection_tolerance, angular_tolerance)

    obj_path.parent.mkdir(parents=True, exist_ok=True)

    all_vertices: List[List[float]] = []
    instance_rows: List[Dict[str, Any]] = []
    mtl_names_in_order: List[str] = []
    face_index_global = 0
    v_base = 0
    width = _instance_mtl_width(len(products))

    mtl_name_only = mtl_path.name
    group_tris: List[Tuple[str, List[Tuple[int, int, int]]]] = []

    instances_dir = obj_path.parent / "instances"
    helios_scene_parts: List[str] = []
    meshed_instance_ids_in_part_order: List[int] = []
    instance_hit_mapping: List[Dict[str, Any]] = []

    data_root_prefix = "data/sceneparts/pc2beam/instances"

    for instance_id, product in enumerate(products, start=1):
        mtl_name = f"i{instance_id:0{width}d}"
        mtl_names_in_order.append(mtl_name)
        gid = getattr(product, "GlobalId", None) or ""
        express_id = product.id()
        ifc_class = product.is_a()
        is_beam = product.is_a("IfcBeam")

        row: Dict[str, Any] = {
            "instance_id": instance_id,
            "mtl_name": mtl_name,
            "express_id": int(express_id),
            "global_id": gid,
            "ifc_class": ifc_class,
            "is_beam": is_beam,
        }
        if is_beam:
            rec = beam_by_gid.get(gid)
            if rec is None:
                pn, ar = beam_profile_and_area(product)
                mn = beam_material_name(product)
                rec = BeamRecord(gid, pn, mn, ar, sanitize_mtl_key(pn, mn), {})
            row.update(beam_sidecar_fields(product, rec))
        else:
            row.update(non_beam_sidecar_fields(product))

        mesh = tessellate_product(settings, product)
        if mesh is None:
            row["meshed"] = False
            row["first_face_index"] = None
            row["face_count"] = 0
            row["vertex_offset"] = len(all_vertices)
            row["vertex_count"] = 0
            row["helios_part_index"] = None
            row["instance_obj_relpath"] = None
            instance_rows.append(row)
            if not helios_one_obj_per_instance:
                group_tris.append((mtl_name, []))
            continue

        v_arr, f_arr = mesh
        n_verts = len(v_arr)
        n_faces = len(f_arr)
        v_off = len(all_vertices)
        first_face = face_index_global

        row["meshed"] = True
        row["face_count"] = int(n_faces)
        row["vertex_count"] = int(n_verts)
        if helios_one_obj_per_instance:
            row["first_face_index"] = 0
            row["vertex_offset"] = 0
        else:
            row["first_face_index"] = int(first_face)
            row["vertex_offset"] = int(v_off)

        if helios_one_obj_per_instance:
            part_idx = len(meshed_instance_ids_in_part_order)
            row["helios_part_index"] = int(part_idx)
            inst_name = f"i{instance_id:0{width}d}.obj"
            rel = f"{data_root_prefix}/{inst_name}"
            row["instance_obj_relpath"] = rel
            meshed_instance_ids_in_part_order.append(int(instance_id))
            helios_scene_parts.append(rel)
            _write_minimal_instance_obj(
                instances_dir / inst_name,
                v_arr,
                f_arr,
                source_name=ifc_path.name,
            )
            map_row: Dict[str, Any] = {
                "helios_part_index": int(part_idx),
                "instance_id": int(instance_id),
                "global_id": gid,
                "ifc_class": ifc_class,
                "is_beam": is_beam,
            }
            for k, v in row.items():
                if k in map_row or k in (
                    "meshed",
                    "first_face_index",
                    "face_count",
                    "vertex_offset",
                    "vertex_count",
                    "helios_part_index",
                    "instance_obj_relpath",
                    "mtl_name",
                ):
                    continue
                if v is not None and k != "psets":
                    map_row[k] = v
            if row.get("psets"):
                map_row["psets"] = row["psets"]
            instance_hit_mapping.append(map_row)
        else:
            row["helios_part_index"] = None
            row["instance_obj_relpath"] = None

        instance_rows.append(row)

        if not helios_one_obj_per_instance:
            for row_v in v_arr:
                all_vertices.append([float(row_v[0]), float(row_v[1]), float(row_v[2])])

            tris: List[Tuple[int, int, int]] = []
            for a, b, c in f_arr:
                tris.append(
                    (v_base + int(a) + 1, v_base + int(b) + 1, v_base + int(c) + 1)
                )
            group_tris.append((mtl_name, tris))
            v_base += n_verts
            face_index_global += n_faces
        else:
            face_index_global += n_faces

    if not helios_one_obj_per_instance:
        with open(obj_path, "w", encoding="utf-8") as obj_f:
            obj_f.write(f"# pc2beam scene export from {ifc_path.name}\n")
            obj_f.write(f"# include_all_geometries={include_all_geometries}\n")
            obj_f.write(f"mtllib {mtl_name_only}\n")
            for x, y, z in all_vertices:
                obj_f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            for mtl_name, tris in group_tris:
                obj_f.write(f"usemtl {mtl_name}\n")
                obj_f.write(f"g {mtl_name}\n")
                for a, b, c in tris:
                    obj_f.write(f"f {a} {b} {c}\n")

        with open(mtl_path, "w", encoding="utf-8") as mtl_f:
            mtl_f.write(
                "# pc2beam per-instance materials (diffuse only; HELIOS++ may map to LAS channels)\n"
            )
            for mtl_name in mtl_names_in_order:
                r, g, b = _kd_rgb_for_label(mtl_name)
                mtl_f.write(f"newmtl {mtl_name}\n")
                mtl_f.write(f"Kd {r:.4f} {g:.4f} {b:.4f}\n")
                mtl_f.write("Ka 0.2 0.2 0.2\n")
                mtl_f.write("Ks 0.0 0.0 0.0\n")
                mtl_f.write("d 1.0\n")
                mtl_f.write("illum 1\n\n")

    beams_subset = [r for r in instance_rows if r.get("is_beam")]
    beam_hit_mapping = [m for m in instance_hit_mapping if m.get("is_beam")]

    sidecar_dict: Dict[str, Any] = {
        "source_ifc": str(ifc_path.resolve()),
        "include_all_geometries": include_all_geometries,
        "helios_one_obj_per_instance": helios_one_obj_per_instance,
        "instance_count": len(instance_rows),
        "meshed_instance_count": sum(1 for r in instance_rows if r.get("meshed")),
        "material_names": mtl_names_in_order,
        "instances": instance_rows,
        "beam_count": len(beams_subset),
        "meshed_beam_count": sum(1 for r in beams_subset if r.get("meshed")),
        "beams": beams_subset,
    }

    if helios_one_obj_per_instance:
        sidecar_dict["obj"] = None
        sidecar_dict["mtl"] = None
        sidecar_dict["merged_obj"] = None
        sidecar_dict["merged_mtl"] = None
        sidecar_dict["instances_directory"] = str(instances_dir.resolve())
        sidecar_dict["helios_scene_parts"] = helios_scene_parts
        sidecar_dict["meshed_instance_ids_in_part_order"] = meshed_instance_ids_in_part_order
        sidecar_dict["instance_hit_mapping"] = instance_hit_mapping
        sidecar_dict["beam_instance_hit_mapping"] = beam_hit_mapping
        sidecar_dict["helios_hit_object_id_note"] = (
            "HELIOS++ LAS extra dimension hitObjectId is typically the 0-based index of "
            "the scene <part> that was hit, i.e. helios_part_index. Map to IFC instance_id "
            "via meshed_instance_ids_in_part_order[hitObjectId] or instance_hit_mapping."
        )
    else:
        sidecar_dict["obj"] = str(obj_path.resolve())
        sidecar_dict["mtl"] = str(mtl_path.resolve())
        sidecar_dict["helios_scene_parts"] = [f"data/sceneparts/pc2beam/{obj_path.name}"]
        sidecar_dict["meshed_instance_ids_in_part_order"] = [
            r["instance_id"] for r in instance_rows if r.get("meshed")
        ]
        sidecar_dict["instance_hit_mapping"] = []
        sidecar_dict["beam_instance_hit_mapping"] = []
        sidecar_dict["helios_hit_object_id_note"] = (
            "Single merged OBJ loads as one scene part; hitObjectId may be constant. "
            "Use helios_one_obj_per_instance=True for per-instance LAS ids."
        )

    sidecar = OmegaConf.create(sidecar_dict)
    return sidecar


def instance_id_for_las_hit_object_id(hit_object_id: int, sidecar: DictConfig) -> Optional[int]:
    """
    Map LAS ``hitObjectId`` (0-based scene part index when using split instance OBJs)
    to IFC ``instance_id``.
    """
    order = OmegaConf.select(sidecar, "meshed_instance_ids_in_part_order")
    if order is None:
        return None
    lst = list(order)
    hid = int(hit_object_id)
    if 0 <= hid < len(lst):
        return int(lst[hid])
    return None


def export_beams_obj_mtl(
    ifc_path: Union[str, Path],
    obj_path: Union[str, Path],
    mtl_path: Union[str, Path],
    *,
    deflection_tolerance: Optional[float] = None,
    angular_tolerance: Optional[float] = None,
    helios_one_obj_per_instance: bool = False,
) -> Tuple[List[BeamRecord], DictConfig]:
    """
    Tessellate **IfcBeam** only (same mesh/sidecar layout as full scene export).

    Returns
    -------
    records
        All beam records from the IFC (for callers that need the full beam list).
    sidecar
        Scene sidecar; ``instances`` / ``beams`` contain only beam rows.
    """
    sidecar = export_ifc_scene_obj_mtl(
        ifc_path,
        obj_path,
        mtl_path,
        include_all_geometries=False,
        helios_one_obj_per_instance=helios_one_obj_per_instance,
        deflection_tolerance=deflection_tolerance,
        angular_tolerance=angular_tolerance,
    )
    records_list = iter_beam_records(ifc_path)
    return records_list, sidecar


def save_sidecar(sidecar: DictConfig, yaml_path: Union[str, Path]) -> None:
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(sidecar, yaml_path)


def load_mesh_for_preview(
    obj_path: Union[str, Path],
    max_triangles: int = 50_000,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Lightweight OBJ reader for notebook preview (vertices, triangle indices, per-triangle group id).

    If max_triangles is exceeded, triangles are randomly subsampled (fixed seed).
    """
    obj_path = Path(obj_path)
    vertices: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    group_ids: List[int] = []
    current_group = 0

    with open(obj_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "g" and len(parts) >= 2:
                current_group += 1
            elif parts[0] == "f":
                idxs = []
                for p in parts[1:]:
                    vi = int(p.split("/")[0])
                    if vi < 0:
                        vi = len(vertices) + vi + 1
                    idxs.append(vi - 1)
                if len(idxs) >= 3:
                    for k in range(1, len(idxs) - 1):
                        faces.append((idxs[0], idxs[k], idxs[k + 1]))
                        group_ids.append(current_group)

    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    g = np.asarray(group_ids, dtype=np.int32) if group_ids else None

    if len(f) > max_triangles:
        rng = np.random.default_rng(42)
        sel = rng.choice(len(f), size=max_triangles, replace=False)
        f = f[sel]
        if g is not None:
            g = g[sel]

    return v, f, g
