"""
IFC structural elements: read beam metadata (profile, material, identifiers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import ifcopenshell
import ifcopenshell.util.element as element_util


@dataclass
class BeamRecord:
    """One IfcBeam occurrence with resolved type information."""

    global_id: str
    profile_name: str
    material_name: str
    cross_section_area: Optional[float]
    mtl_key: str
    extra_properties: Dict[str, Any] = field(default_factory=dict)


def _first_profile_from_item(item) -> Optional[Any]:
    """Resolve swept profile from a geometric representation item."""
    if item is None:
        return None
    if item.is_a("IfcExtrudedAreaSolid"):
        return item.SweptArea
    if item.is_a("IfcRevolvedAreaSolid"):
        return item.SweptArea
    if item.is_a("IfcSweptDiskSolid"):
        return None
    if item.is_a("IfcBooleanResult"):
        return _first_profile_from_item(item.FirstOperand) or _first_profile_from_item(
            item.SecondOperand
        )
    if item.is_a("IfcMappedItem"):
        m = item.MappingSource
        if m and m.MappedRepresentation:
            for sub in m.MappedRepresentation.Items or []:
                p = _first_profile_from_item(sub)
                if p is not None:
                    return p
    return None


def _profile_name_from_def(profile) -> str:
    if profile is None:
        return "unknown_profile"
    name = getattr(profile, "ProfileName", None) or getattr(profile, "Name", None)
    if name:
        return str(name)
    return profile.is_a()


def _cross_section_area(profile) -> Optional[float]:
    if profile is None:
        return None
    if hasattr(profile, "ProfileType") and profile.ProfileType == "AREA":
        pass
    for inv in getattr(profile, "HasProperties", None) or []:
        if not inv.is_a("IfcPropertySingleValue"):
            continue
        if inv.Name and str(inv.Name).lower() in ("crosssectionarea", "cross_section_area"):
            v = inv.NominalValue
            if v is None:
                continue
            w = getattr(v, "wrappedValue", None)
            if w is not None:
                try:
                    return float(w)
                except (TypeError, ValueError):
                    continue
    return None


def beam_profile_and_area(beam) -> tuple[str, Optional[float]]:
    """Extract profile label and cross-section area from IfcBeam representation."""
    rep = getattr(beam, "Representation", None)
    if not rep:
        return "unknown_profile", None
    for r in rep.Representations or []:
        for item in r.Items or []:
            prof = _first_profile_from_item(item)
            if prof is not None:
                return _profile_name_from_def(prof), _cross_section_area(prof)
    return "unknown_profile", None


def beam_material_name(beam) -> str:
    """Best-effort material name for an IfcBeam."""
    try:
        mats = element_util.get_materials(beam)
    except Exception:
        mats = []
    if not mats:
        return "unknown_material"
    names = []
    for m in mats:
        if hasattr(m, "Name") and m.Name:
            names.append(str(m.Name))
        elif m.is_a("IfcMaterialLayerSetUsage") and m.ForLayerSet:
            for layer in m.ForLayerSet.MaterialLayers or []:
                if layer.Material and layer.Material.Name:
                    names.append(str(layer.Material.Name))
        elif m.is_a("IfcMaterialProfileSet") and m.MaterialProfiles:
            for mp in m.MaterialProfiles:
                if mp.Material and mp.Material.Name:
                    names.append(str(mp.Material.Name))
    return ", ".join(names) if names else "unknown_material"


def sanitize_mtl_key(profile: str, material: str) -> str:
    """ASCII-safe material name for MTL / OBJ usemtl."""
    raw = f"{profile}_{material}"
    out = []
    for ch in raw:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        elif ch.isspace():
            out.append("_")
    key = "".join(out).strip("_") or "beamtype"
    if len(key) > 60:
        key = key[:60]
    return key


def iter_beam_records(ifc_path: Union[str, Path]) -> List[BeamRecord]:
    """
    Enumerate IfcBeam entities with profile and material metadata.

    Parameters
    ----------
    ifc_path
        Path to an IFC2x3 / IFC4 file.
    """
    path = Path(ifc_path)
    f = ifcopenshell.open(str(path))
    records: List[BeamRecord] = []

    for beam in f.by_type("IfcBeam"):
        gid = beam.GlobalId or ""
        profile_name, area = beam_profile_and_area(beam)
        material_name = beam_material_name(beam)
        mtl_key = sanitize_mtl_key(profile_name, material_name)

        extra: Dict[str, Any] = {}
        if getattr(beam, "Name", None):
            extra["name"] = beam.Name
        if getattr(beam, "Tag", None):
            extra["tag"] = beam.Tag

        records.append(
            BeamRecord(
                global_id=gid,
                profile_name=profile_name,
                material_name=material_name,
                cross_section_area=area,
                mtl_key=mtl_key,
                extra_properties=extra,
            )
        )
    return records


def open_ifc(ifc_path: Union[str, Path]) -> ifcopenshell.file:
    """Open an IFC model (thin wrapper for consistency)."""
    return ifcopenshell.open(str(Path(ifc_path)))


def iter_eligible_scene_products(ifc_file) -> List:
    """
    IfcProduct instances with a representation, excluding openings and spaces.

    Sorted by express id then GlobalId for stable instance_id assignment across runs.
    """
    out: List = []
    for p in ifc_file.by_type("IfcProduct"):
        if p.is_a("IfcOpeningElement") or p.is_a("IfcSpace"):
            continue
        if not getattr(p, "Representation", None):
            continue
        out.append(p)
    out.sort(key=lambda x: (x.id(), getattr(x, "GlobalId", None) or ""))
    return out


def _predefined_type_str(element) -> Optional[str]:
    pt = getattr(element, "PredefinedType", None)
    if pt is None:
        return None
    if hasattr(pt, "name"):
        return str(pt.name)
    s = str(pt)
    return s if s and s != "NOTDEFINED" else None


def type_relation_info(element) -> Dict[str, Any]:
    """IfcRelDefinesByType / PredefinedType fields for sidecar metadata."""
    out: Dict[str, Any] = {}
    ps = _predefined_type_str(element)
    if ps:
        out["predefined_type"] = ps
    for rel in getattr(element, "IsTypedBy", None) or []:
        if not rel.is_a("IfcRelDefinesByType"):
            continue
        t = rel.RelatingType
        if t is None:
            continue
        if getattr(t, "GlobalId", None):
            out["element_type_global_id"] = str(t.GlobalId)
        if getattr(t, "Name", None):
            out["element_type_name"] = str(t.Name)
        tps = _predefined_type_str(t)
        if tps and "predefined_type" not in out:
            out["predefined_type"] = tps
        if t.is_a("IfcBeamType") and getattr(t, "Tag", None):
            out["beam_type_tag"] = str(t.Tag)
        break
    return out


def _flatten_psets(raw: Dict[str, Any], max_keys: int) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    n = 0
    for pset_name, props in raw.items():
        if not isinstance(props, dict):
            continue
        for key, val in props.items():
            if n >= max_keys:
                return flat
            k = f"{pset_name}.{key}"
            if isinstance(val, (str, int, float, bool)) or val is None:
                flat[k] = val
            else:
                flat[k] = str(val)
            n += 1
    return flat


def bounded_element_psets(element, max_keys: int = 40) -> Dict[str, Any]:
    """Flattened pset key/value pairs, capped for sidecar size."""
    try:
        raw = element_util.get_psets(element)
    except Exception:
        return {}
    if not raw:
        return {}
    return _flatten_psets(raw, max_keys)


def beam_sidecar_fields(beam, rec: BeamRecord) -> Dict[str, Any]:
    """Per-beam IFC fields merged into a scene instance record."""
    d: Dict[str, Any] = {
        "profile_name": rec.profile_name,
        "material_name": rec.material_name,
        "cross_section_area": rec.cross_section_area,
        "mtl_key": rec.mtl_key,
    }
    d.update(rec.extra_properties)
    d.update(type_relation_info(beam))
    psets = bounded_element_psets(beam)
    if psets:
        d["psets"] = psets
    return d


def non_beam_sidecar_fields(product) -> Dict[str, Any]:
    """Name, tag, type, and bounded psets for non-beam products."""
    d: Dict[str, Any] = {}
    if getattr(product, "Name", None):
        d["name"] = product.Name
    if getattr(product, "Tag", None):
        d["tag"] = product.Tag
    d.update(type_relation_info(product))
    psets = bounded_element_psets(product)
    if psets:
        d["psets"] = psets
    return d
