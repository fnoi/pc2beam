"""
Write HELIOS++ scene and survey XML for pc2beam IFC exports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

from omegaconf import DictConfig, OmegaConf

SCENE_ID = "pc2beam_scene"


def write_scene_xml(
    path: Union[str, Path],
    *,
    scenepart_obj_relpath: Optional[str] = None,
    part_obj_relpaths: Optional[Sequence[str]] = None,
) -> None:
    """
    HELIOS++ scene document: one ``<part>`` per OBJ path (relative to run ``data/`` root).

    Use ``part_obj_relpaths`` for multiple instance meshes (distinct LAS ``hitObjectId``
    per part in typical HELIOS++ builds). If omitted, ``scenepart_obj_relpath`` defaults
    to a single merged ``scene.obj``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if part_obj_relpaths is None:
        rel = scenepart_obj_relpath or "data/sceneparts/pc2beam/scene.obj"
        part_obj_relpaths = [rel]
    parts_xml = []
    for rel in part_obj_relpaths:
        parts_xml.append(
            f"""        <part>
            <filter type="objloader">
                <param type="string" key="filepath" value="{rel}" />
            </filter>
        </part>"""
        )
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
    <scene id="{SCENE_ID}" name="{SCENE_ID}">
{chr(10).join(parts_xml)}
    </scene>
</document>
"""
    path.write_text(body, encoding="utf-8")


def _num(x) -> str:
    return f"{float(x):.6g}"


def write_survey_xml(path: Union[str, Path], scan_cfg: DictConfig) -> None:
    """
    TLS-style survey with one ``leg`` per entry in ``scan_cfg.scanners``.

    ``scan_cfg.helios`` must define ``platform_ref``, ``scanner_ref``, default frequencies,
    and default head rotation parameters (overridable per scanner).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h = scan_cfg.helios
    template_id = "pc2beam_scan_profile"
    legs_xml = []
    for s in scan_cfg.scanners:
        x, y, z = (float(s.position[0]), float(s.position[1]), float(s.position[2]))
        h_start = float(s.get("head_rotate_start_deg", h.head_rotate_start_deg))
        h_stop = float(s.get("head_rotate_stop_deg", h.head_rotate_stop_deg))
        h_rate = float(s.get("head_rotate_per_sec_deg", h.head_rotate_per_sec_deg))
        legs_xml.append(
            f"""        <leg>
            <platformSettings x="{_num(x)}" y="{_num(y)}" z="{_num(z)}" />
            <scannerSettings template="{template_id}" headRotatePerSec_deg="{_num(h_rate)}" headRotateStart_deg="{_num(h_start)}" headRotateStop_deg="{_num(h_stop)}" />
        </leg>"""
        )

    platform_ref = str(h.platform_ref)
    scanner_ref = str(h.scanner_ref)
    pulse = _num(h.pulse_freq_hz)
    scanf = _num(h.scan_freq_hz)

    doc = f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
    <scannerSettings id="{template_id}" active="true" pulseFreq_hz="{pulse}" scanFreq_hz="{scanf}"/>
    <survey name="pc2beam_ifc_tls" scene="data/scenes/pc2beam_scene.xml#{SCENE_ID}"
            platform="{platform_ref}" scanner="{scanner_ref}">
{chr(10).join(legs_xml)}
    </survey>
</document>
"""
    path.write_text(doc, encoding="utf-8")


def load_scanners_config(path: Union[str, Path]) -> DictConfig:
    """Load scanner / HELIOS survey defaults from YAML."""
    return OmegaConf.load(str(path))
