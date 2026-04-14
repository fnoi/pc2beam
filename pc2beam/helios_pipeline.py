"""
IFC tessellation → HELIOS++ asset bundle → optional ``helios`` run.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from omegaconf import DictConfig, OmegaConf

from pc2beam.helios_runner import resolve_helios_data_root, run_helios
from pc2beam.helios_survey import load_scanners_config, write_scene_xml, write_survey_xml
from pc2beam.ifc_mesh_export import export_ifc_scene_obj_mtl, save_sidecar


def run_ifc_helios_pipeline(
    ifc_path: Union[str, Path] = "data/model_0_z_up.ifc",
    scanners_yaml: Union[str, Path] = "config/scanners_example.yaml",
    output_root: Union[str, Path] = "output/helios_pc2beam",
    *,
    run_id: Optional[str] = None,
    helios_data_path: Optional[Union[str, Path]] = None,
    run_simulation: bool = True,
    include_all_geometries: bool = True,
    helios_one_obj_per_instance: bool = True,
    mesher_linear_deflection: float = 0.02,
    mesher_angular_deflection_deg: Optional[float] = None,
    helios_extra_args: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Export IFC geometry to OBJ/MTL, write HELIOS scene + survey XML, optionally run the simulator.

    Returns
    -------
    dict
        Paths and metadata: ``run_dir``, ``survey_xml``, ``scene_obj``, ``scene_mtl``,
        ``scene_sidecar_yaml``, ``beams_obj`` (merged mesh path, or ``None`` when
        ``helios_one_obj_per_instance`` is True), ``instances_dir`` when split,
        ``sidecar_yaml`` (alias of ``scene_sidecar_yaml``), ``sim_output_dir``,
        ``helios_data`` (``None`` when ``run_simulation`` is ``False``),
        ``completed_process`` / ``returncode`` when the simulator runs.
    """
    ifc_path = Path(ifc_path)
    output_root = Path(output_root)
    run_id = run_id or uuid.uuid4().hex[:12]
    run_dir = (output_root / run_id).resolve()
    data_dir = run_dir / "data"
    sceneparts = data_dir / "sceneparts" / "pc2beam"
    scenes = data_dir / "scenes"
    surveys = run_dir / "surveys"
    sim_out = run_dir / "sim_output"

    sceneparts.mkdir(parents=True, exist_ok=True)
    scenes.mkdir(parents=True, exist_ok=True)
    surveys.mkdir(parents=True, exist_ok=True)

    obj_path = sceneparts / "scene.obj"
    mtl_path = sceneparts / "scene.mtl"
    sidecar_path = sceneparts / "scene_sidecar.yaml"

    sidecar = export_ifc_scene_obj_mtl(
        ifc_path,
        obj_path,
        mtl_path,
        include_all_geometries=include_all_geometries,
        helios_one_obj_per_instance=helios_one_obj_per_instance,
        deflection_tolerance=mesher_linear_deflection,
        angular_tolerance=mesher_angular_deflection_deg,
    )
    save_sidecar(sidecar, sidecar_path)

    scene_xml = scenes / "pc2beam_scene.xml"
    if helios_one_obj_per_instance:
        parts = OmegaConf.select(sidecar, "helios_scene_parts") or []
        write_scene_xml(scene_xml, part_obj_relpaths=list(parts))
    else:
        write_scene_xml(
            scene_xml,
            scenepart_obj_relpath="data/sceneparts/pc2beam/scene.obj",
        )

    scan_cfg: DictConfig = load_scanners_config(scanners_yaml)
    survey_xml = surveys / "pc2beam_survey.xml"
    write_survey_xml(survey_xml, scan_cfg)

    result: Dict[str, Any] = {
        "run_dir": run_dir,
        "run_id": run_id,
        "ifc_path": ifc_path.resolve(),
        "scanners_yaml": Path(scanners_yaml).resolve(),
        "include_all_geometries": include_all_geometries,
        "helios_one_obj_per_instance": helios_one_obj_per_instance,
        "scene_obj": None if helios_one_obj_per_instance else obj_path.resolve(),
        "scene_mtl": None if helios_one_obj_per_instance else mtl_path.resolve(),
        "scene_sidecar_yaml": sidecar_path.resolve(),
        "instances_dir": (sceneparts / "instances").resolve()
        if helios_one_obj_per_instance
        else None,
        "beams_obj": None if helios_one_obj_per_instance else obj_path.resolve(),
        "beams_mtl": None if helios_one_obj_per_instance else mtl_path.resolve(),
        "sidecar_yaml": sidecar_path.resolve(),
        "scene_xml": scene_xml.resolve(),
        "survey_xml": survey_xml.resolve(),
        "sim_output_dir": sim_out.resolve(),
        "helios_data": None,
        "sidecar": sidecar,
        "scan_config": scan_cfg,
    }

    if run_simulation:
        helios_data = resolve_helios_data_root(helios_data_path)
        result["helios_data"] = helios_data
        extra = list(helios_extra_args) if helios_extra_args else None
        cp = run_helios(survey_xml, helios_data, run_dir, sim_out, extra_args=extra)
        result["completed_process"] = cp
        result["returncode"] = cp.returncode
    else:
        result["completed_process"] = None
        result["returncode"] = None

    return result
