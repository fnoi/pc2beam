"""
Entry point for IFC → HELIOS++ laser scanning (function-based, no CLI parsing).
"""

from pathlib import Path

from pc2beam.helios_pipeline import run_ifc_helios_pipeline


def main():
    root = Path(__file__).resolve().parent
    result = run_ifc_helios_pipeline(
        ifc_path=root / "data" / "model_0_z_up.ifc",
        scanners_yaml=root / "config" / "scanners_example.yaml",
        output_root=root / "output" / "helios_pc2beam",
        run_simulation=True,
        include_all_geometries=True,
        helios_one_obj_per_instance=True,
        mesher_linear_deflection=0.02,
    )
    print("run_dir:", result["run_dir"])
    print("survey_xml:", result["survey_xml"])
    print("scene_obj:", result["scene_obj"])
    print("sim_output_dir:", result["sim_output_dir"])
    if result.get("completed_process") is not None:
        print("helios return code:", result["returncode"])


if __name__ == "__main__":
    main()
