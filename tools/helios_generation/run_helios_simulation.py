"""
Entry point for IFC -> HELIOS++ laser scanning from tools/helios_generation.
"""

from pathlib import Path

from pc2beam.helios_pipeline import run_ifc_helios_pipeline


def main():
    repo_root = Path(__file__).resolve().parents[2]
    result = run_ifc_helios_pipeline(
        ifc_path=repo_root / "data" / "model_0_z_up.ifc",
        scanners_yaml=repo_root
        / "tools"
        / "helios_generation"
        / "config"
        / "scanners_example.yaml",
        output_root=repo_root / "output" / "helios_pc2beam",
        run_simulation=True,
        include_all_geometries=True,
        helios_one_obj_per_instance=True,
        mesher_linear_deflection=0.02,
    )
    print("run_dir:", result["run_dir"])
    print("survey_xml:", result["survey_xml"])
    print("scene_obj:", result["scene_obj"])
    print("ground_truth_yaml:", result["ground_truth_yaml"])
    print("sim_output_dir:", result["sim_output_dir"])
    if result.get("completed_process") is not None:
        print("pc2beam_input_txt:", result["pc2beam_input_txt"])
        print("helios return code:", result["returncode"])


if __name__ == "__main__":
    main()
