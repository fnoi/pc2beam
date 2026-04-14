"""
Locate HELIOS++ data assets and invoke the ``helios`` CLI.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Union


def resolve_helios_data_root(explicit: Optional[Union[str, Path]] = None) -> Path:
    """
    Return the directory that contains HELIOS++'s ``data/`` tree (platforms, scanners, demo scenes).

    Resolution order:
    1. ``explicit`` argument
    2. ``HELIOS_DATA_PATH`` environment variable
    3. ``$CONDA_PREFIX/share/helios`` when that directory exists
    """
    if explicit is not None:
        p = Path(explicit)
        if not p.is_dir():
            raise FileNotFoundError(f"HELIOS data path does not exist: {p}")
        return p.resolve()

    env = os.environ.get("HELIOS_DATA_PATH")
    if env:
        p = Path(env)
        if p.is_dir():
            return p.resolve()

    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        p = Path(conda) / "share" / "helios"
        if p.is_dir():
            return p.resolve()

    spec = importlib.util.find_spec("pyhelios")
    if spec is not None and spec.submodule_search_locations:
        root = Path(spec.submodule_search_locations[0])
        if (root / "data" / "platforms.xml").is_file():
            return root.resolve()

    raise FileNotFoundError(
        "Could not find HELIOS++ data directory. Install with "
        "`conda install -c conda-forge helios`, then set HELIOS_DATA_PATH to the "
        "folder containing `data/platforms.xml`, or rely on CONDA_PREFIX/share/helios."
    )


def helios_executable() -> str:
    exe = shutil.which("helios")
    if not exe:
        raise FileNotFoundError(
            "The `helios` program is not on PATH. Install HELIOS++ "
            "(e.g. `conda install -c conda-forge helios`)."
        )
    return exe


def run_helios(
    survey_xml: Union[str, Path],
    helios_data: Union[str, Path],
    run_assets: Union[str, Path],
    output_dir: Union[str, Path],
    extra_args: Optional[Sequence[str]] = None,
) -> subprocess.CompletedProcess:
    """
    Run a HELIOS++ survey with two asset roots: stock data and this run's ``data/`` overlay.

    Parameters
    ----------
    survey_xml
        Path to the generated survey file (may live under ``run_assets/surveys/``).
    helios_data
        Stock HELIOS++ ``share/helios`` (or clone ``data`` root).
    run_assets
        Run directory containing ``data/scenes/`` and ``data/sceneparts/`` for this model.
    output_dir
        Directory for simulator output (LAS, etc.).
    extra_args
        Additional CLI flags (e.g. ``("-j", "4")``).
    """
    survey_xml = Path(survey_xml)
    helios_data = Path(helios_data)
    run_assets = Path(run_assets)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        helios_executable(),
        str(survey_xml),
        "--assets",
        str(helios_data),
        "--assets",
        str(run_assets),
        "--output",
        str(output_dir),
        "--lasOutput",
    ]
    if extra_args:
        cmd.extend(str(a) for a in extra_args)

    return subprocess.run(cmd, check=False)
