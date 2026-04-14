"""
Compatibility entry point.

Canonical HELIOS generation entry script:
`tools/helios_generation/run_helios_simulation.py`
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = (
        Path(__file__).resolve().parent
        / "tools"
        / "helios_generation"
        / "run_helios_simulation.py"
    )
    runpy.run_path(str(target), run_name="__main__")
