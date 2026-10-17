"""Run the whole analysis from the raw data to every table and figure.

Usage (from the package root):
    python code/run_all.py
"""
import platform
import shutil
import sys
import time
from importlib.metadata import version

import config as C
import s01_data_description
import s02_price_models
import s03_mining_simulation
import s04_revenue_cost_profit
import s05_regional_map
import s06_latex_tables

PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "tensorflow", "matplotlib", "geopandas", "openpyxl"]


def main() -> None:
    if C.OUT_DIR.exists():
        shutil.rmtree(C.OUT_DIR)   # start from an empty output folder every time
    C.OUT_DIR.mkdir(parents=True)
    start = time.time()
    for step in [s01_data_description, s02_price_models, s03_mining_simulation,
                 s04_revenue_cost_profit, s05_regional_map, s06_latex_tables]:
        step.main()
    lines = [f"python {sys.version.split()[0]}", f"platform {platform.platform()}"]
    for p in PACKAGES:
        for name in [p, f"{p}-cpu"]:   # TensorFlow may be installed as tensorflow-cpu
            try:
                lines.append(f"{p} {version(name)}")
                break
            except Exception:
                continue
        else:
            lines.append(f"{p} not installed")
    lines.append(f"run time {time.time() - start:.0f} s")
    (C.OUT_DIR / "environment_used.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
