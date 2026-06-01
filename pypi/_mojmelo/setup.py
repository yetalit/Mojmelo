import platform
from pathlib import Path
import site
import subprocess
import sys
from importlib.resources import files
import os


def main():
    system = platform.system()

    if system not in ("Linux", "Darwin"):
        print(f"Error: Unsupported platform -> {system}")
        sys.exit(1)

    lib_dir = Path(sys.prefix) / "lib"
    var_name = "MODULAR_MOJO_MAX_IMPORT_PATH"

    # Find sitecustomize.py
    for p in site.getsitepackages():
        sc = Path(p) / "sitecustomize.py"
        if sc.exists():
            break
    else:
        # Create it in the first site-packages directory
        sc = Path(site.getsitepackages()[0]) / "sitecustomize.py"
        sc.touch()

    code = f"import os\nos.environ['{var_name}'] = '{lib_dir}'\n"

    text = sc.read_text(encoding="utf-8")
    if code not in text:
        with open(sc, "a", encoding="utf-8") as f:
            f.write("\n" + code)

    root_dir = files("_mojmelo")

    build_command = [f"{os.path.dirname(sys.executable)}/mojo", "build", f"{root_dir}/setup.mojo", "-o", f"{root_dir}/setup"]
    subprocess.run(build_command, cwd=root_dir, check=True)

    for i in range(10):
        run_command = [f"{root_dir}/setup"] if i == 0 else [f"{root_dir}/setup", str(i)]
        subprocess.run(run_command, cwd=root_dir, check=True)

    os.remove(f"{root_dir}/setup")
