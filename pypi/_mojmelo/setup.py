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

    with open(Path(site.getsitepackages()[0]) / "mojmelo_import.pth", "w") as f:
        file.write(f"import os; os.environ['{var_name}'] = '{lib_dir}';")

    root_dir = files("_mojmelo")

    build_command = [f"{os.path.dirname(sys.executable)}/mojo", "build", f"{root_dir}/setup.mojo", "-o", f"{root_dir}/setup"]
    subprocess.run(build_command, cwd=root_dir, check=True)

    for i in range(10):
        run_command = [f"{root_dir}/setup"] if i == 0 else [f"{root_dir}/setup", str(i)]
        subprocess.run(run_command, cwd=root_dir, check=True)

    os.remove(f"{root_dir}/setup")
