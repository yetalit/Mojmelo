import platform
from pathlib import Path
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
    var_line = f'export {var_name}="{lib_dir}"\n'

    # Detect shell rc files
    rc_files = [Path.home() / ".bashrc", Path.home() / ".zshrc"]

    for rc in rc_files:
        if rc.exists():
            content = rc.read_text()
            if f'export {var_name}=' not in content:
                with open(rc, "a") as f:
                    f.write(var_line)
        else:
            with open(rc, "w") as f:
                f.write(var_line)

    root_dir = files("_mojmelo")

    build_command = [f"{os.path.dirname(sys.executable)}/mojo", "build", f"{root_dir}/setup.mojo", "-o", f"{root_dir}/setup"]
    subprocess.run(build_command, check=True)

    for i in range(10):
        run_command = [f"{root_dir}/setup"] if i == 0 else [f"{root_dir}/setup", str(i)]
        subprocess.run(run_command, check=True)

    os.remove(f"{root_dir}/setup")
