import platform
import sys
from importlib.resources import files
from mojo._package_root import get_package_root
import os
import subprocess


def main():
    system = platform.system()

    if system not in ("Linux", "Darwin"):
        print(f"Error: Unsupported platform -> {system}")
        sys.exit(1)

    root_dir = files("_mojmelo")
    mojo_lib_dir = get_package_root() / "lib" / "mojo"
    var_name = "MODULAR_MOJO_MAX_IMPORT_PATH"

    with open(root_dir.parent / "mojmelo_import.pth", "w") as f:
        f.write(f"import os; os.environ['{var_name}']=','.join(dict.fromkeys([*filter(None, os.environ.get('{var_name}','').split(',')), '{mojo_lib_dir}', '{root_dir}']));")

    build_command = [os.path.join(os.path.dirname(sys.executable), "mojo"), "build", os.path.join(root_dir, "setup.mojo"), "-o", os.path.join(root_dir, "setup")]
    subprocess.run(build_command, cwd=root_dir, check=True)

    setup_path = os.path.join(root_dir, "setup")
    for i in range(10):
        run_command = [setup_path] if i == 0 else [setup_path, str(i)]
        subprocess.run(run_command, cwd=root_dir, check=True)

    os.remove(setup_path)
