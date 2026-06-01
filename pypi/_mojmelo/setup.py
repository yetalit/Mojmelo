import platform
from pathlib import Path
import subprocess
import sys
from importlib.resources import files
import os
import re


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
        content = rc.read_text() if rc.exists() else ""

        pattern = rf"^export\s+{re.escape(var_name)}=.*$"

        if re.search(pattern, content, flags=re.MULTILINE):
            content = re.sub(
                pattern,
                var_line.rstrip(),
                content,
                flags=re.MULTILINE,
            )
        else:
            if content and not content.endswith("\n"):
                content += "\n"
            content += var_line

        rc.write_text(content)

    root_dir = files("_mojmelo")

    build_command = [f"{os.path.dirname(sys.executable)}/mojo", "build", f"{root_dir}/setup.mojo", "-o", f"{root_dir}/setup"]
    subprocess.run(build_command, cwd=root_dir, check=True)

    for i in range(10):
        run_command = [f"{root_dir}/setup"] if i == 0 else [f"{root_dir}/setup", str(i)]
        subprocess.run(run_command, cwd=root_dir, check=True)

    os.remove(f"{root_dir}/setup")
