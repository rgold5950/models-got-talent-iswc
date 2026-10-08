import os
import sys
from pathlib import Path

def get_project_root():
    """
    Detect project root automatically.

    Priority order:
    1. PROJECT_ROOT environment variable (if set)
    2. project_settings file in current directory or parents (contains export PROJECT_ROOT=...)
    3. Auto-detect from current working directory or file location
    4. Default to current working directory (top level of directory)

    To set a custom path, use:
        export PROJECT_ROOT="/path/to/models_got_talent"
    Or create a project_settings file in the project root with:
        export PROJECT_ROOT="/path/to/models_got_talent"
    """
    # 1. Check environment variable first (highest priority)
    env_path = os.environ.get("PROJECT_ROOT")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return str(path)

    # 2. Check for project_settings file in current directory or parents
    current = Path.cwd()
    for _ in range(5):  # Check up to 5 levels up
        config_file = current / "project_settings"
        if config_file.exists():
            try:
                # Source the shell script to get PROJECT_ROOT
                import subprocess
                result = subprocess.run(
                    ['bash', '-c', f'source {config_file} && echo $PROJECT_ROOT'],
                    capture_output=True, text=True, cwd=str(current)
                )
                if result.returncode == 0 and result.stdout.strip():
                    custom_path = Path(result.stdout.strip())
                    if custom_path.exists():
                        return str(custom_path)
            except Exception:
                pass
        current = current.parent

    # 3. Auto-detect: try to get path from __file__ if available (works in scripts)
    try:
        file_path = Path(__file__).parent.parent
        if (file_path / "train.py").exists():
            return str(file_path)
    except NameError:
        # __file__ not available (e.g., in notebooks or interactive Python)
        pass

    # 4. Auto-detect: check current working directory
    cwd = Path.cwd()
    if (cwd / "train.py").exists():
        return str(cwd)

    # 5. Auto-detect: check parent directories for train.py
    current = Path.cwd()
    for _ in range(5):  # Check up to 5 levels up
        if (current / "train.py").exists():
            return str(current)
        current = current.parent

    # Final fallback: current working directory (top level of directory)
    return str(Path.cwd())

PROJECT_ROOT = get_project_root()

def setup_path():
    """Add project root to sys.path if not already there."""
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    return PROJECT_ROOT
