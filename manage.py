"""
Project manage tasks for CLI automation.
"""

import subprocess  # nosec B404 - legitimate use of subprocess for dev tools


def format() -> None:
    """Run Ruff linting and formatting."""
    commands = [
        ["uv", "run", "ruff", "format", "."],
        ["uv", "run", "ruff", "check", ".", "--fix", "--unsafe-fixes"],
        ["echo", "\n🟢 Linting → ✅ Code clean\n"],
    ]
    for cmd in commands:
        subprocess.run(cmd, check=True)  # nosec B603,B607 - trusted dev commands


def clean() -> None:
    """Clean up the project."""
    commands = [
        # Basic clean up
        ["find", ".", "-type", "d", "-name", "__pycache__", "-exec", "rm", "-rf", "{}", "+"],
        ["find", ".", "-type", "f", "-name", "*.pyc", "-delete"],
        # Extended clean up
        ## Python cache files
        # ["find", ".", "-type", "d", "-name", "__pycache__", "-exec", "rm", "-rf", "{}", "+"],
        # ["find", ".", "-type", "f", "-name", "*.pyc", "-delete"],
        # ["find", ".", "-type", "f", "-name", "*.pyo", "-delete"],
        # ["find", ".", "-type", "f", "-name", "*.pyd", "-delete"],
        # ## Test and coverage artifacts
        # ["rm", "-rf", ".pytest_cache"],
        # ["rm", "-rf", ".coverage"],
        # ["rm", "-rf", "htmlcov"],
        # ## Tool caches
        # ["rm", "-rf", ".ruff_cache"],
        # ["rm", "-rf", ".mypy_cache"],
        # ## Build artifacts
        # ["rm", "-rf", "dist"],
        # ["rm", "-rf", "build"],
        # ["find", ".", "-type", "d", "-name", "*.egg-info", "-exec", "rm", "-rf", "{}", "+"],
        ["echo", "\n🧹 CleanUp → ✅ Code clean\n"],
    ]
    for cmd in commands:
        subprocess.run(cmd, check=True)  # nosec B603,B607 - trusted dev commands
