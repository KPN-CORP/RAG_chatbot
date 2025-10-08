"""Run the repository housekeeping tasks via pre-commit.

This script keeps the convenience of the original clean_project.py command
while delegating the actual work to the configured pre-commit hooks.  Running
this file is therefore equivalent to executing ``pre-commit run --all-files``
followed by the manual Vulture report hook.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "application" / "dead_code_report.txt"


def ensure_pre_commit_available() -> None:
    """Exit early if the pre-commit executable is missing."""

    if shutil.which("pre-commit") is None:
        print(
            "❌ The 'pre-commit' executable is not available. "
            "Install it with 'pip install pre-commit' and try again."
        )
        sys.exit(1)


def format_command(cmd: Sequence[str]) -> str:
    return " ".join(cmd)


def run(cmd: Sequence[str], desc: str) -> bool:
    """Run a shell command and report whether it succeeded."""

    print(f"\n🔹 {desc}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"⚠️ Failed: {format_command(cmd)}")
        return False
    return True


def main() -> int:
    ensure_pre_commit_available()

    commands = [
        (
            ["pre-commit", "run", "--all-files"],
            "Running formatters and linters via pre-commit",
        ),
        (
            ["pre-commit", "run", "vulture-report", "--all-files"],
            f"Refreshing dead code report at {REPORT_PATH.relative_to(PROJECT_ROOT)}",
        ),
    ]

    for cmd, desc in commands:
        if not run(cmd, desc):
            print(
                "\n⚠️ Some steps did not complete successfully. "
                "Address the reported issues and re-run the script."
            )
            return 1

    print(
        "\n✅ Cleanup complete! Check "
        f"'{REPORT_PATH.relative_to(PROJECT_ROOT)}' for possible unused functions/classes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
