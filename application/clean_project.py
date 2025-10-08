"""Run or inspect the repository housekeeping workflow.

The maintenance tasks live in the project's ``.pre-commit-config.yaml`` so they
can be triggered either as a Git hook or on demand from the command line.  This
helper keeps the convenience of the original ``clean_project.py`` command while
delegating the actual work to the configured hooks.

To execute the full cleanup from the project root::

    python -m application.clean_project

Other handy entry points::

    python -m application.clean_project --install   # install the Git hook
    python -m application.clean_project --explain   # show what the workflow does
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "application" / "dead_code_report.txt"

COMMANDS: Sequence[tuple[list[str], str]] = (
    (
        ["pre-commit", "run", "--all-files"],
        "Running formatters and linters via pre-commit",
    ),
    (
        ["pre-commit", "run", "vulture-report", "--all-files"],
        (
            "Refreshing dead code report at "
            f"{REPORT_PATH.relative_to(PROJECT_ROOT)}"
        ),
    ),
)


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


def describe_workflow(lines: Iterable[str]) -> None:
    """Print a short explanation of the configured maintenance steps."""

    print("\nWhat the maintenance workflow does:\n")
    for line in lines:
        print(f"  • {line}")
    print(
        "\nRun `python -m application.clean_project` from the project root "
        "to execute these tasks on demand."
    )
    print(
        "Use `python -m application.clean_project --install` if you want the "
        "same checks to run automatically before each commit."
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the repository housekeeping tasks, install the Git hook, or "
            "describe the maintenance workflow."
        ),
        epilog=(
            "Examples:\n"
            "  python -m application.clean_project\n"
            "  python -m application.clean_project --install\n"
            "  python -m application.clean_project --explain"
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--install",
        action="store_true",
        help=(
            "Install or update the repository's pre-commit hook without "
            "running any checks."
        ),
    )
    group.add_argument(
        "--explain",
        action="store_true",
        help="Describe the maintenance workflow and exit without running it.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.explain:
        describe_workflow(
            (
                "Run Ruff with autofix enabled to tidy imports and flag lint issues.",
                "Reorder imports via isort using Black's profile for compatibility.",
                "Format Python files with Black.",
                "Generate an updated dead-code report with Vulture at "
                f"{REPORT_PATH.relative_to(PROJECT_ROOT)}.",
            )
        )
        return 0

    ensure_pre_commit_available()

    if args.install:
        if run(["pre-commit", "install"], "Installing the local pre-commit hook"):
            print(
                "\n✅ Hook installed. Future commits will run the same checks "
                "automatically."
            )
            return 0
        return 1

    for cmd, desc in COMMANDS:
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
