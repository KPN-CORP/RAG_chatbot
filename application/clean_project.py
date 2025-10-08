"""Run or inspect the repository housekeeping workflow.

The maintenance tasks live in the project's ``.pre-commit-config.yaml`` so they
can be triggered either as a Git hook or on demand from the command line.  This
helper keeps the convenience of the original ``clean_project.py`` command while
delegating the actual work to the configured hooks when available.  If the
``pre-commit`` executable is missing, the script falls back to calling Ruff,
isort, Black, and Vulture directly so you can still tidy the project with a
single command.

To execute the full cleanup from the project root::

    python -m application.clean_project

Other handy entry points::

    python -m application.clean_project --install   # install the Git hook
    python -m application.clean_project --explain   # show what the workflow does
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPLICATION_DIR = PROJECT_ROOT / "application"
REPORT_PATH = APPLICATION_DIR / "dead_code_report.txt"


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


def pre_commit_env() -> Mapping[str, str]:
    """Return the environment used to invoke ``pre-commit`` commands.

    The pre-commit framework creates its own virtual environments on the fly.
    On some Windows installations pip looks up system-wide configuration
    directories via the Windows registry, which may be unavailable (for
    example, in corporate images with locked down registry keys).  The lookup
    failure raises ``FileNotFoundError`` and prevents the hook environments
    from installing.  ``platformdirs`` – the library responsible for that
    lookup – honours the ``PLATFORMDIRS_FORCE_POSIX`` variable to skip the
    Windows-specific registry access and fall back to a cross-platform
    implementation.  Setting it keeps the workflow functional while leaving
    non-Windows platforms unaffected.
    """
    env = os.environ.copy()
    if sys.platform.startswith("win"):
        env.setdefault("PLATFORMDIRS_FORCE_POSIX", "1")
        env.setdefault("PIP_CONFIG_FILE", "NUL")  # ← stop pip from scanning global config
        env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    return env


def run_pre_commit_workflow() -> bool:
    """Execute the maintenance workflow through ``pre-commit``."""

    commands: Sequence[tuple[Sequence[str], str]] = (
        (
            ("pre-commit", "run", "--all-files"),
            "Running formatters and linters on application/ via pre-commit",
        ),
        (
            (
                "pre-commit",
                "run",
                "--hook-stage",
                "manual",
                "vulture-report",
                "--all-files",
            ),
            (
                "Refreshing application/ dead code report at "
                f"{REPORT_PATH.relative_to(PROJECT_ROOT)}"
            ),
        ),
    )

    for cmd, desc in commands:
        if not run(cmd, desc, env=pre_commit_env()):
            return False
    return True


def write_dead_code_report(output: str) -> None:
    REPORT_PATH.write_text(output, encoding="utf-8")
    print(f"   Report saved to {REPORT_PATH.relative_to(PROJECT_ROOT)}")


def ensure_modules_available(modules: Sequence[str]) -> bool:
    """Verify that the requested Python modules are importable."""

    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        print("❌ Missing required Python packages: " + ", ".join(sorted(missing)))
        print(
            "   Install them with `pip install "
            + " ".join(sorted(missing))
            + "` and re-run the script."
        )
        return False
    return True


def run_direct_workflow() -> bool:
    """Execute the maintenance workflow with direct tool invocations."""

    python = sys.executable

    direct_commands: Sequence[tuple[Sequence[str], str, Sequence[str]]] = (
        (
            (python, "-m", "ruff", "check", "--fix", str(APPLICATION_DIR)),
            "Running Ruff lint + fix on application/",
            ("ruff",),
        ),
        (
            (python, "-m", "isort", "--profile=black", str(APPLICATION_DIR)),
            "Sorting imports with isort (Black profile) on application/",
            ("isort",),
        ),
        (
            (python, "-m", "black", "-l", "100", str(APPLICATION_DIR)),
            "Formatting Python files in application/ with Black",
            ("black",),
        ),
    )

    for cmd, desc, modules in direct_commands:
        if not ensure_modules_available(modules):
            return False
        if not run(cmd, desc):
            return False

    print("\n🔹 Refreshing application/ dead code report using Vulture (direct invocation)")
    if not ensure_modules_available(("vulture",)):
        return False
    try:
        result = subprocess.run(
            (python, "-m", "vulture", str(APPLICATION_DIR)),
            check=True,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        print("⚠️ Failed: python -m vulture application")
        if exc.stdout:
            print(exc.stdout)
        if exc.stderr:
            print(exc.stderr)
        return False

    print(result.stdout, end="")
    write_dead_code_report(result.stdout)
    return True


def run(cmd: Sequence[str], desc: str, *, env: Mapping[str, str] | None = None) -> bool:
    """Run a shell command and report whether it succeeded."""

    print(f"\n🔹 {desc}")
    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)
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
            "Install or update the repository's Git pre-commit hook so the "
            "same checks run automatically before each commit."
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
                "Run Ruff with autofix enabled to tidy imports and flag lint issues in application/.",
                "Reorder imports via isort using Black's profile for compatibility in application/.",
                "Format Python files within application/ using Black.",
                "Generate an updated dead-code report for application/ with Vulture at "
                f"{REPORT_PATH.relative_to(PROJECT_ROOT)}.",
            )
        )
        return 0

    if args.install:
        ensure_pre_commit_available()
        if run(
            ["pre-commit", "install"],
            "Installing the local pre-commit hook",
            env=pre_commit_env(),
        ):
            print(
                "\n✅ Hook installed. Future commits will run the same checks "
                "automatically before Git creates a commit."
            )
            print(
                "   (The command above writes .git/hooks/pre-commit so that "
                "pre-commit can block commits when the cleanup fails.)"
            )
            return 0
        return 1

    use_pre_commit = shutil.which("pre-commit") is not None

    if use_pre_commit:
        workflow_ok = run_pre_commit_workflow()
    else:
        print("ℹ️ 'pre-commit' is not installed. Running Ruff, isort, Black, and Vulture directly.")
        workflow_ok = run_direct_workflow()

    if not workflow_ok:
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
