import subprocess


def run(cmd, desc):
    print(f"\n🔹 {desc}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print(f"⚠️ Failed: {cmd}")


commands = [
    ("ruff check . --fix", "Fixing unused imports and variables with Ruff"),
    ("isort .", "Sorting imports"),
    ("black .", "Formatting code"),
    ("vulture . > dead_code_report.txt", "Detecting dead code (Vulture)"),
]

for cmd, desc in commands:
    run(cmd, desc)

print(
    "\n✅ Cleanup complete! Check 'dead_code_report.txt' for possible unused functions/classes."
)
