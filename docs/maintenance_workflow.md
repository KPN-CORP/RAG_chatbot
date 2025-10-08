# Repository Maintenance Workflow

This project uses [pre-commit](https://pre-commit.com/) to run a consistent set of
formatting and quality checks. The tooling mirrors the previous
`application/clean_project.py` helper so you can keep running it manually or let
Git run the same tasks automatically before each commit.

## Should you run `clean_project.py` or `pre-commit`?

Both entry points execute the same underlying workflow. The helper script is a
thin wrapper that ensures the `pre-commit` executable is available and then runs
the configured checks for you. If you prefer, you can call `pre-commit` directly
with the commands shown below or install the Git hook so the checks run
automatically before each commit. Choose whichever flow best fits how you like
to work.

## What the workflow does

When triggered, either through the `clean_project` helper or by Git's
pre-commit hook, the following steps run in order:

1. **Ruff (`ruff --fix`)** – lints Python files and automatically fixes issues
   such as unused imports or variables.
2. **isort (`isort --profile=black`)** – reorders imports using Black's
   formatting rules to stay consistent with the code formatter.
3. **Black** – reformats Python source files to match the project style guide.
4. **Vulture dead code report** – scans the repository for unreachable code and
   writes the findings to `application/dead_code_report.txt` for review.

These commands live in [`.pre-commit-config.yaml`](../.pre-commit-config.yaml),
so updating that single file keeps the scripted and hook-driven workflows in
sync.

## Running the helper script directly

From the project root you can invoke the script with:

```bash
python -m application.clean_project
```

This ensures `pre-commit` is installed and then executes the steps above. The
script mirrors the previous behaviour of `clean_project.py`, including the
summary message pointing to the Vulture report.

### Installation mode

To install (or update) the Git pre-commit hook so the checks run automatically
before each commit:

```bash
python -m application.clean_project --install
```

This only installs the hook; it does not run the checks immediately.

### Explain mode

If you want to see what the workflow will do without modifying any files, use
the explain flag:

```bash
python -m application.clean_project --explain
```

This prints the descriptions shown above and exits.

## Running via `pre-commit`

Once `pre-commit` is installed you can run the same maintenance tasks without
touching the helper script:

```bash
pre-commit run --all-files
pre-commit run vulture-report --all-files  # regenerate the dead code report
```

Running `pre-commit install` (either manually or via the helper's `--install`
mode) ensures the checks run automatically whenever you attempt to create a
commit.

## Requirements

* Python 3
* `pre-commit` Python package (`pip install pre-commit`)

The individual tools (`ruff`, `isort`, `black`, `vulture`) are downloaded and
managed automatically by pre-commit.
