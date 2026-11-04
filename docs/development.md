# Development setup

Use Python 3.11 (the original notebook records Python 3.11.15).

## Install

```bash
python -m pip install poetry
poetry env use python3.11
poetry install
poetry run pre-commit install
```

On Windows, use `py -3.11` to locate Python if `python3.11` is unavailable.
Poetry can also use an already-active Python 3.11 environment.

## Check

```bash
poetry run ruff check src tests
poetry run ruff format --check src tests
poetry run pytest -q
poetry run pre-commit run --all-files
```

## Dependency lock

Run `poetry lock` with Poetry 2.x, then commit the generated `poetry.lock`.
The lockfile has not yet been generated or validated in CI.
Do not commit `.venv/` or an Anaconda environment.

## Scope

The initial package covers data conversion and evaluation.
The existing notebooks remain research artifacts and are excluded from Ruff.
Statistical and deep-learning notebook dependencies will be isolated after
the ML pipeline is extracted and its compatibility is verified.
