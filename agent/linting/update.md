## Linting cleanup update

- Added missing module/class/method docstrings in `bot.py` and tightened type hints.
- Fixed import ordering and long lines in tests.
- Cleaned dataset generator lint issues (unused import, loop vars, one-line control flow).
- Added per-file ignores for long lines and docstrings in `dataset/generate_dataset.py`.

### Lint checks
- `ruff check .` (PASS)

### Tests executed
- `python -m unittest discover -s tests` (PASS, 85 tests)

