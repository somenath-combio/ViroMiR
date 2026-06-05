# Release Checklist

Before cutting a new GitHub release or uploading to PyPI, verify the following:

- [ ] All tests pass (`pytest tests/`)
- [ ] `pip install -e .` works in a clean virtual environment
- [ ] `viromir-scan --help` renders correctly without stack traces
- [ ] The bundled XGBoost `.pkl` model file is present in `viromir/models/`
- [ ] Version numbers are updated in `setup.py` (or `pyproject.toml`) and `__init__.py`
- [ ] `CHANGELOG.md` is updated with the current date and release notes
- [ ] `README.md` examples still exactly match the CLI parameter names
