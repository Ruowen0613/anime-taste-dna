# Machine learning workspace

This package will hold offline-only recommendation work. No training runs in the FastAPI process.

- `configs/`: reproducible preprocessing, training, and evaluation configuration.
- `src/anime_taste_dna/`: importable preprocessing, model, inference, and metric modules.
- `tests/`: focused ML correctness tests.

Raw data belongs in root `data/`, while generated runs/checkpoints belong in ignored artifact directories. The first ML milestone will add a separately installable `pyproject.toml` once concrete dependencies and commands are implemented.
