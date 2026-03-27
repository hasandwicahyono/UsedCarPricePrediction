# IJAI Experiment Framework

This repository contains the code and notebooks used to run machine learning experiments for the IJAI paper. It combines dataset loading, preprocessing, model training, nested cross-validation, hyperparameter tuning, evaluation, plotting, and SHAP-based explainability in a reusable Python experiment framework.

## Repository Layout

- `src/exp/`: core experiment package
- `config/hyperparams.json`: hyperparameter search-space configuration
- `Dataset/data/`: input CSV datasets
- `tests/`: validation tests for core factories and metrics
- `outputs/`: generated experiment tables, figures, and model artifacts
- `model_works_refactored.ipynb`: main end-to-end notebook
- `model_works_colab.ipynb`: Colab-oriented workflow

## Main Workflow

The primary entry point is `model_works_refactored.ipynb`.

The notebook demonstrates the full pipeline:

- dataset loading
- preprocessing
- model selection
- hyperparameter tuning
- evaluation
- figure and artifact generation

The package under `src/exp/` can also be used directly in Python through `ExperimentFacade` and related configuration classes.

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If dependency resolution fails, use:

```bash
python install_requirements.py
```

## Configuration

Experiment behavior is controlled through code-level configuration objects and JSON-based hyperparameter settings.

### Experiment Configuration

`ExperimentConfig` controls runtime behavior such as:

- outer and inner cross-validation folds
- number of tuning trials
- random seed
- target transformation options such as log scaling

### Data Configuration

`DataReadConfig` controls dataset loading, including:

- dataset root directory
- recursive file loading
- filename exclusions
- source-column options

### Hyperparameter Configuration

Hyperparameter search spaces are defined in:

- `config/hyperparams.json`

This file specifies parameter ranges used during model tuning.

## Policies and Conventions

The codebase follows these practical policies:

- prefer factory entry points in `src/exp/factories.py` for constructing metrics, models, and preprocessors
- keep reusable implementation in `src/exp/` and use notebooks as the main user-facing workflow
- place input datasets under `Dataset/data/`
- write generated artifacts under `outputs/`
- preserve direct class imports only where backward compatibility is needed

## Testing

Tests currently live in:

- `tests/test_factories_metrics.py`

Run them with:

```bash
pytest
```

## Outputs

Running experiments produces artifacts such as:

- result tables in `outputs/`
- plots in `outputs/figures/`
- selected parameters in `outputs/best_params/` and `outputs/hyperparameters/`
- model artifacts in `outputs/artifacts/`

## Notes

This repository includes generated outputs and environment-related files in the working tree. For long-term maintenance, treat `outputs/`, `catboost_info/`, `.venv/`, and `__pycache__/` content as generated rather than source code.
