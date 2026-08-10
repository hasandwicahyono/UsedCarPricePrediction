# DEPRECATION NOTICE: Prefer factory entrypoints in exp.factories for constructing objects; direct use remains for backward compatibility.
"""
Note: Prefer the factory entrypoints (build_metric/build_model/build_preprocessor/
build_shap_explainer) for constructing objects. Direct class construction remains
supported for backward compatibility.
"""

from .config import FeatureSchema, ExperimentConfig
from .data_io import DataReadConfig, read_csv_folder, coerce_dtypes, basic_clean, CsvFolderSource
from .plot_manager import PlotManager, PointRangePlot
from .preprocess import PreprocessorBuilder
from .factories import build_metric, build_model, build_preprocessor#, build_shap_explainer
from .patterns import RunContext


def __getattr__(name):
    if name == "ExperimentFacade":
        from .facade import ExperimentFacade

        return ExperimentFacade
    if name in {"load_best_model_name", "load_model_artifacts", "make_predictor"}:
        from . import deploy

        return getattr(deploy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
