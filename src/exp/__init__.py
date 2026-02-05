# DEPRECATION NOTICE: Prefer factory entrypoints in exp.factories for constructing objects; direct use remains for backward compatibility.
"""
Note: Prefer the factory entrypoints (build_metric/build_model/build_preprocessor/
build_shap_explainer) for constructing objects. Direct class construction remains
supported for backward compatibility.
"""

from .config import FeatureSchema, ExperimentConfig
from .facade import ExperimentFacade
from .data_io import DataReadConfig, read_csv_folder, coerce_dtypes, basic_clean, CsvFolderSource
from .plot_manager import PlotManager, PointRangePlot
from .preprocess import PreprocessorBuilder
from .factories import build_metric, build_model, build_preprocessor#, build_shap_explainer
from .patterns import RunContext
from .deploy import load_best_model_name, load_model_artifacts, make_predictor
