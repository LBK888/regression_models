"""V4 small-data regression benchmark.

The V4 package is intentionally independent from the V3 GUI module.  V3 remains
available as a compatibility reference while the new, testable services are
introduced incrementally.
"""

from .domain import (
    EvaluationConfig,
    EvaluationStrategy,
    EvidenceStatus,
    ExperimentCatalog,
    ExperimentRecord,
    FeatureStructure,
    ProgressState,
    TargetTask,
)
from .metrics import RegressionMetrics, calculate_regression_metrics

__all__ = [
    "EvaluationConfig",
    "EvaluationStrategy",
    "EvidenceStatus",
    "ExperimentCatalog",
    "ExperimentRecord",
    "FeatureStructure",
    "ProgressState",
    "RegressionMetrics",
    "TargetTask",
    "calculate_regression_metrics",
]

__version__ = "4.0.0-dev1"
