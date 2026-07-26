"""Colab-compatible VQR demand-forecasting training workflow.

This module contains the reusable logic used by
``ml/notebooks/train_vqr_colab.ipynb``. Importing it does not start training.
Running it as a script does run the configured workflow.
"""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from ml.algorithms.vqr import VQR, build_vqr
from ml.utils.path import (
    PERISHABLE_GOODS_DATA_PATH,
    PERISHABLE_VQR_MODEL_PATH,
    VQR_EXPERIMENT_RESULTS_PATH,
    VQR_LOSS_HISTORY_PATH,
)

TARGET = "units_sold"
DATE_COLUMN = "transaction_date"
MODEL_NAME = "perishable-demand-vqr"
MODEL_VERSION = "1.0.0"
MAPE_EPSILON = 1e-8

CANDIDATE_MODEL_FEATURES = [
    "shelf_life_days",
    "cost_price",
    "spoilage_sensitivity",
    "spoilage_risk",
    "selling_price",
    "discount_pct",
    "is_promoted",
    "supplier_score",
    "storage_temp",
    "category_Bakery",
    "category_Beverages",
    "category_Dairy",
    "category_Deli",
    "category_Frozen_Meals",
    "category_Meat",
    "category_Pharmaceuticals",
    "category_Produce",
    "category_Ready_to_Eat",
    "category_Seafood",
    "region_Midwest",
    "region_Northeast",
    "region_Southeast",
    "region_Southwest",
    "region_West",
]

LEAKAGE_COLUMNS = {
    "units_sold",
    "revenue",
    "profit",
    "profit_margin_pct",
    "units_wasted",
    "waste_pct",
    "waste_cost",
    "was_spoiled",
    "daily_demand",
    "future_demand",
    "future_sales",
}

DEFAULT_FEATURE_COUNTS = (4, 6, 8, 10, 12, 14)


@dataclass(frozen=True)
class ExperimentConfig:
    """One VQR circuit and optimizer configuration."""

    feature_map: str
    feature_map_reps: int
    ansatz: str
    ansatz_reps: int
    entanglement: str
    optimizer: str
    maxiter: int


DEFAULT_EXPERIMENT_CONFIGS = (
    ExperimentConfig(
        feature_map="zz",
        feature_map_reps=1,
        ansatz="real_amplitudes",
        ansatz_reps=1,
        entanglement="linear",
        optimizer="COBYLA",
        maxiter=20,
    ),
    ExperimentConfig(
        feature_map="z",
        feature_map_reps=1,
        ansatz="real_amplitudes",
        ansatz_reps=1,
        entanglement="linear",
        optimizer="COBYLA",
        maxiter=20,
    ),
)


@dataclass(frozen=True)
class WorkflowConfig:
    """Safe defaults for a small pipeline experiment."""

    fast_mode: bool = True
    maxiter_fast: int = 20
    maxiter_final: int = 100
    feature_counts: tuple[int, ...] = DEFAULT_FEATURE_COUNTS
    train_sample_size: int = 100
    validation_sample_size: int = 30
    test_sample_size: int = 30
    random_state: int = 42
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    include_spoilage_risk: bool = False
    experiment_configs: tuple[ExperimentConfig, ...] = DEFAULT_EXPERIMENT_CONFIGS


@dataclass(frozen=True)
class WorkflowPaths:
    """Inputs and outputs for the workflow."""

    data_path: Path
    experiment_results_path: Path
    loss_history_path: Path
    model_dir: Path


@dataclass
class RawSplit:
    """Raw, sampled train/validation/test frames."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    method: str


@dataclass
class PreparedData:
    """Model-ready candidate matrices before feature selection or scaling."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: NDArray[np.float64]
    y_validation: NDArray[np.float64]
    y_test: NDArray[np.float64]
    raw_test: pd.DataFrame
    feature_schema: dict[str, object]
    split_method: str


@dataclass
class ExperimentMatrices:
    """One feature-selection and scaling fit, trained on training data only."""

    selector: SelectKBest
    x_scaler: MinMaxScaler
    y_scaler: MinMaxScaler
    selected_features: list[str]
    X_train_scaled: NDArray[np.float64]
    X_validation_scaled: NDArray[np.float64]
    X_test_scaled: NDArray[np.float64]
    y_train_scaled: NDArray[np.float64]
    y_validation_scaled: NDArray[np.float64]
    y_test_scaled: NDArray[np.float64]


@dataclass
class FinalTrainingResult:
    """Winning fitted objects and final, one-time test results."""

    model: VQR | None
    matrices: ExperimentMatrices
    experiment_config: ExperimentConfig
    feature_count: int
    training_seconds: float
    test_predictions: NDArray[np.float64]
    test_metrics: dict[str, float]
    baseline_test_predictions: NDArray[np.float64]
    baseline_test_metrics: dict[str, float]
    loss_rows: list[dict[str, object]]


@dataclass
class LossRecorder:
    """Record objective evaluations without excessive console output."""

    experiment_id: str
    started_at: float = field(default_factory=time.perf_counter)
    rows: list[dict[str, object]] = field(default_factory=list)

    def __call__(self, weights: NDArray[np.float64], objective_value: float) -> None:
        del weights
        evaluation_number = len(self.rows) + 1
        elapsed = time.perf_counter() - self.started_at
        self.rows.append(
            {
                "experiment_id": self.experiment_id,
                "evaluation_number": evaluation_number,
                "objective_value": float(objective_value),
                "elapsed_seconds": elapsed,
            }
        )
        if evaluation_number == 1 or evaluation_number % 5 == 0:
            print(
                f"{self.experiment_id}: Evaluation {evaluation_number}, "
                f"objective={objective_value:.6f}, elapsed={elapsed:.1f}s"
            )


def default_paths() -> WorkflowPaths:
    """Build workflow paths from the shared project constants."""
    return WorkflowPaths(
        data_path=Path(PERISHABLE_GOODS_DATA_PATH),
        experiment_results_path=Path(VQR_EXPERIMENT_RESULTS_PATH),
        loss_history_path=Path(VQR_LOSS_HISTORY_PATH),
        model_dir=Path(PERISHABLE_VQR_MODEL_PATH),
    )


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load the requested CSV without mutating it."""
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Confirm the repository root and data file."
        )
    data = pd.read_csv(data_path)
    if data.empty:
        raise ValueError(f"Dataset at {data_path} is empty.")
    return data


def _available_candidate_sources(
    data: pd.DataFrame, include_spoilage_risk: bool
) -> list[str]:
    """Resolve raw or already encoded candidate inputs without double encoding."""
    allowed = [
        name
        for name in CANDIDATE_MODEL_FEATURES
        if include_spoilage_risk or name != "spoilage_risk"
    ]
    sources = [
        name
        for name in allowed
        if name in data.columns
        and not name.startswith("category_")
        and not name.startswith("region_")
    ]
    for prefix, raw_name in (("category_", "category"), ("region_", "region")):
        encoded = [name for name in allowed if name.startswith(prefix) and name in data.columns]
        if encoded:
            sources.extend(encoded)
        elif raw_name in data.columns:
            sources.append(raw_name)
    return sources


def validate_dataset(
    data: pd.DataFrame,
    *,
    config: WorkflowConfig,
    target: str = TARGET,
) -> dict[str, object]:
    """Report and validate dataset issues before any split or fitting."""
    if target not in data.columns:
        raise ValueError(f"Required target column {target!r} is missing.")

    candidate_sources = _available_candidate_sources(data, config.include_spoilage_risk)
    duplicated_features = sorted(
        {name for name in candidate_sources if candidate_sources.count(name) > 1}
    )
    if duplicated_features:
        raise ValueError(f"Duplicate selected features detected: {duplicated_features}")
    if len(candidate_sources) < min(config.feature_counts):
        raise ValueError(
            f"Only {len(candidate_sources)} candidate inputs are available, fewer than "
            f"the smallest requested feature count ({min(config.feature_counts)})."
        )

    target_numeric = pd.to_numeric(data[target], errors="coerce")
    invalid_target = int((target_numeric.isna() & data[target].notna()).sum())
    missing_values = data[[target, *candidate_sources]].isna().sum()

    numeric_sources = [
        name for name in candidate_sources if name not in {"category", "region"}
    ]
    invalid_numeric: dict[str, int] = {}
    infinite_values: dict[str, int] = {}
    constant_columns: list[str] = []
    for column in numeric_sources:
        converted = pd.to_numeric(data[column], errors="coerce")
        invalid_count = int((converted.isna() & data[column].notna()).sum())
        if invalid_count:
            invalid_numeric[column] = invalid_count
        finite_array = converted.to_numpy(dtype=float, na_value=np.nan)
        infinite_count = int(np.isinf(finite_array).sum())
        if infinite_count:
            infinite_values[column] = infinite_count
        if converted.nunique(dropna=False) <= 1:
            constant_columns.append(column)

    target_array = target_numeric.to_numpy(dtype=float, na_value=np.nan)
    target_infinite = int(np.isinf(target_array).sum())
    if target_infinite:
        infinite_values[target] = target_infinite
    if target_numeric.nunique(dropna=False) <= 1:
        constant_columns.append(target)

    duplicate_rows = int(data.duplicated().sum())
    requested_total = (
        config.train_sample_size
        + config.validation_sample_size
        + config.test_sample_size
    )
    report: dict[str, object] = {
        "Rows": int(data.shape[0]),
        "Columns": int(data.shape[1]),
        "Target": target,
        "Target minimum": float(target_numeric.min()),
        "Target maximum": float(target_numeric.max()),
        "Target mean": float(target_numeric.mean()),
        "Missing values": int(data.isna().sum().sum()),
        "Duplicate rows": duplicate_rows,
        "Candidate feature count": len(candidate_sources),
    }
    print("Dataset report")
    for label, value in report.items():
        print(f"{label}: {value}")
    print("Missing values by selected column:", missing_values[missing_values > 0].to_dict())
    print("Infinite values by selected column:", infinite_values)
    print("Constant selected columns:", constant_columns)
    print("Invalid numeric conversions:", invalid_numeric)
    if len(data) < requested_total:
        print(
            f"WARNING: {len(data)} rows are fewer than the requested sampled total "
            f"of {requested_total}; split-specific sample sizes will be reduced safely."
        )

    validation_errors: list[str] = []
    if invalid_target:
        validation_errors.append(f"{invalid_target} target values are not numeric")
    if int(data[target].isna().sum()):
        validation_errors.append("the target contains missing values")
    if target_infinite:
        validation_errors.append(f"the target contains {target_infinite} infinite values")
    if invalid_numeric:
        validation_errors.append(f"numeric conversion failed: {invalid_numeric}")
    if int(missing_values.sum()):
        validation_errors.append(
            "selected inputs or target contain missing values; no rows were deleted"
        )
    if infinite_values:
        validation_errors.append(f"selected inputs contain infinite values: {infinite_values}")
    if len(data) < 9:
        validation_errors.append("at least 9 rows are required for three usable subsets")
    if validation_errors:
        raise ValueError("Dataset validation failed: " + "; ".join(validation_errors))
    return report


def review_target_leakage(
    data: pd.DataFrame,
    *,
    include_spoilage_risk: bool,
    target: str = TARGET,
) -> dict[str, object]:
    """Identify obvious post-outcome fields and enforce the candidate whitelist."""
    normalized_columns = {
        column: column.strip().lower().replace(" ", "_") for column in data.columns
    }
    present_risky = sorted(
        column
        for column, normalized in normalized_columns.items()
        if normalized in LEAKAGE_COLUMNS
        or (
            "future" in normalized
            and ("demand" in normalized or "sales" in normalized)
        )
    )
    candidate_sources = _available_candidate_sources(data, include_spoilage_risk)
    forbidden_selected = sorted(
        column
        for column in candidate_sources
        if normalized_columns[column] in LEAKAGE_COLUMNS - {target}
    )
    if forbidden_selected:
        raise ValueError(f"Leakage columns selected as model inputs: {forbidden_selected}")

    print("Leakage review")
    print("Present direct/post-outcome fields (excluded from inputs):", present_risky)
    print(
        "WARNING: spoilage_risk is safe only if it is known when the demand prediction "
        "is generated and is not derived from future sales, waste, or spoilage outcomes."
    )
    print("spoilage_risk included:", include_spoilage_risk)
    return {
        "present_risky_columns": present_risky,
        "selected_candidate_sources": candidate_sources,
        "spoilage_risk_included": include_spoilage_risk,
    }


def prepare_raw_inputs(
    data: pd.DataFrame,
    *,
    include_spoilage_risk: bool,
    target: str = TARGET,
) -> pd.DataFrame:
    """Copy only prediction-time candidates, target, date, and source row identity."""
    candidate_sources = _available_candidate_sources(data, include_spoilage_risk)
    columns = [*candidate_sources, target]
    if DATE_COLUMN in data.columns:
        columns.append(DATE_COLUMN)
    prepared = data.loc[:, columns].copy()
    prepared[target] = pd.to_numeric(prepared[target], errors="raise").astype(float)
    prepared["__source_index__"] = data.index.to_numpy()
    return prepared


def _subset_sample(
    frame: pd.DataFrame,
    requested_size: int,
    *,
    chronological: bool,
    random_state: int,
) -> pd.DataFrame:
    """Take a safe deterministic sample from one already isolated subset."""
    if requested_size < 1:
        raise ValueError("All requested sample sizes must be positive.")
    size = min(requested_size, len(frame))
    if size < requested_size:
        print(
            f"WARNING: requested {requested_size} rows from a {len(frame)}-row subset; "
            f"using {size}."
        )
    if chronological:
        sampled = frame.iloc[:size]
    else:
        sampled = frame.sample(n=size, random_state=random_state)
    return sampled.reset_index(drop=True)


def split_raw_data(data: pd.DataFrame, config: WorkflowConfig) -> RawSplit:
    """Split first, then sample within each isolated subset."""
    fractions_sum = (
        config.train_fraction + config.validation_fraction + config.test_fraction
    )
    if not math.isclose(fractions_sum, 1.0, abs_tol=1e-9):
        raise ValueError("Train, validation, and test fractions must sum to 1.0.")

    chronological = False
    working = data.copy()
    if DATE_COLUMN in working.columns:
        parsed_dates = pd.to_datetime(working[DATE_COLUMN], errors="coerce", utc=True)
        if parsed_dates.notna().all():
            working = (
                working.assign(**{DATE_COLUMN: parsed_dates})
                .sort_values(DATE_COLUMN, kind="stable")
                .reset_index(drop=True)
            )
            chronological = True
            method = "chronological"
        else:
            print(
                f"WARNING: {DATE_COLUMN} could not be parsed completely; using a "
                "deterministic random split. This limits forecasting interpretation."
            )
            method = "deterministic_random"
    else:
        print(
            f"WARNING: {DATE_COLUMN} is absent, so a deterministic random split is used. "
            "Future records cannot be isolated chronologically in this featured CSV."
        )
        method = "deterministic_random"

    if chronological:
        train_end = max(1, int(len(working) * config.train_fraction))
        validation_end = train_end + max(
            1, int(len(working) * config.validation_fraction)
        )
        validation_end = min(validation_end, len(working) - 1)
        train_frame = working.iloc[:train_end]
        validation_frame = working.iloc[train_end:validation_end]
        test_frame = working.iloc[validation_end:]
    else:
        train_frame, remainder = train_test_split(
            working,
            test_size=1.0 - config.train_fraction,
            random_state=config.random_state,
            shuffle=True,
        )
        relative_test_fraction = config.test_fraction / (
            config.validation_fraction + config.test_fraction
        )
        validation_frame, test_frame = train_test_split(
            remainder,
            test_size=relative_test_fraction,
            random_state=config.random_state,
            shuffle=True,
        )

    subsets = (train_frame, validation_frame, test_frame)
    if any(len(frame) < 2 for frame in subsets):
        raise ValueError(
            "The configured split produced a subset with fewer than two rows. "
            "Use more data or adjust the split fractions."
        )

    return RawSplit(
        train=_subset_sample(
            train_frame,
            config.train_sample_size,
            chronological=chronological,
            random_state=config.random_state,
        ),
        validation=_subset_sample(
            validation_frame,
            config.validation_sample_size,
            chronological=chronological,
            random_state=config.random_state + 1,
        ),
        test=_subset_sample(
            test_frame,
            config.test_sample_size,
            chronological=chronological,
            random_state=config.random_state + 2,
        ),
        method=method,
    )


def _safe_category_token(value: str) -> str:
    """Create deterministic column suffixes for raw categorical levels."""
    token = "".join(character if character.isalnum() else "_" for character in value)
    return "_".join(part for part in token.split("_") if part) or "EMPTY"


def fit_candidate_schema(
    training_data: pd.DataFrame,
    *,
    include_spoilage_risk: bool,
    split_method: str,
) -> dict[str, object]:
    """Fit categorical vocabulary on training data only."""
    allowed = [
        name
        for name in CANDIDATE_MODEL_FEATURES
        if include_spoilage_risk or name != "spoilage_risk"
    ]
    numeric_features = [
        name
        for name in allowed
        if name in training_data.columns
        and not name.startswith("category_")
        and not name.startswith("region_")
    ]
    categorical_encoding: dict[str, object] = {}
    candidate_order = list(numeric_features)
    raw_candidate_features = list(numeric_features)

    for prefix, raw_name in (("category_", "category"), ("region_", "region")):
        pre_encoded = [
            name for name in allowed if name.startswith(prefix) and name in training_data.columns
        ]
        if pre_encoded:
            categorical_encoding[raw_name] = {
                "mode": "pre_encoded",
                "source_column": None,
                "categories": [],
                "output_columns": pre_encoded,
                "unknown_policy": "not_applicable",
            }
            candidate_order.extend(pre_encoded)
            raw_candidate_features.extend(pre_encoded)
            continue
        if raw_name not in training_data.columns:
            continue

        normalized = training_data[raw_name].fillna("__MISSING__").astype(str)
        categories = sorted(normalized.unique().tolist())
        output_columns = [f"{raw_name}_{_safe_category_token(item)}" for item in categories]
        if len(output_columns) != len(set(output_columns)):
            raise ValueError(
                f"Categorical values in {raw_name!r} collapse to duplicate encoded names."
            )
        categorical_encoding[raw_name] = {
            "mode": "raw_one_hot",
            "source_column": raw_name,
            "categories": categories,
            "output_columns": output_columns,
            "unknown_policy": "all_zero",
        }
        candidate_order.extend(output_columns)
        raw_candidate_features.append(raw_name)

    if len(candidate_order) != len(set(candidate_order)):
        raise ValueError("Candidate feature schema contains duplicate output columns.")
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "target": TARGET,
        "raw_candidate_features": raw_candidate_features,
        "selected_features": [],
        "feature_order": [],
        "feature_count": 0,
        "categorical_encoding": categorical_encoding,
        "target_scaled": True,
        "raw_numeric_features": numeric_features,
        "candidate_feature_order": candidate_order,
        "split_method": split_method,
        "date_column": DATE_COLUMN if DATE_COLUMN in training_data.columns else None,
        "spoilage_risk_included": include_spoilage_risk,
    }


def _string_list(value: object, field_name: str) -> list[str]:
    """Validate and return a JSON-style list of strings."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{field_name} must be a list of strings.")
    return list(value)


def transform_candidates_from_schema(
    raw_data: pd.DataFrame, schema: Mapping[str, object]
) -> pd.DataFrame:
    """Apply saved deterministic candidate preparation without fitting."""
    numeric_features = _string_list(
        schema.get("raw_numeric_features"), "raw_numeric_features"
    )
    candidate_order = _string_list(
        schema.get("candidate_feature_order"), "candidate_feature_order"
    )
    output = pd.DataFrame(index=raw_data.index)

    for column in numeric_features:
        if column not in raw_data.columns:
            raise ValueError(f"Inference input is missing numeric feature {column!r}.")
        converted = pd.to_numeric(raw_data[column], errors="coerce")
        if converted.isna().any() or np.isinf(converted.to_numpy(dtype=float)).any():
            raise ValueError(f"Inference feature {column!r} is missing, invalid, or infinite.")
        output[column] = converted.astype(float)

    encoding_value = schema.get("categorical_encoding")
    if not isinstance(encoding_value, Mapping):
        raise TypeError("categorical_encoding must be a mapping.")
    for group_name, raw_rule in encoding_value.items():
        if not isinstance(group_name, str) or not isinstance(raw_rule, Mapping):
            raise TypeError("Each categorical encoding rule must be a named mapping.")
        mode = raw_rule.get("mode")
        output_columns = _string_list(
            raw_rule.get("output_columns"), f"{group_name}.output_columns"
        )
        if mode == "pre_encoded":
            for column in output_columns:
                if column not in raw_data.columns:
                    raise ValueError(
                        f"Inference input is missing encoded feature {column!r}."
                    )
                converted = pd.to_numeric(raw_data[column], errors="coerce")
                if converted.isna().any():
                    raise ValueError(f"Encoded feature {column!r} is not numeric.")
                output[column] = converted.astype(float)
        elif mode == "raw_one_hot":
            source = raw_rule.get("source_column")
            if not isinstance(source, str) or source not in raw_data.columns:
                raise ValueError(f"Inference input is missing categorical feature {source!r}.")
            categories = _string_list(
                raw_rule.get("categories"), f"{group_name}.categories"
            )
            if len(categories) != len(output_columns):
                raise ValueError(f"Invalid saved encoding rule for {group_name!r}.")
            values = raw_data[source].fillna("__MISSING__").astype(str)
            for category, column in zip(categories, output_columns, strict=True):
                output[column] = (values == category).astype(float)
        else:
            raise ValueError(f"Unsupported categorical encoding mode {mode!r}.")

    missing_outputs = [name for name in candidate_order if name not in output.columns]
    if missing_outputs:
        raise ValueError(f"Schema transformation did not produce: {missing_outputs}")
    transformed = output.loc[:, candidate_order]
    if transformed.columns.duplicated().any():
        raise ValueError("Transformed candidate features contain duplicate columns.")
    return transformed


def build_prepared_data(raw_split: RawSplit, schema: dict[str, object]) -> PreparedData:
    """Transform all subsets using the training-fitted categorical schema."""
    X_train = transform_candidates_from_schema(raw_split.train, schema)
    X_validation = transform_candidates_from_schema(raw_split.validation, schema)
    X_test = transform_candidates_from_schema(raw_split.test, schema)
    return PreparedData(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=raw_split.train[TARGET].to_numpy(dtype=float),
        y_validation=raw_split.validation[TARGET].to_numpy(dtype=float),
        y_test=raw_split.test[TARGET].to_numpy(dtype=float),
        raw_test=raw_split.test.copy(),
        feature_schema=schema,
        split_method=raw_split.method,
    )


def _mutual_information(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    random_state: int,
) -> NDArray[np.float64]:
    """Typed score function wrapper for deterministic mutual information."""
    return mutual_info_regression(X, y, random_state=random_state)


def prepare_experiment_matrices(
    data: PreparedData,
    *,
    feature_count: int,
    random_state: int,
) -> ExperimentMatrices:
    """Fit selector and scalers only on training data."""
    if feature_count > data.X_train.shape[1]:
        raise ValueError(
            f"k={feature_count} exceeds {data.X_train.shape[1]} available candidates."
        )
    score_function = partial(_mutual_information, random_state=random_state)
    selector = SelectKBest(score_func=score_function, k=feature_count)
    X_train_selected = selector.fit_transform(data.X_train, data.y_train)
    X_validation_selected = selector.transform(data.X_validation)
    X_test_selected = selector.transform(data.X_test)
    selected_features = data.X_train.columns[selector.get_support()].tolist()

    x_scaler = MinMaxScaler()
    X_train_scaled = x_scaler.fit_transform(X_train_selected)
    X_validation_scaled = x_scaler.transform(X_validation_selected)
    X_test_scaled = x_scaler.transform(X_test_selected)

    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(data.y_train.reshape(-1, 1)).ravel()
    y_validation_scaled = y_scaler.transform(
        data.y_validation.reshape(-1, 1)
    ).ravel()
    y_test_scaled = y_scaler.transform(data.y_test.reshape(-1, 1)).ravel()

    return ExperimentMatrices(
        selector=selector,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        selected_features=selected_features,
        X_train_scaled=np.asarray(X_train_scaled, dtype=float),
        X_validation_scaled=np.asarray(X_validation_scaled, dtype=float),
        X_test_scaled=np.asarray(X_test_scaled, dtype=float),
        y_train_scaled=np.asarray(y_train_scaled, dtype=float),
        y_validation_scaled=np.asarray(y_validation_scaled, dtype=float),
        y_test_scaled=np.asarray(y_test_scaled, dtype=float),
    )


def regression_metrics(
    y_true: NDArray[np.float64],
    y_predicted: NDArray[np.float64],
    *,
    epsilon: float = MAPE_EPSILON,
) -> dict[str, float]:
    """Compute original-unit metrics; MAPE excludes |target| <= 1e-8 by default."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_predicted, dtype=float).reshape(-1)
    mse = float(mean_squared_error(actual, predicted))
    nonzero = np.abs(actual) > epsilon
    mape = (
        float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)
        if nonzero.any()
        else float("nan")
    )
    denominator = np.abs(actual) + np.abs(predicted)
    valid_smape = denominator > epsilon
    smape = (
        float(
            np.mean(
                2.0
                * np.abs(predicted[valid_smape] - actual[valid_smape])
                / denominator[valid_smape]
            )
            * 100
        )
        if valid_smape.any()
        else 0.0
    )
    return {
        "r2_score": float(r2_score(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mape": mape,
        "smape": smape,
    }


def _result_row(
    *,
    experiment_id: str,
    model_name: str,
    feature_count: int,
    selected_features: Sequence[str],
    experiment_config: ExperimentConfig | None,
    training_samples: int,
    validation_samples: int,
    training_seconds: float,
    metrics: Mapping[str, float] | None,
    status: str,
    error_message: str,
) -> dict[str, object]:
    """Create one stable experiment result record."""
    config_values = (
        asdict(experiment_config)
        if experiment_config is not None
        else {
            "feature_map": "",
            "feature_map_reps": 0,
            "ansatz": "",
            "ansatz_reps": 0,
            "entanglement": "",
            "optimizer": "Ridge",
            "maxiter": 0,
        }
    )
    metric_values = metrics or {
        "r2_score": float("nan"),
        "mae": float("nan"),
        "mse": float("nan"),
        "rmse": float("nan"),
        "mape": float("nan"),
        "smape": float("nan"),
    }
    return {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "feature_count": feature_count,
        "selected_features": json.dumps(list(selected_features)),
        **config_values,
        "training_samples": training_samples,
        "validation_samples": validation_samples,
        "training_seconds": training_seconds,
        **metric_values,
        "status": status,
        "error_message": error_message,
    }


def valid_feature_counts(
    requested: Sequence[int], available_count: int
) -> list[int]:
    """Skip impossible feature counts with a visible message."""
    valid: list[int] = []
    for feature_count in requested:
        if feature_count < 1:
            print(f"Skipping invalid feature count {feature_count}.")
        elif feature_count > available_count:
            print(
                f"Skipping k={feature_count}; only {available_count} candidate features exist."
            )
        elif feature_count not in valid:
            valid.append(feature_count)
    if not valid:
        raise ValueError("No requested feature count can be trained.")
    return valid


def run_classical_baselines(
    data: PreparedData, config: WorkflowConfig
) -> list[dict[str, object]]:
    """Train a fast Ridge baseline for each valid k on the same split."""
    rows: list[dict[str, object]] = []
    counts = valid_feature_counts(config.feature_counts, data.X_train.shape[1])
    for feature_count in counts:
        experiment_id = f"ridge_k{feature_count}"
        selected_features: list[str] = []
        training_seconds = 0.0
        try:
            matrices = prepare_experiment_matrices(
                data,
                feature_count=feature_count,
                random_state=config.random_state,
            )
            selected_features = matrices.selected_features
            model = Ridge(alpha=1.0)
            started = time.perf_counter()
            model.fit(matrices.X_train_scaled, matrices.y_train_scaled)
            training_seconds = time.perf_counter() - started
            scaled_predictions = model.predict(matrices.X_validation_scaled)
            predictions = matrices.y_scaler.inverse_transform(
                np.asarray(scaled_predictions).reshape(-1, 1)
            ).ravel()
            predictions = np.clip(predictions, 0.0, None)
            metrics = regression_metrics(data.y_validation, predictions)
            rows.append(
                _result_row(
                    experiment_id=experiment_id,
                    model_name="Ridge",
                    feature_count=feature_count,
                    selected_features=selected_features,
                    experiment_config=None,
                    training_samples=len(data.y_train),
                    validation_samples=len(data.y_validation),
                    training_seconds=training_seconds,
                    metrics=metrics,
                    status="success",
                    error_message="",
                )
            )
        except Exception as exc:
            rows.append(
                _result_row(
                    experiment_id=experiment_id,
                    model_name="Ridge",
                    feature_count=feature_count,
                    selected_features=selected_features,
                    experiment_config=None,
                    training_samples=len(data.y_train),
                    validation_samples=len(data.y_validation),
                    training_seconds=training_seconds,
                    metrics=None,
                    status="failed",
                    error_message=f"{type(exc).__name__}: {exc}",
                )
            )
            print(f"{experiment_id} failed: {type(exc).__name__}: {exc}")
    return rows


def effective_experiment_configs(config: WorkflowConfig) -> list[ExperimentConfig]:
    """Apply visible fast/final iteration limits to shallow candidate circuits."""
    maxiter = config.maxiter_fast if config.fast_mode else config.maxiter_final
    selected = list(config.experiment_configs)
    if config.fast_mode:
        selected = selected[:2]
        print(
            "FAST_MODE=True: using small samples, shallow circuits, and at most "
            f"{maxiter} optimizer evaluations. This is a pipeline experiment, "
            "not a production-quality conclusion."
        )
    else:
        print(
            "FAST_MODE=False: this can be expensive. Review sample sizes, feature "
            f"counts (qubits), circuit depth, {len(selected)} configurations, and "
            f"maxiter={maxiter} before continuing."
        )
    return [replace(item, maxiter=maxiter) for item in selected]


def run_vqr_experiments(
    data: PreparedData, config: WorkflowConfig
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run isolated VQR experiments; failures are recorded and do not abort the loop."""
    rows: list[dict[str, object]] = []
    loss_rows: list[dict[str, object]] = []
    counts = valid_feature_counts(config.feature_counts, data.X_train.shape[1])
    experiment_configs = effective_experiment_configs(config)

    for feature_count in counts:
        for config_index, experiment_config in enumerate(experiment_configs, start=1):
            experiment_id = (
                f"vqr_k{feature_count}_{experiment_config.feature_map}_{config_index}"
            )
            selected_features: list[str] = []
            training_seconds = 0.0
            model: VQR | None = None
            callback: LossRecorder | None = None
            try:
                matrices = prepare_experiment_matrices(
                    data,
                    feature_count=feature_count,
                    random_state=config.random_state,
                )
                selected_features = matrices.selected_features
                callback = LossRecorder(experiment_id)
                model = build_vqr(
                    feature_count=feature_count,
                    feature_map_name=experiment_config.feature_map,
                    feature_map_reps=experiment_config.feature_map_reps,
                    ansatz_name=experiment_config.ansatz,
                    ansatz_reps=experiment_config.ansatz_reps,
                    entanglement=experiment_config.entanglement,
                    optimizer_name=experiment_config.optimizer,
                    maxiter=experiment_config.maxiter,
                    random_state=config.random_state,
                    callback=callback,
                )
                started = time.perf_counter()
                model.fit(matrices.X_train_scaled, matrices.y_train_scaled)
                training_seconds = time.perf_counter() - started
                scaled_predictions = model.predict(matrices.X_validation_scaled)
                predictions = matrices.y_scaler.inverse_transform(
                    np.asarray(scaled_predictions).reshape(-1, 1)
                ).ravel()
                predictions = np.clip(predictions, 0.0, None)
                metrics = regression_metrics(data.y_validation, predictions)
                loss_rows.extend(callback.rows)
                rows.append(
                    _result_row(
                        experiment_id=experiment_id,
                        model_name="VQR",
                        feature_count=feature_count,
                        selected_features=selected_features,
                        experiment_config=experiment_config,
                        training_samples=len(data.y_train),
                        validation_samples=len(data.y_validation),
                        training_seconds=training_seconds,
                        metrics=metrics,
                        status="success",
                        error_message="",
                    )
                )
            except Exception as exc:
                if callback is not None:
                    loss_rows.extend(callback.rows)
                rows.append(
                    _result_row(
                        experiment_id=experiment_id,
                        model_name="VQR",
                        feature_count=feature_count,
                        selected_features=selected_features,
                        experiment_config=experiment_config,
                        training_samples=len(data.y_train),
                        validation_samples=len(data.y_validation),
                        training_seconds=training_seconds,
                        metrics=None,
                        status="failed",
                        error_message=f"{type(exc).__name__}: {exc}",
                    )
                )
                print(f"{experiment_id} failed: {type(exc).__name__}: {exc}")
            finally:
                model = None
                gc.collect()
    return rows, loss_rows


def save_experiment_tables(
    result_rows: Sequence[Mapping[str, object]],
    loss_rows: Sequence[Mapping[str, object]],
    paths: WorkflowPaths,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Persist validation experiments and loss history."""
    paths.experiment_results_path.parent.mkdir(parents=True, exist_ok=True)
    result_columns = [
        "experiment_id",
        "model_name",
        "feature_count",
        "selected_features",
        "feature_map",
        "feature_map_reps",
        "ansatz",
        "ansatz_reps",
        "entanglement",
        "optimizer",
        "maxiter",
        "training_samples",
        "validation_samples",
        "training_seconds",
        "r2_score",
        "mae",
        "mse",
        "rmse",
        "mape",
        "smape",
        "status",
        "error_message",
    ]
    loss_columns = [
        "experiment_id",
        "evaluation_number",
        "objective_value",
        "elapsed_seconds",
    ]
    results = pd.DataFrame(result_rows, columns=result_columns)
    loss_history = pd.DataFrame(loss_rows, columns=loss_columns)
    results.to_csv(paths.experiment_results_path, index=False)
    loss_history.to_csv(paths.loss_history_path, index=False)
    print(f"Saved experiment results: {paths.experiment_results_path}")
    print(f"Saved loss history: {paths.loss_history_path}")
    return results, loss_history


def rank_successful_vqr(results: pd.DataFrame) -> pd.DataFrame:
    """Rank successful VQR configurations by the documented validation priority."""
    successful = results[
        (results["model_name"] == "VQR") & (results["status"] == "success")
    ].copy()
    if successful.empty:
        failures = results[results["model_name"] == "VQR"][
            ["experiment_id", "error_message"]
        ]
        raise RuntimeError(f"No VQR experiment succeeded.\n{failures.to_string(index=False)}")
    successful["_r2_tie_group"] = successful["r2_score"].round(4)
    successful["_mae_tie_group"] = successful["mae"].round(4)
    ranked = successful.sort_values(
        ["_r2_tie_group", "_mae_tie_group", "rmse", "training_seconds"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    return ranked.drop(columns=["_r2_tie_group", "_mae_tie_group"])


def experiment_config_from_row(row: pd.Series) -> ExperimentConfig:
    """Reconstruct the selected configuration from a result row."""
    return ExperimentConfig(
        feature_map=str(row["feature_map"]),
        feature_map_reps=int(row["feature_map_reps"]),
        ansatz=str(row["ansatz"]),
        ansatz_reps=int(row["ansatz_reps"]),
        entanglement=str(row["entanglement"]),
        optimizer=str(row["optimizer"]),
        maxiter=int(row["maxiter"]),
    )


def fit_winner_and_evaluate_test(
    data: PreparedData,
    winning_row: pd.Series,
    *,
    random_state: int,
) -> FinalTrainingResult:
    """Retrain the chosen VQR and inspect the untouched test set exactly once."""
    feature_count = int(winning_row["feature_count"])
    experiment_config = experiment_config_from_row(winning_row)
    matrices = prepare_experiment_matrices(
        data,
        feature_count=feature_count,
        random_state=random_state,
    )
    callback = LossRecorder("final_winner_retrain")
    model = build_vqr(
        feature_count=feature_count,
        feature_map_name=experiment_config.feature_map,
        feature_map_reps=experiment_config.feature_map_reps,
        ansatz_name=experiment_config.ansatz,
        ansatz_reps=experiment_config.ansatz_reps,
        entanglement=experiment_config.entanglement,
        optimizer_name=experiment_config.optimizer,
        maxiter=experiment_config.maxiter,
        random_state=random_state,
        callback=callback,
    )
    started = time.perf_counter()
    model.fit(matrices.X_train_scaled, matrices.y_train_scaled)
    training_seconds = time.perf_counter() - started
    scaled_predictions = model.predict(matrices.X_test_scaled)
    predictions = matrices.y_scaler.inverse_transform(
        np.asarray(scaled_predictions).reshape(-1, 1)
    ).ravel()
    predictions = np.clip(predictions, 0.0, None)
    test_metrics = regression_metrics(data.y_test, predictions)

    baseline = Ridge(alpha=1.0)
    baseline.fit(matrices.X_train_scaled, matrices.y_train_scaled)
    baseline_scaled = baseline.predict(matrices.X_test_scaled)
    baseline_predictions = matrices.y_scaler.inverse_transform(
        np.asarray(baseline_scaled).reshape(-1, 1)
    ).ravel()
    baseline_predictions = np.clip(baseline_predictions, 0.0, None)
    baseline_metrics = regression_metrics(data.y_test, baseline_predictions)

    print("Selected features in exact order:", matrices.selected_features)
    print("Final VQR test metrics:", test_metrics)
    print("Ridge test metrics on the same selected features:", baseline_metrics)
    print(
        "Interpret these results comparatively; a small error metric alone does not "
        "establish that the VQR generalizes well."
    )
    return FinalTrainingResult(
        model=model,
        matrices=matrices,
        experiment_config=experiment_config,
        feature_count=feature_count,
        training_seconds=training_seconds,
        test_predictions=predictions,
        test_metrics=test_metrics,
        baseline_test_predictions=baseline_predictions,
        baseline_test_metrics=baseline_metrics,
        loss_rows=callback.rows,
    )


def _package_version(package_name: str) -> str:
    """Return an installed package version for metadata."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


def save_winning_artifacts(
    final: FinalTrainingResult,
    data: PreparedData,
    paths: WorkflowPaths,
) -> dict[str, Path]:
    """Save the model, preprocessing, schema, and metadata for inference."""
    if final.model is None:
        raise ValueError("The winning model is not available to save.")
    paths.model_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "model": paths.model_dir / "vqr_model.dill",
        "x_scaler": paths.model_dir / "x_scaler.joblib",
        "y_scaler": paths.model_dir / "y_scaler.joblib",
        "selector": paths.model_dir / "feature_selector.joblib",
        "feature_schema": paths.model_dir / "feature_schema.json",
        "metadata": paths.model_dir / "model_metadata.json",
        "test_prediction": paths.model_dir / "test_prediction.json",
    }
    final.model.to_dill(str(artifact_paths["model"]))
    joblib.dump(final.matrices.x_scaler, artifact_paths["x_scaler"])
    joblib.dump(final.matrices.y_scaler, artifact_paths["y_scaler"])
    joblib.dump(final.matrices.selector, artifact_paths["selector"])

    feature_schema = dict(data.feature_schema)
    feature_schema.update(
        {
            "selected_features": final.matrices.selected_features,
            "feature_order": final.matrices.selected_features,
            "feature_count": final.feature_count,
        }
    )
    artifact_paths["feature_schema"].write_text(
        json.dumps(feature_schema, indent=2), encoding="utf-8"
    )

    metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_type": "VQR",
        "target": TARGET,
        "feature_count": final.feature_count,
        "training_sample_count": len(data.y_train),
        "validation_sample_count": len(data.y_validation),
        "test_sample_count": len(data.y_test),
        "qiskit_version": _package_version("qiskit"),
        "qiskit_machine_learning_version": _package_version(
            "qiskit-machine-learning"
        ),
        "scikit_learn_version": _package_version("scikit-learn"),
        "training_seconds": final.training_seconds,
        "test_metrics": {
            "r2": final.test_metrics["r2_score"],
            "mae": final.test_metrics["mae"],
            "mse": final.test_metrics["mse"],
            "rmse": final.test_metrics["rmse"],
            "mape": final.test_metrics["mape"],
            "smape": final.test_metrics["smape"],
        },
        "baseline_test_metrics": final.baseline_test_metrics,
        "split_method": data.split_method,
        "experiment_config": asdict(final.experiment_config),
    }
    artifact_paths["metadata"].write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return artifact_paths


def reload_and_test_inference(
    raw_test_row: pd.DataFrame,
    artifact_paths: Mapping[str, Path],
) -> dict[str, object]:
    """Reload every artifact and perform one fit-free preprocessing/prediction pass."""
    model = VQR.from_dill(str(artifact_paths["model"]))
    loaded_x_scaler = joblib.load(artifact_paths["x_scaler"])
    loaded_y_scaler = joblib.load(artifact_paths["y_scaler"])
    loaded_selector = joblib.load(artifact_paths["selector"])
    if not isinstance(loaded_x_scaler, MinMaxScaler):
        raise TypeError("Saved x_scaler is not a MinMaxScaler.")
    if not isinstance(loaded_y_scaler, MinMaxScaler):
        raise TypeError("Saved y_scaler is not a MinMaxScaler.")
    if not isinstance(loaded_selector, SelectKBest):
        raise TypeError("Saved feature selector is not SelectKBest.")

    loaded_schema_value = json.loads(
        artifact_paths["feature_schema"].read_text(encoding="utf-8")
    )
    if not isinstance(loaded_schema_value, dict):
        raise TypeError("feature_schema.json must contain a JSON object.")
    candidate_row = transform_candidates_from_schema(raw_test_row, loaded_schema_value)
    selected_row = loaded_selector.transform(candidate_row)
    scaled_row = loaded_x_scaler.transform(selected_row)
    scaled_prediction = model.predict(scaled_row)
    prediction = float(
        loaded_y_scaler.inverse_transform(
            np.asarray(scaled_prediction).reshape(-1, 1)
        ).ravel()[0]
    )
    prediction = max(0.0, prediction)
    if not math.isfinite(prediction):
        raise ValueError("Reloaded inference produced a non-finite prediction.")

    actual = float(raw_test_row[TARGET].iloc[0]) if TARGET in raw_test_row else None
    source_index_value = (
        raw_test_row["__source_index__"].iloc[0]
        if "__source_index__" in raw_test_row
        else None
    )
    source_index = (
        int(source_index_value)
        if isinstance(source_index_value, (int, np.integer))
        else str(source_index_value)
    )
    result: dict[str, object] = {
        "model_name": MODEL_NAME,
        "target": TARGET,
        "source_index": source_index,
        "actual_units_sold": actual,
        "predicted_units_sold": prediction,
        "prediction_is_numeric": isinstance(prediction, float),
        "prediction_is_non_negative": prediction >= 0.0,
    }
    artifact_paths["test_prediction"].write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print("Clean reload inference succeeded:", result)
    return result


def plot_validation_metric(
    results: pd.DataFrame, metric: str, ylabel: str
) -> None:
    """Plot the best successful VQR validation metric at each feature count."""
    successful = results[
        (results["model_name"] == "VQR") & (results["status"] == "success")
    ]
    if successful.empty:
        print(f"No successful VQR data available for {metric}.")
        return
    aggregation = "max" if metric == "r2_score" else "min"
    summary = (
        successful.groupby("feature_count", as_index=False)[metric]
        .agg(aggregation)
        .sort_values("feature_count")
    )
    plt.figure()
    plt.plot(summary["feature_count"], summary[metric], marker="o")
    plt.xlabel("Feature count")
    plt.ylabel(ylabel)
    plt.title(f"Best VQR validation {ylabel} by feature count")
    plt.tight_layout()
    plt.show()


def plot_training_duration(results: pd.DataFrame) -> None:
    """Plot the fastest successful VQR duration at each feature count."""
    successful = results[
        (results["model_name"] == "VQR") & (results["status"] == "success")
    ]
    if successful.empty:
        print("No successful VQR timing data available.")
        return
    summary = (
        successful.groupby("feature_count", as_index=False)["training_seconds"]
        .min()
        .sort_values("feature_count")
    )
    plt.figure()
    plt.plot(summary["feature_count"], summary["training_seconds"], marker="o")
    plt.xlabel("Feature count")
    plt.ylabel("Training seconds")
    plt.title("Fastest VQR training duration by feature count")
    plt.tight_layout()
    plt.show()


def plot_loss_curve(loss_history: pd.DataFrame, experiment_id: str) -> None:
    """Plot one experiment's objective history."""
    selected = loss_history[loss_history["experiment_id"] == experiment_id]
    if selected.empty:
        print(f"No loss history recorded for {experiment_id}.")
        return
    plt.figure()
    plt.plot(selected["evaluation_number"], selected["objective_value"])
    plt.xlabel("Evaluation number")
    plt.ylabel("Objective value")
    plt.title(f"Loss curve: {experiment_id}")
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(
    actual: NDArray[np.float64], predicted: NDArray[np.float64]
) -> None:
    """Plot held-out test actuals against final VQR predictions."""
    plt.figure()
    plt.scatter(actual, predicted)
    lower = float(min(np.min(actual), np.min(predicted)))
    upper = float(max(np.max(actual), np.max(predicted)))
    plt.plot([lower, upper], [lower, upper])
    plt.xlabel("Actual units_sold")
    plt.ylabel("Predicted units_sold")
    plt.title("VQR actual versus predicted on held-out test data")
    plt.tight_layout()
    plt.show()


def run_workflow(
    config: WorkflowConfig | None = None,
    paths: WorkflowPaths | None = None,
) -> dict[str, object]:
    """Run the complete workflow when explicitly called."""
    config = config or WorkflowConfig()
    paths = paths or default_paths()

    data = load_dataset(paths.data_path)
    validate_dataset(data, config=config)
    review_target_leakage(
        data, include_spoilage_risk=config.include_spoilage_risk
    )
    raw_inputs = prepare_raw_inputs(
        data, include_spoilage_risk=config.include_spoilage_risk
    )
    raw_split = split_raw_data(raw_inputs, config)
    schema = fit_candidate_schema(
        raw_split.train,
        include_spoilage_risk=config.include_spoilage_risk,
        split_method=raw_split.method,
    )
    prepared = build_prepared_data(raw_split, schema)

    baseline_rows = run_classical_baselines(prepared, config)
    vqr_rows, loss_rows = run_vqr_experiments(prepared, config)
    results, loss_history = save_experiment_tables(
        [*baseline_rows, *vqr_rows], loss_rows, paths
    )
    ranked = rank_successful_vqr(results)
    print(ranked.head().to_string(index=False))
    winning_row = ranked.iloc[0]
    final = fit_winner_and_evaluate_test(
        prepared, winning_row, random_state=config.random_state
    )
    loss_rows.extend(final.loss_rows)
    _, loss_history = save_experiment_tables(
        [*baseline_rows, *vqr_rows], loss_rows, paths
    )
    artifact_paths = save_winning_artifacts(final, prepared, paths)

    final.model = None
    gc.collect()
    reload_result = reload_and_test_inference(
        prepared.raw_test.iloc[[0]].copy(), artifact_paths
    )
    return {
        "results": results,
        "loss_history": loss_history,
        "ranked_vqr": ranked,
        "final": final,
        "artifact_paths": artifact_paths,
        "reload_result": reload_result,
        "prepared_data": prepared,
    }


def main() -> None:
    """Run with conservative defaults when this file is explicitly executed."""
    print(
        "Starting the VQR workflow with FAST_MODE defaults. "
        "This will fit quantum models and may take substantial time."
    )
    run_workflow()


if __name__ == "__main__":
    main()
