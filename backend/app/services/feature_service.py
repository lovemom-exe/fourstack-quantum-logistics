"""Artifact-driven conversion from canonical business data to model inputs."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from app.core.exceptions import (
    FeatureResolutionError,
    ModelFeatureMismatchError,
)


def _feature_token(value: object) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_")
    return re.sub(r"_+", "_", text).lower()


class FeatureService:
    """Resolve only features named by the installed artifact schema."""

    @staticmethod
    def input_feature_order(schema: Mapping[str, object]) -> list[str]:
        value = schema.get("candidate_feature_order") or schema.get("feature_order")
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ModelFeatureMismatchError("The schema has no valid input feature order.")
        if "units_sold" in value:
            raise ModelFeatureMismatchError("units_sold cannot appear in the feature matrix.")
        return list(value)

    @staticmethod
    def selected_feature_order(schema: Mapping[str, object]) -> list[str]:
        value = schema.get("feature_order") or schema.get("selected_features")
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ModelFeatureMismatchError("The schema has no valid selected feature order.")
        count = schema.get("feature_count")
        if not isinstance(count, int) or count != len(value):
            raise ModelFeatureMismatchError(
                "Saved feature_count does not match feature_order."
            )
        if "units_sold" in value:
            raise ModelFeatureMismatchError("units_sold cannot be a model input.")
        return list(value)

    def build_frame(
        self,
        contexts: Sequence[Mapping[str, object]],
        schema: Mapping[str, object],
    ) -> pd.DataFrame:
        input_order = self.input_feature_order(schema)
        self.selected_feature_order(schema)
        rows: list[dict[str, float]] = []
        for row_number, context in enumerate(contexts, start=1):
            row: dict[str, float] = {}
            missing: list[str] = []
            for feature in input_order:
                resolved = self._resolve_feature(feature, context)
                if resolved is None:
                    missing.append(feature)
                    continue
                try:
                    numeric = float(resolved)
                except (TypeError, ValueError) as exc:
                    raise FeatureResolutionError(
                        f"Feature {feature!r} is not numeric.",
                        details={"feature": feature, "row": row_number},
                    ) from exc
                if not np.isfinite(numeric):
                    raise FeatureResolutionError(
                        f"Feature {feature!r} is not finite.",
                        details={"feature": feature, "row": row_number},
                    )
                row[feature] = numeric
            if missing:
                raise FeatureResolutionError(
                    "Required model features could not be resolved.",
                    details={"features": missing, "row": row_number},
                )
            rows.append(row)
        frame = pd.DataFrame(rows, columns=input_order)
        if list(frame.columns) != input_order:
            raise ModelFeatureMismatchError("Resolved feature order changed unexpectedly.")
        return frame

    @staticmethod
    def _resolve_feature(
        feature: str, context: Mapping[str, object]
    ) -> object | None:
        if feature == "units_sold":
            raise ModelFeatureMismatchError("units_sold cannot be a model input.")
        if feature in context and context[feature] is not None:
            return context[feature]
        if feature.startswith("category_"):
            category = context.get("category")
            if category is None:
                return None
            expected = _feature_token(feature.removeprefix("category_"))
            return float(_feature_token(category) == expected)
        if feature.startswith("region_"):
            region = context.get("region")
            if region is None:
                return None
            expected = _feature_token(feature.removeprefix("region_"))
            return float(_feature_token(region) == expected)
        return None

    def build_mock_frame(
        self, contexts: Sequence[Mapping[str, object]]
    ) -> pd.DataFrame:
        """Build a stable non-model matrix used only by the labeled mock adapter."""
        columns = [
            "shelf_life_days",
            "cost_price",
            "selling_price",
            "discount_pct",
            "is_promoted",
            "supplier_score",
            "storage_temp",
            "spoilage_sensitivity",
            "spoilage_risk",
        ]
        rows: list[dict[str, float]] = []
        for context in contexts:
            rows.append(
                {
                    column: float(context.get(column) or 0.0)
                    for column in columns
                }
            )
        return pd.DataFrame(rows, columns=columns)
