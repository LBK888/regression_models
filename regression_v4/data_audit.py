"""Column health checks shown next to each column on the Data tab.

Small scientific tables fail quietly: a typo such as a biomass of 1.75 among
values near 260, a target that is an exact multiple of another column, or two
sensors that report the same quantity. None of these stop training, but each
one changes what a model can learn. The checks here are deliberately cheap and
descriptive — they flag, they never modify data.

The functions return structured findings; the UI turns them into translated text.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ColumnFinding:
    column: str
    kind: str  # "outliers" | "constant" | "multiple_of" | "near_duplicate"
    other: str = ""
    value: float = float("nan")
    count: int = 0


def audit_columns(
    frame: pd.DataFrame,
    *,
    outlier_z: float = 8.0,
    duplicate_rho: float = 0.90,
    min_rows: int = 8,
) -> tuple[ColumnFinding, ...]:
    """Flag extreme values, constant columns, exact multiples and near-duplicates.

    * Outliers use a robust z-score (median / 1.4826·MAD), so the outlier itself
      does not inflate the yardstick. ``outlier_z=8`` only catches gross errors.
    * An exact multiple (constant ratio) means one column is a unit conversion
      of the other: as inputs they are redundant, as two targets one is enough.
    * A near-duplicate is a pair with |Spearman ρ| >= ``duplicate_rho``.
    """

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    usable = [
        column
        for column in numeric.columns
        if numeric[column].notna().sum() >= min_rows
    ]
    findings: list[ColumnFinding] = []
    for column in usable:
        series = numeric[column].dropna()
        if series.nunique() <= 1:
            findings.append(ColumnFinding(str(column), "constant", value=float(series.iloc[0])))
            continue
        median = float(series.median())
        mad = float((series - median).abs().median()) * 1.4826
        if mad > 0:
            deviation = (series - median).abs()
            robust_z = deviation / mad
            # A tightly controlled variable (water temperature, say) can sit many
            # MADs away for a perfectly real reading, so also require the value to
            # be far from the median in relative terms.
            relative = deviation > 0.5 * abs(median) if median != 0 else deviation > 0
            extreme = series[(robust_z > outlier_z) & relative]
            if len(extreme):
                findings.append(
                    ColumnFinding(str(column), "outliers", value=float(extreme.iloc[0]), count=int(len(extreme)))
                )

    varied = [column for column in usable if numeric[column].nunique() > 2]
    for index, left in enumerate(varied):
        for right in varied[index + 1 :]:
            pair = numeric[[left, right]].dropna()
            if len(pair) < min_rows:
                continue
            denominator = pair[right].to_numpy(dtype=float)
            numerator = pair[left].to_numpy(dtype=float)
            nonzero = np.abs(denominator) > np.finfo(float).eps
            if nonzero.sum() >= min_rows:
                ratio = numerator[nonzero] / denominator[nonzero]
                mean_ratio = float(np.mean(ratio))
                if np.isfinite(mean_ratio) and mean_ratio != 0 and float(np.std(ratio)) <= 1e-6 * abs(mean_ratio):
                    findings.append(ColumnFinding(str(left), "multiple_of", other=str(right), value=mean_ratio))
                    continue
            rho = float(pair[left].corr(pair[right], method="spearman"))
            if np.isfinite(rho) and abs(rho) >= duplicate_rho:
                findings.append(ColumnFinding(str(right), "near_duplicate", other=str(left), value=rho))
    return tuple(findings)
