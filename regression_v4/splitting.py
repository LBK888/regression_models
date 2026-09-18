"""Leakage-aware observation identity and deterministic split generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split


@dataclass(frozen=True)
class DatasetIdentity:
    observation_ids: np.ndarray
    group_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        observations = np.asarray(self.observation_ids, dtype=object).reshape(-1)
        if observations.size == 0:
            raise ValueError("At least one observation ID is required")
        if len(set(observations.tolist())) != observations.size:
            raise ValueError("Observation IDs must be unique")
        object.__setattr__(self, "observation_ids", observations)
        if self.group_ids is not None:
            groups = np.asarray(self.group_ids, dtype=object).reshape(-1)
            if groups.size != observations.size:
                raise ValueError("group_ids must have the same length as observation_ids")
            object.__setattr__(self, "group_ids", groups)

    @property
    def size(self) -> int:
        return int(self.observation_ids.size)

    @property
    def is_grouped(self) -> bool:
        return self.group_ids is not None


@dataclass(frozen=True)
class IndexSplit:
    train: np.ndarray
    test: np.ndarray


def holdout_split(identity: DatasetIdentity, test_size: float, random_seed: int) -> IndexSplit:
    """Create a holdout split; a zero test ratio deliberately returns no test rows."""

    if not 0.0 <= test_size <= 0.50:
        raise ValueError("test_size must be between 0 and 0.50")
    indexes = np.arange(identity.size, dtype=int)
    if test_size == 0:
        return IndexSplit(train=indexes, test=np.empty(0, dtype=int))
    if identity.group_ids is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
        train, test = next(splitter.split(indexes, groups=identity.group_ids))
    else:
        train, test = train_test_split(indexes, test_size=test_size, random_state=random_seed, shuffle=True)
    return IndexSplit(train=np.asarray(train, dtype=int), test=np.asarray(test, dtype=int))


def cross_validation_splits(
    identity: DatasetIdentity,
    folds: int,
    random_seed: int,
) -> Iterator[IndexSplit]:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    indexes = np.arange(identity.size, dtype=int)
    if identity.group_ids is not None:
        unique_groups = np.unique(identity.group_ids)
        if unique_groups.size < folds:
            raise ValueError(f"Need at least {folds} unique groups, found {unique_groups.size}")
        try:
            splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=random_seed)
        except TypeError:  # Compatibility with older scikit-learn releases.
            rng = np.random.default_rng(random_seed)
            shuffled_groups = unique_groups.copy()
            rng.shuffle(shuffled_groups)
            order = {value: index for index, value in enumerate(shuffled_groups)}
            sortable = np.asarray([order[value] for value in identity.group_ids])
            splitter = GroupKFold(n_splits=folds)
            indexes = np.argsort(sortable, kind="stable")
        iterator = splitter.split(indexes, groups=identity.group_ids[indexes])
    else:
        if identity.size < folds:
            raise ValueError(f"Need at least {folds} observations, found {identity.size}")
        iterator = KFold(n_splits=folds, shuffle=True, random_state=random_seed).split(indexes)
    for train, test in iterator:
        yield IndexSplit(train=indexes[np.asarray(train, dtype=int)], test=indexes[np.asarray(test, dtype=int)])


def assert_no_group_leakage(split: IndexSplit, identity: DatasetIdentity) -> None:
    if identity.group_ids is None:
        return
    train_groups = set(identity.group_ids[split.train].tolist())
    test_groups = set(identity.group_ids[split.test].tolist())
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise AssertionError(f"Groups occur in both train and test: {sorted(overlap)!r}")
