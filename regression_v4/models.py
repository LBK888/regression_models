"""Bounded V4 model registry and factories for the core release."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import importlib.util
import copy
import re
from typing import Any, Callable

import torch
from torch import nn
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR


V4_TORCH_ARCHITECTURES = {
    "CNN1D",
    "ResNet1D",
    "GroupedFusion",
    "NumericalEmbeddingMLP",
    "FTTransformer",
    "TabM",
}

NEURAL_MODEL_IDS = {
    "deep_mlp",
    "wide_mlp",
    "residual_mlp",
    "multi_branch_mlp",
    "bottleneck_mlp",
    "cnn1d",
    "resnet1d",
    "grouped_fusion",
    "mlp_embeddings",
    "ft_transformer",
    "tabm",
    "realmlp",
    "modern_nca",
}

LOSS_CONFIGURABLE_MODEL_IDS = NEURAL_MODEL_IDS - {"realmlp", "modern_nca"}


def infer_feature_groups(input_columns: tuple[str, ...]) -> tuple[tuple[int, ...], ...]:
    grouped: dict[str, list[int]] = {}
    for index, column in enumerate(input_columns):
        group = column.rsplit("::", 1)[0] if "::" in column else "__ungrouped__"
        grouped.setdefault(group, []).append(index)
    return tuple(tuple(indexes) for indexes in grouped.values())


def has_single_ordered_numeric_axis(input_columns: tuple[str, ...]) -> bool:
    if len(input_columns) < 4 or len(infer_feature_groups(input_columns)) != 1:
        return False
    coordinates: list[float] = []
    for column in input_columns:
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:nm)?$", column, re.IGNORECASE)
        if match is None:
            return False
        coordinates.append(float(match.group(1)))
    return len(set(coordinates)) == len(coordinates) and all(
        right > left for left, right in zip(coordinates, coordinates[1:])
    )


def model_incompatibility_reason(
    model_id: str,
    input_columns: tuple[str, ...],
    feature_structure: str | None = None,
    structure_confirmed: bool | None = None,
) -> str | None:
    if model_id in {"cnn1d", "resnet1d", "grouped_fusion"} and structure_confirmed is not True:
        return "requires user confirmation of the Feature Structure"
    if model_id in {"cnn1d", "resnet1d"} and len(input_columns) < 4:
        return "requires at least four features in one confirmed ordered 1D sequence"
    if model_id in {"cnn1d", "resnet1d"} and len(infer_feature_groups(input_columns)) != 1:
        return "requires one Feature Group; use Grouped Fusion for multiple groups"
    if model_id in {"cnn1d", "resnet1d"} and feature_structure != "ordered_1d":
        return "requires user-confirmed Feature Structure 'ordered_1d'; spectral naming is not required"
    if model_id == "grouped_fusion" and len(infer_feature_groups(input_columns)) < 2:
        return "requires at least two Feature Groups"
    if model_id == "grouped_fusion" and feature_structure not in {
        None,
        "grouped",
        "grouped_ordered_1d",
    }:
        return "requires Feature Structure 'grouped' or 'grouped_ordered_1d'"
    return None


class CNN1DNetwork(nn.Module):
    def __init__(self, n_features: int, n_outputs: int, channels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(channels, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x.unsqueeze(1)).squeeze(-1))


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ResNet1DNetwork(nn.Module):
    def __init__(self, n_features: int, n_outputs: int, channels: int, blocks: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, channels, 5, padding=2), nn.GELU())
        self.blocks = nn.Sequential(
            *(ResidualBlock1D(channels, 1 if index % 2 == 0 else 2) for index in range(blocks))
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.blocks(self.stem(x.unsqueeze(1)))
        return self.head(self.pool(hidden).squeeze(-1))


class GroupedFusionNetwork(nn.Module):
    def __init__(
        self,
        groups: tuple[tuple[int, ...], ...],
        n_outputs: int,
        group_width: int,
        fusion_width: int,
    ) -> None:
        super().__init__()
        self.group_indexes = groups
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(len(indexes), group_width),
                    nn.GELU(),
                    nn.Linear(group_width, group_width),
                    nn.GELU(),
                )
                for indexes in groups
            ]
        )
        self.fusion = nn.Sequential(
            nn.Linear(group_width * len(groups), fusion_width),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_width, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = [encoder(x[:, indexes]) for encoder, indexes in zip(self.encoders, self.group_indexes)]
        return self.fusion(torch.cat(encoded, dim=1))


class NumericalEmbeddingMLPNetwork(nn.Module):
    def __init__(self, n_features: int, n_outputs: int, embedding_dim: int, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, embedding_dim))
        self.bias = nn.Parameter(torch.empty(n_features, embedding_dim))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.bias, std=0.02)
        self.network = nn.Sequential(
            nn.Linear(n_features * embedding_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = x.unsqueeze(-1) * self.weight + self.bias
        return self.network(embedded.flatten(1))


class FTTransformerNetwork(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        token_dim: int,
        heads: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.feature_weight = nn.Parameter(torch.empty(n_features, token_dim))
        self.feature_bias = nn.Parameter(torch.empty(n_features, token_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        nn.init.normal_(self.feature_weight, std=0.02)
        nn.init.normal_(self.feature_bias, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=token_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.normalization = nn.LayerNorm(token_dim)
        self.head = nn.Linear(token_dim, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.feature_weight + self.feature_bias
        cls = self.cls_token.expand(len(x), -1, -1)
        encoded = self.encoder(torch.cat((cls, tokens), dim=1))
        return self.head(self.normalization(encoded[:, 0]))


def _build_v4_torch_model(
    architecture_id: str,
    n_features: int,
    n_outputs: int,
    multiplier: float,
    groups: tuple[tuple[int, ...], ...],
    parameters: dict[str, Any],
) -> nn.Module:
    width = max(8, round(32 * multiplier))
    if architecture_id == "CNN1D":
        return CNN1DNetwork(n_features, n_outputs, int(parameters.get("channels", width)))
    if architecture_id == "ResNet1D":
        return ResNet1DNetwork(
            n_features,
            n_outputs,
            int(parameters.get("channels", width)),
            int(parameters.get("blocks", 2)),
        )
    if architecture_id == "GroupedFusion":
        if len(groups) < 2:
            raise ValueError("Grouped Fusion requires at least two Feature Groups")
        return GroupedFusionNetwork(
            groups,
            n_outputs,
            int(parameters.get("group_width", width)),
            int(parameters.get("fusion_width", width * 2)),
        )
    if architecture_id == "NumericalEmbeddingMLP":
        return NumericalEmbeddingMLPNetwork(
            n_features,
            n_outputs,
            int(parameters.get("embedding_dim", 8)),
            int(parameters.get("hidden", width * 2)),
        )
    if architecture_id == "FTTransformer":
        token_dim = int(parameters.get("token_dim", 16))
        heads = int(parameters.get("heads", 4))
        if token_dim % heads:
            raise ValueError("FT-Transformer token_dim must be divisible by heads")
        return FTTransformerNetwork(
            n_features,
            n_outputs,
            token_dim,
            heads,
            int(parameters.get("layers", 2)),
        )
    if architecture_id == "TabM":
        from tabm import TabM

        return TabM.make(
            n_num_features=n_features,
            d_out=n_outputs,
            k=int(parameters.get("k", 32)),
            n_blocks=int(parameters.get("n_blocks", 2)),
            d_block=int(parameters.get("d_block", max(64, width * 2))),
        )
    raise KeyError(architecture_id)


class ModelAvailability(str, Enum):
    CORE = "core"
    OPTIONAL_AVAILABLE = "optional_available"
    OPTIONAL_MISSING = "optional_missing"
    COMING_LATER = "coming_later"


@dataclass(frozen=True)
class ModelCapability:
    model_id: str
    display_name: str
    availability: ModelAvailability
    reason: str
    factory: Callable[[int, int], Any] | None = None

    @property
    def enabled(self) -> bool:
        return self.factory is not None and self.availability in {
            ModelAvailability.CORE,
            ModelAvailability.OPTIONAL_AVAILABLE,
        }


class LegacyTorchRegressor:
    """Small sklearn-like adapter around the five V3 PyTorch architectures."""

    def __init__(
        self,
        architecture_id: str,
        random_seed: int,
        *,
        epochs: int = 500,
        patience: int = 30,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_ratio: float = 0.20,
        model_multiplier: float = 1.0,
        loss_id: str = "mse",
    ) -> None:
        self.architecture_id = architecture_id
        self.random_seed = random_seed
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_ratio = validation_ratio
        self.model_multiplier = model_multiplier
        self.loss_id = loss_id
        self.input_columns: tuple[str, ...] = ()
        self.feature_structure: str | None = None
        self.structure_confirmed: bool | None = None
        self.architecture_parameters: dict[str, Any] = {}
        self.model_: Any = None
        self.trained_epochs_: int | None = None

    def fit(self, X: Any, y: Any) -> "LegacyTorchRegressor":
        import numpy as np
        from sklearn.model_selection import train_test_split
        import torch
        from torch import nn, optim
        from torch.utils.data import DataLoader, TensorDataset
        from regression_core import (
            ArchitectureSpec,
            DEVICE,
            EarlyStopping,
            _safe_batch_size,
            build_model,
            design_architectures,
            seed_everything,
        )

        features = np.asarray(X, dtype=np.float32)
        targets = np.asarray(y, dtype=np.float32)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        if len(features) < 5:
            raise ValueError("Legacy PyTorch models require at least five fold-training rows")
        train_index, validation_index = train_test_split(
            np.arange(len(features)), test_size=self.validation_ratio, random_state=self.random_seed
        )
        seed_everything(self.random_seed, deterministic=True)
        if self.architecture_id == "FTTransformer" and torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        if self.architecture_id in V4_TORCH_ARCHITECTURES:
            incompatibility = model_incompatibility_reason(
                {
                    "CNN1D": "cnn1d",
                    "ResNet1D": "resnet1d",
                    "GroupedFusion": "grouped_fusion",
                }.get(self.architecture_id, ""),
                self.input_columns,
                self.feature_structure,
                self.structure_confirmed,
            )
            if incompatibility:
                raise ValueError(f"{self.architecture_id} {incompatibility}")
            model = _build_v4_torch_model(
                self.architecture_id,
                features.shape[1],
                targets.shape[1],
                self.model_multiplier,
                infer_feature_groups(self.input_columns),
                self.architecture_parameters,
            ).to(DEVICE)
        else:
            designed = design_architectures(
                features.shape[1], targets.shape[1], len(train_index), (self.architecture_id,), self.model_multiplier
            )
            architecture = designed.get(self.architecture_id, ArchitectureSpec(self.architecture_id, {}))
            model = build_model(architecture, features.shape[1], targets.shape[1]).to(DEVICE)
        train_data = TensorDataset(torch.from_numpy(features[train_index]), torch.from_numpy(targets[train_index]))
        validation_X = torch.from_numpy(features[validation_index]).to(DEVICE)
        validation_y = torch.from_numpy(targets[validation_index]).to(DEVICE)
        generator = torch.Generator().manual_seed(self.random_seed)
        loader = DataLoader(
            train_data,
            batch_size=_safe_batch_size(len(train_data), self.batch_size),
            shuffle=True,
            generator=generator,
        )
        if self.loss_id == "mse":
            criterion = nn.MSELoss()
        elif self.loss_id == "huber":
            criterion = nn.SmoothL1Loss()
        elif self.loss_id == "mae":
            criterion = nn.L1Loss()
        elif self.loss_id == "log_cosh":
            criterion = lambda predicted, actual: torch.mean(
                predicted - actual + torch.nn.functional.softplus(-2.0 * (predicted - actual)) - np.log(2.0)
            )
        else:
            raise ValueError(f"Unsupported neural loss: {self.loss_id}")
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        stopper = EarlyStopping(patience=self.patience)
        for epoch in range(self.epochs):
            model.train()
            for batch_X, batch_y in loader:
                if len(batch_X) == 1:
                    continue
                optimizer.zero_grad(set_to_none=True)
                predicted = model(batch_X.to(DEVICE))
                expected = batch_y.to(DEVICE)
                if predicted.ndim == 3:
                    expected = expected[:, None, :].expand_as(predicted)
                loss = criterion(predicted, expected)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                predicted = model(validation_X)
                expected = validation_y
                if predicted.ndim == 3:
                    expected = expected[:, None, :].expand_as(predicted)
                validation_loss = float(criterion(predicted, expected).item())
            stopper(validation_loss, model, epoch)
            if stopper.early_stop:
                break
        if stopper.best_model_state is None:
            raise RuntimeError("Neural training did not capture a valid model state")
        model.load_state_dict(stopper.best_model_state)
        self.model_ = model
        self.trained_epochs_ = epoch + 1
        return self

    def predict(self, X: Any) -> Any:
        if self.model_ is None:
            raise RuntimeError("The model has not been fit")
        import numpy as np
        import torch
        from regression_core import DEVICE

        self.model_.eval()
        with torch.no_grad():
            output = self.model_(torch.from_numpy(np.asarray(X, dtype=np.float32)).to(DEVICE))
        if output.ndim == 3:
            output = output.mean(dim=1)
        prediction = output.detach().cpu().numpy()
        return prediction.ravel() if prediction.shape[1] == 1 else prediction


class RealMLPRegressorAdapter(BaseEstimator, RegressorMixin):
    """Thin sklearn-compatible wrapper around pytabkit's tuned-default RealMLP."""

    def __init__(
        self,
        random_seed: int = 42,
        n_outputs: int = 1,
        epochs: int = 256,
        batch_size: int = 256,
        learning_rate: float = 0.04,
    ) -> None:
        self.random_seed = random_seed
        self.n_outputs = n_outputs
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_: Any = None

    def fit(self, X: Any, y: Any) -> "RealMLPRegressorAdapter":
        import numpy as np
        from pytabkit import RealMLP_TD_Regressor

        target = np.asarray(y)
        if self.n_outputs == 1 and target.ndim == 2:
            target = target.ravel()
        self.model_ = RealMLP_TD_Regressor(
            random_state=self.random_seed,
            n_cv=1,
            n_refit=0,
            n_epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.learning_rate,
            verbosity=0,
        )
        self.model_.fit(X, target)
        return self

    def predict(self, X: Any) -> Any:
        if self.model_ is None:
            raise RuntimeError("The model has not been fit")
        return self.model_.predict(X)


class ModernNCAEmbeddingNetwork(nn.Module):
    """Deep projection used by the differentiable neighbor regressor."""

    def __init__(self, n_features: int, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.network(x), dim=1)


class ModernNCARegressor(BaseEstimator, RegressorMixin):
    """ModernNCA-style deep embedding with stochastic-neighbor regression.

    Training uses a randomly sampled neighbor pool and a differentiable soft
    nearest-neighbor objective. Prediction uses the full development reference
    neighborhood, matching the central train/inference distinction of ModernNCA.
    """

    def __init__(
        self,
        random_seed: int = 42,
        n_outputs: int = 1,
        epochs: int = 500,
        patience: int = 30,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        validation_ratio: float = 0.20,
        embedding_dim: int = 16,
        hidden_dim: int = 64,
        temperature: float = 0.20,
        neighbor_sample_size: int = 256,
        weight_decay: float = 0.0001,
    ) -> None:
        self.random_seed = random_seed
        self.n_outputs = n_outputs
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_ratio = validation_ratio
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.neighbor_sample_size = neighbor_sample_size
        self.weight_decay = weight_decay
        self.model_: ModernNCAEmbeddingNetwork | None = None
        self.reference_X_: Any = None
        self.reference_y_: Any = None
        self.trained_epochs_: int | None = None

    @staticmethod
    def _neighbor_prediction(
        query_embedding: torch.Tensor,
        reference_embedding: torch.Tensor,
        reference_y: torch.Tensor,
        temperature: float,
        self_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        distances = torch.cdist(query_embedding, reference_embedding).square()
        if self_mask is not None:
            distances = distances.masked_fill(self_mask, torch.inf)
        weights = torch.softmax(-distances / max(float(temperature), 1e-4), dim=1)
        return weights @ reference_y

    def fit(self, X: Any, y: Any) -> "ModernNCARegressor":
        import numpy as np
        from sklearn.model_selection import train_test_split
        from regression_core import DEVICE, seed_everything

        features = np.asarray(X, dtype=np.float32)
        targets = np.asarray(y, dtype=np.float32)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        if features.ndim != 2 or targets.ndim != 2 or len(features) != len(targets):
            raise ValueError("ModernNCA expects aligned two-dimensional X/y")
        if len(features) < 6:
            raise ValueError("ModernNCA requires at least six fold-training rows")
        if self.temperature <= 0 or self.neighbor_sample_size < 2:
            raise ValueError("ModernNCA temperature and neighbor_sample_size must be positive")

        train_index, validation_index = train_test_split(
            np.arange(len(features)),
            test_size=self.validation_ratio,
            random_state=self.random_seed,
        )
        seed_everything(self.random_seed, deterministic=True)
        model = ModernNCAEmbeddingNetwork(
            features.shape[1],
            int(self.embedding_dim),
            int(self.hidden_dim),
        ).to(DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        feature_tensor = torch.from_numpy(features).to(DEVICE)
        target_tensor = torch.from_numpy(targets).to(DEVICE)
        train_tensor = torch.as_tensor(train_index, dtype=torch.long, device=DEVICE)
        validation_tensor = torch.as_tensor(validation_index, dtype=torch.long, device=DEVICE)
        generator = torch.Generator(device="cpu").manual_seed(self.random_seed)
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0

        for _epoch in range(int(self.epochs)):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            shuffled = train_index[
                torch.randperm(len(train_index), generator=generator).cpu().numpy()
            ]
            pool_size = min(len(shuffled), int(self.neighbor_sample_size))
            pool_numpy = shuffled[:pool_size]
            pool_tensor = torch.as_tensor(pool_numpy, dtype=torch.long, device=DEVICE)
            anchor_embedding = model(feature_tensor[train_tensor])
            neighbor_embedding = model(feature_tensor[pool_tensor])
            self_mask = train_tensor[:, None] == pool_tensor[None, :]
            prediction = self._neighbor_prediction(
                anchor_embedding,
                neighbor_embedding,
                target_tensor[pool_tensor],
                self.temperature,
                self_mask,
            )
            loss = torch.nn.functional.mse_loss(prediction, target_tensor[train_tensor])
            if not torch.isfinite(loss):
                raise RuntimeError("ModernNCA training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                train_embedding = model(feature_tensor[train_tensor])
                validation_embedding = model(feature_tensor[validation_tensor])
                validation_prediction = self._neighbor_prediction(
                    validation_embedding,
                    train_embedding,
                    target_tensor[train_tensor],
                    self.temperature,
                )
                validation_loss = float(
                    torch.nn.functional.mse_loss(
                        validation_prediction, target_tensor[validation_tensor]
                    ).item()
                )
            if validation_loss < best_loss - 1e-8:
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= int(self.patience):
                    break
        if best_state is None:
            raise RuntimeError("ModernNCA did not capture a valid model state")
        model.load_state_dict(best_state)
        self.model_ = model
        self.reference_X_ = features.copy()
        self.reference_y_ = targets.copy()
        self.trained_epochs_ = _epoch + 1
        return self

    def predict(self, X: Any) -> Any:
        import numpy as np
        from regression_core import DEVICE

        if self.model_ is None or self.reference_X_ is None or self.reference_y_ is None:
            raise RuntimeError("The model has not been fit")
        features = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            reference_X = torch.from_numpy(self.reference_X_).to(DEVICE)
            reference_y = torch.from_numpy(self.reference_y_).to(DEVICE)
            reference_embedding = self.model_(reference_X)
            predictions: list[torch.Tensor] = []
            for start in range(0, len(features), max(1, int(self.batch_size))):
                query = torch.from_numpy(features[start : start + int(self.batch_size)]).to(DEVICE)
                query_embedding = self.model_(query)
                predictions.append(
                    self._neighbor_prediction(
                        query_embedding,
                        reference_embedding,
                        reference_y,
                        self.temperature,
                    ).cpu()
                )
        output = torch.cat(predictions, dim=0).numpy()
        return output.ravel() if output.shape[1] == 1 else output


def _single_or_multi(factory: Callable[[], Any], n_outputs: int) -> Any:
    estimator = factory()
    return estimator if n_outputs == 1 else MultiOutputRegressor(estimator)


def _core_capabilities() -> list[ModelCapability]:
    return [
        ModelCapability("mean", "Mean / Dummy baseline", ModelAvailability.CORE, "Required baseline", lambda seed, outputs: DummyRegressor(strategy="mean")),
        ModelCapability("ridge", "Ridge", ModelAvailability.CORE, "Linear regularized baseline", lambda seed, outputs: Ridge(alpha=1.0)),
        ModelCapability("pls", "PLS", ModelAvailability.CORE, "Small-data latent-component model", lambda seed, outputs: PLSRegression(n_components=1, scale=False)),
        ModelCapability("svr_rbf", "SVR (RBF)", ModelAvailability.CORE, "Nonlinear small-data baseline", lambda seed, outputs: _single_or_multi(lambda: SVR(C=10.0, epsilon=0.1, kernel="rbf"), outputs)),
        ModelCapability("random_forest", "Random Forest", ModelAvailability.CORE, "Tree ensemble baseline", lambda seed, outputs: RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1)),
        ModelCapability("extra_trees", "Extra Trees", ModelAvailability.CORE, "Randomized tree ensemble baseline", lambda seed, outputs: ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1)),
        ModelCapability("deep_mlp", "Deep MLP", ModelAvailability.CORE, "V3 Deep_FFN architecture with corrected display name", lambda seed, outputs: LegacyTorchRegressor("Deep_FFN", seed)),
        ModelCapability("wide_mlp", "Wide MLP", ModelAvailability.CORE, "V3 Wide_Network architecture with corrected display name", lambda seed, outputs: LegacyTorchRegressor("Wide_Network", seed)),
        ModelCapability("residual_mlp", "Residual MLP", ModelAvailability.CORE, "V3 ResNet architecture with corrected display name", lambda seed, outputs: LegacyTorchRegressor("ResNet", seed)),
        ModelCapability("multi_branch_mlp", "Multi-branch MLP", ModelAvailability.CORE, "V3 Ensemble_Net architecture with corrected display name", lambda seed, outputs: LegacyTorchRegressor("Ensemble_Net", seed)),
        ModelCapability("bottleneck_mlp", "Bottleneck MLP", ModelAvailability.CORE, "V3 AutoEncoder_Net architecture with corrected display name", lambda seed, outputs: LegacyTorchRegressor("AutoEncoder_Net", seed)),
        ModelCapability("cnn1d", "CNN1D", ModelAvailability.CORE, "Ordered-axis convolutional regressor; only compatible experiments are executed", lambda seed, outputs: LegacyTorchRegressor("CNN1D", seed)),
        ModelCapability("resnet1d", "ResNet1D", ModelAvailability.CORE, "Residual ordered-axis convolutional regressor; only compatible experiments are executed", lambda seed, outputs: LegacyTorchRegressor("ResNet1D", seed)),
        ModelCapability("grouped_fusion", "Grouped Fusion", ModelAvailability.CORE, "Independent Feature Group encoders followed by learned fusion", lambda seed, outputs: LegacyTorchRegressor("GroupedFusion", seed)),
        ModelCapability("mlp_embeddings", "Learned Numerical-Embedding MLP", ModelAvailability.CORE, "Per-feature learned numerical embeddings followed by an MLP", lambda seed, outputs: LegacyTorchRegressor("NumericalEmbeddingMLP", seed)),
        ModelCapability("ft_transformer", "FT-Transformer", ModelAvailability.CORE, "Feature-token Transformer for numerical tabular inputs", lambda seed, outputs: LegacyTorchRegressor("FTTransformer", seed)),
        ModelCapability("modern_nca", "ModernNCA", ModelAvailability.CORE, "Deep learned embedding with stochastic-neighbor training and full-neighborhood regression", lambda seed, outputs: ModernNCARegressor(seed, outputs)),
    ]


def _optional_capability(model_id: str, display: str, package: str, reason: str) -> ModelCapability:
    installed = importlib.util.find_spec(package) is not None
    availability = ModelAvailability.OPTIONAL_AVAILABLE if installed else ModelAvailability.OPTIONAL_MISSING
    suffix = "Adapter discovery succeeded; implementation gate remains" if installed else f"Install optional package '{package}'"
    return ModelCapability(model_id, display, availability, f"{reason}. {suffix}", None)


def _tabm_capability() -> ModelCapability:
    if importlib.util.find_spec("tabm") is None:
        return ModelCapability(
            "tabm", "TabM", ModelAvailability.OPTIONAL_MISSING,
            "Official TabM adapter; install optional package 'tabm'", None,
        )
    return ModelCapability(
        "tabm", "TabM", ModelAvailability.OPTIONAL_AVAILABLE,
        "Official tabm.TabM parameter-efficient ensemble adapter",
        lambda seed, outputs: LegacyTorchRegressor("TabM", seed, learning_rate=0.002),
    )


def _realmlp_capability() -> ModelCapability:
    if importlib.util.find_spec("pytabkit") is None:
        return ModelCapability(
            "realmlp", "RealMLP", ModelAvailability.OPTIONAL_MISSING,
            "Install optional package 'pytabkit' to activate the RealMLP adapter", None,
        )
    return ModelCapability(
        "realmlp", "RealMLP", ModelAvailability.OPTIONAL_AVAILABLE,
        "pytabkit RealMLP_TD_Regressor adapter",
        lambda seed, outputs: RealMLPRegressorAdapter(seed, outputs),
    )


def _catboost_capability() -> ModelCapability:
    if importlib.util.find_spec("catboost") is None:
        return ModelCapability(
            "catboost", "CatBoost", ModelAvailability.OPTIONAL_MISSING,
            "Install optional package 'catboost' to activate the adapter", None,
        )

    def factory(seed: int, outputs: int) -> Any:
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="MultiRMSE" if outputs > 1 else "RMSE",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )

    return ModelCapability(
        "catboost", "CatBoost", ModelAvailability.OPTIONAL_AVAILABLE,
        "Native CatBoostRegressor adapter", factory,
    )


def _xgboost_capability() -> ModelCapability:
    if importlib.util.find_spec("xgboost") is None:
        return ModelCapability(
            "xgboost", "XGBoost", ModelAvailability.OPTIONAL_MISSING,
            "Install optional package 'xgboost' to activate the adapter", None,
        )

    def factory(seed: int, outputs: int) -> Any:
        from xgboost import XGBRegressor

        return _single_or_multi(
            lambda: XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=seed,
                n_jobs=-1,
            ),
            outputs,
        )

    return ModelCapability(
        "xgboost", "XGBoost", ModelAvailability.OPTIONAL_AVAILABLE,
        "XGBRegressor adapter with explicit multi-output wrapping", factory,
    )


def _lightgbm_capability() -> ModelCapability:
    if importlib.util.find_spec("lightgbm") is None:
        return ModelCapability(
            "lightgbm", "LightGBM", ModelAvailability.OPTIONAL_MISSING,
            "Install optional package 'lightgbm' to activate the adapter", None,
        )

    def factory(seed: int, outputs: int) -> Any:
        from lightgbm import LGBMRegressor

        return _single_or_multi(
            lambda: LGBMRegressor(
                n_estimators=300,
                num_leaves=15,
                learning_rate=0.05,
                min_child_samples=10,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=-1,
                verbosity=-1,
            ),
            outputs,
        )

    return ModelCapability(
        "lightgbm", "LightGBM", ModelAvailability.OPTIONAL_AVAILABLE,
        "LGBMRegressor adapter with explicit multi-output wrapping", factory,
    )


def model_registry() -> tuple[ModelCapability, ...]:
    models = _core_capabilities()
    models.extend(
        [
            _tabm_capability(),
            _realmlp_capability(),
            _catboost_capability(),
            _xgboost_capability(),
            _lightgbm_capability(),
        ]
    )
    return tuple(models)


def model_configuration(model_id: str) -> str:
    """Human-readable screening preset included in reports and CSV output."""

    return {
        "mean": "strategy=mean",
        "ridge": "alpha=1.0",
        "pls": "n_components=1, scale=False (fold-local scaling)",
        "svr_rbf": "kernel=rbf, C=10, epsilon=0.1",
        "random_forest": "n_estimators=300, min_samples_leaf=2",
        "extra_trees": "n_estimators=300, min_samples_leaf=2",
        "deep_mlp": "V3 Deep_FFN, epochs<=500, patience=30",
        "wide_mlp": "V3 Wide_Network, epochs<=500, patience=30",
        "residual_mlp": "V3 ResNet, epochs<=500, patience=30",
        "multi_branch_mlp": "V3 Ensemble_Net, epochs<=500, patience=30",
        "bottleneck_mlp": "V3 AutoEncoder_Net, epochs<=500, patience=30",
        "cnn1d": "two Conv1D layers, global pooling, ordered-axis only",
        "resnet1d": "dilated residual Conv1D blocks, global pooling, ordered-axis only",
        "grouped_fusion": "per-Feature-Group encoders with learned late fusion",
        "mlp_embeddings": "learned per-feature numerical embeddings + MLP",
        "ft_transformer": "numerical feature tokens + Transformer encoder",
        "modern_nca": "deep embedding, stochastic neighbor pool during training, full reference neighborhood at inference",
        "tabm": "official TabM defaults, k=32 when optional package is installed",
        "realmlp": "pytabkit RealMLP tuned-default regressor, n_cv=1, n_refit=0",
        "catboost": "iterations=300, depth=6, learning_rate=0.05",
        "xgboost": "n_estimators=300, max_depth=4, learning_rate=0.05",
        "lightgbm": "n_estimators=300, num_leaves=15, learning_rate=0.05",
    }.get(model_id, "adapter default preset")


def create_model(
    model_id: str,
    random_seed: int,
    n_outputs: int,
    training_options: dict[str, Any] | None = None,
    model_parameters: dict[str, Any] | None = None,
) -> Any:
    capability = next((model for model in model_registry() if model.model_id == model_id), None)
    if capability is None:
        raise KeyError(model_id)
    if not capability.enabled:
        raise RuntimeError(f"{capability.display_name} is not active: {capability.reason}")
    assert capability.factory is not None
    model = capability.factory(random_seed, n_outputs)
    if isinstance(model, LegacyTorchRegressor) and training_options:
        for name in (
            "epochs",
            "patience",
            "batch_size",
            "learning_rate",
            "validation_ratio",
            "model_multiplier",
            "loss_id",
            "input_columns",
            "feature_structure",
            "structure_confirmed",
        ):
            if name in training_options:
                setattr(model, name, training_options[name])
        if model_parameters:
            model.architecture_parameters.update(model_parameters)
    else:
        if isinstance(model, (RealMLPRegressorAdapter, ModernNCARegressor)) and training_options:
            for source, destination in (
                ("epochs", "epochs"),
                ("patience", "patience"),
                ("batch_size", "batch_size"),
                ("learning_rate", "learning_rate"),
                ("validation_ratio", "validation_ratio"),
            ):
                if source in training_options and hasattr(model, destination):
                    setattr(model, destination, training_options[source])
        if model_parameters:
            if not hasattr(model, "set_params"):
                raise ValueError(f"{capability.display_name} does not expose tunable parameters")
            available = model.get_params(deep=True)
            resolved = {
                (name if name in available else f"estimator__{name}"): value
                for name, value in model_parameters.items()
            }
            model.set_params(**resolved)
    return model
