from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC


def available_models(model_names: list[str]) -> list[str]:
    return list(model_names)


def seed_everything(random_seed: int, deterministic: bool = False) -> None:
    """
    Notebook-style global seed initialization.

    Seeds are intentionally set once per script run (not per model fit) so
    sequential trainings consume RNG state in the same way as notebooks.
    """
    seed = int(random_seed)
    np.random.seed(seed)
    try:
        import torch
    except Exception:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass


def _import_torch_modules():
    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import OneCycleLR
    from torch.utils.data import DataLoader, TensorDataset

    return torch, nn, OneCycleLR, DataLoader, TensorDataset


@dataclass
class TorchRegularizedHybridNetClassifier:
    random_seed: int
    hidden_dims: tuple[int, ...]
    dropout_rate: float
    lr: float
    weight_decay: float
    epochs: int
    batch_size: int
    patience: int
    min_delta: float
    noise_std: float
    label_smoothing: float
    gradient_clip: float
    pct_start: float
    validation_fraction: float
    device: str
    use_batch_norm: bool

    def __init__(self, random_seed: int, **hyperparams: Any):
        hidden_dims = hyperparams.get("hidden_dims", hyperparams.get("hidden_layer_sizes", [256, 128, 64]))
        self.random_seed = int(random_seed)
        self.hidden_dims = tuple(int(v) for v in hidden_dims)
        self.dropout_rate = float(hyperparams.get("dropout_rate", 0.3))
        self.lr = float(hyperparams.get("lr", hyperparams.get("learning_rate_init", 6e-4)))
        self.weight_decay = float(hyperparams.get("weight_decay", hyperparams.get("alpha", 8e-3)))
        self.epochs = int(hyperparams.get("epochs", hyperparams.get("max_iter", 40)))
        self.batch_size = int(hyperparams.get("batch_size", 128))
        self.patience = int(hyperparams.get("patience", hyperparams.get("n_iter_no_change", 6)))
        self.min_delta = float(hyperparams.get("min_delta", hyperparams.get("tol", 1e-4)))
        self.noise_std = float(hyperparams.get("noise_std", 0.01))
        self.label_smoothing = float(hyperparams.get("label_smoothing", 0.1))
        self.gradient_clip = float(hyperparams.get("gradient_clip", 0.5))
        self.pct_start = float(hyperparams.get("pct_start", 0.2))
        self.validation_fraction = float(hyperparams.get("validation_fraction", 0.2))
        self.device = str(hyperparams.get("device", "auto")).strip().lower()
        self.use_batch_norm = bool(hyperparams.get("use_batch_norm", True))
        self.model_ = None
        self.device_ = None
        self.input_dim_ = None
        self.best_epoch_ = None
        self.best_val_roc_auc_ = None
        self.history_ = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_roc_auc": [],
        }

    def _resolve_device(self, torch):
        if self.device == "cpu":
            return torch.device("cpu")
        if self.device == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "mps":
            has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            return torch.device("mps" if has_mps else "cpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _build_network(self, nn, input_dim: int):
        layers = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if self.dropout_rate > 0.0:
                layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))

        class RegularizedHybridNet(nn.Module):
            def __init__(self, modules):
                super().__init__()
                self.network = nn.Sequential(*modules)

            def forward(self, x):
                return self.network(x).squeeze(-1)

        return RegularizedHybridNet(layers)

    def fit(
        self,
        x,
        y,
        x_val=None,
        y_val=None,
    ):
        torch, nn, OneCycleLR, DataLoader, TensorDataset = _import_torch_modules()

        x_train = np.asarray(x, dtype=np.float32)
        y_train = np.asarray(y, dtype=np.float32).reshape(-1)

        if x_val is None or y_val is None or len(y_val) == 0:
            stratify = y_train if np.unique(y_train).size > 1 else None
            x_train, x_val_arr, y_train, y_val_arr = train_test_split(
                x_train,
                y_train,
                test_size=self.validation_fraction,
                random_state=self.random_seed,
                stratify=stratify,
            )
        else:
            x_val_arr = np.asarray(x_val, dtype=np.float32)
            y_val_arr = np.asarray(y_val, dtype=np.float32).reshape(-1)

        self.input_dim_ = int(x_train.shape[1])
        self.device_ = self._resolve_device(torch)
        self.model_ = self._build_network(nn, self.input_dim_).to(self.device_)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        train_dataset = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(x_val_arr, dtype=torch.float32),
            torch.tensor(y_val_arr.reshape(-1, 1), dtype=torch.float32),
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.lr,
            epochs=self.epochs,
            steps_per_epoch=max(1, len(train_loader)),
            pct_start=self.pct_start,
        )

        best_val_roc_auc = -1.0
        best_model_state = None
        best_epoch = 0
        no_improve = 0

        for epoch in range(self.epochs):
            self.model_.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_).squeeze(1)
                if self.noise_std > 0:
                    xb = xb + (torch.randn_like(xb) * self.noise_std)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(xb)
                if self.label_smoothing > 0:
                    yb_smooth = yb * (1 - 2 * self.label_smoothing) + self.label_smoothing
                    loss = criterion(logits, yb_smooth)
                else:
                    loss = criterion(logits, yb)
                loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model_.parameters(), self.gradient_clip)
                optimizer.step()
                scheduler.step()

                batch_size = int(xb.shape[0])
                train_loss_sum += float(loss.item()) * batch_size
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                train_correct += int((preds == yb).sum().item())
                train_total += batch_size

            self.model_.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0
            val_probs: list[float] = []
            val_labels: list[int] = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device_)
                    yb = yb.to(self.device_).squeeze(1)
                    logits = self.model_(xb)
                    val_loss = criterion(logits, yb)
                    probs = torch.sigmoid(logits).detach().cpu().numpy()
                    val_probs.extend(probs.tolist())

                    y_true_batch = yb.detach().cpu().numpy().astype(int)
                    val_labels.extend(y_true_batch.tolist())
                    y_pred_batch = (probs >= 0.5).astype(int)
                    val_correct += int((y_pred_batch == y_true_batch).sum())
                    batch_size = int(xb.shape[0])
                    val_total += batch_size
                    val_loss_sum += float(val_loss.item()) * batch_size

            val_true = np.asarray(val_labels, dtype=int)
            val_prob = np.asarray(val_probs, dtype=float)
            if np.unique(val_true).size > 1:
                val_roc_auc = float(roc_auc_score(val_true, val_prob))
            else:
                val_roc_auc = 0.0

            train_loss = float(train_loss_sum / max(1, train_total))
            val_loss = float(val_loss_sum / max(1, val_total))
            train_acc = float(train_correct / max(1, train_total))
            val_acc = float(val_correct / max(1, val_total))

            self.history_["train_loss"].append(train_loss)
            self.history_["val_loss"].append(val_loss)
            self.history_["train_acc"].append(train_acc)
            self.history_["val_acc"].append(val_acc)
            self.history_["val_roc_auc"].append(val_roc_auc)

            if val_roc_auc > (best_val_roc_auc + self.min_delta):
                best_val_roc_auc = val_roc_auc
                best_model_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_model_state is not None:
            self.model_.load_state_dict(best_model_state)

        self.best_epoch_ = int(best_epoch)
        self.best_val_roc_auc_ = float(best_val_roc_auc)
        return self

    def predict_proba(self, x):
        if self.model_ is None:
            raise RuntimeError("TorchRegularizedHybridNetClassifier is not fitted yet.")

        torch, _, _, _, _ = _import_torch_modules()
        x_arr = np.asarray(x, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x_arr, dtype=torch.float32).to(self.device_)
            logits = self.model_(x_tensor)
            probs_pos = torch.sigmoid(logits).detach().cpu().numpy().astype(float)
        probs_pos = np.clip(probs_pos, 0.0, 1.0)
        probs_neg = 1.0 - probs_pos
        return np.column_stack([probs_neg, probs_pos])


class TorchResidualHybridNetClassifier(TorchRegularizedHybridNetClassifier):
    """
    Residual MLP variant for tabular acoustic features.
    Keeps the same training procedure/hyperparameter surface as the base model.
    """

    def _build_network(self, nn, input_dim: int):
        class ResidualBlock(nn.Module):
            def __init__(
                self,
                in_dim: int,
                out_dim: int,
                *,
                use_batch_norm: bool,
                dropout_rate: float,
            ) -> None:
                super().__init__()
                self.fc1 = nn.Linear(in_dim, out_dim)
                self.bn1 = nn.BatchNorm1d(out_dim) if use_batch_norm else None
                self.fc2 = nn.Linear(out_dim, out_dim)
                self.bn2 = nn.BatchNorm1d(out_dim) if use_batch_norm else None
                self.act = nn.ReLU()
                self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None
                self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

            def forward(self, x):
                identity = self.skip(x)
                out = self.fc1(x)
                if self.bn1 is not None:
                    out = self.bn1(out)
                out = self.act(out)
                if self.dropout is not None:
                    out = self.dropout(out)
                out = self.fc2(out)
                if self.bn2 is not None:
                    out = self.bn2(out)
                out = out + identity
                out = self.act(out)
                return out

        class ResidualHybridNet(nn.Module):
            def __init__(self, *, input_dim: int, hidden_dims: tuple[int, ...], use_batch_norm: bool, dropout_rate: float):
                super().__init__()
                blocks: list[nn.Module] = []
                prev_dim = int(input_dim)
                for hidden_dim in hidden_dims:
                    blocks.append(
                        ResidualBlock(
                            prev_dim,
                            int(hidden_dim),
                            use_batch_norm=use_batch_norm,
                            dropout_rate=dropout_rate,
                        )
                    )
                    prev_dim = int(hidden_dim)
                self.blocks = nn.ModuleList(blocks)
                self.head = nn.Linear(prev_dim, 1)

            def forward(self, x):
                out = x
                for block in self.blocks:
                    out = block(out)
                return self.head(out).squeeze(-1)

        return ResidualHybridNet(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate,
        )


class TorchWideDeepNetClassifier(TorchRegularizedHybridNetClassifier):
    """
    Wide + deep tabular net.
    Wide branch captures sparse linear effects while deep branch captures nonlinear interactions.
    """

    def _build_network(self, nn, input_dim: int):
        class WideDeepNet(nn.Module):
            def __init__(self, *, input_dim: int, hidden_dims: tuple[int, ...], use_batch_norm: bool, dropout_rate: float):
                super().__init__()
                self.wide = nn.Linear(input_dim, 1)

                layers: list[nn.Module] = []
                prev_dim = int(input_dim)
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, int(hidden_dim)))
                    if use_batch_norm:
                        layers.append(nn.BatchNorm1d(int(hidden_dim)))
                    layers.append(nn.ReLU())
                    if dropout_rate > 0.0:
                        layers.append(nn.Dropout(dropout_rate))
                    prev_dim = int(hidden_dim)

                self.deep = nn.Sequential(*layers) if layers else nn.Identity()
                self.deep_head = nn.Linear(prev_dim, 1)

            def forward(self, x):
                wide_logits = self.wide(x).squeeze(-1)
                deep_features = self.deep(x)
                deep_logits = self.deep_head(deep_features).squeeze(-1)
                return wide_logits + deep_logits

        return WideDeepNet(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate,
        )


def build_model(name: str, hyperparams: dict[str, Any], random_seed: int):
    if name == "logistic_regression":
        params = dict(hyperparams)
        params.setdefault("random_state", random_seed)
        return LogisticRegression(**params)

    if name == "elastic_net_logistic":
        params = dict(hyperparams)
        params.setdefault("random_state", random_seed)
        return SGDClassifier(**params)

    if name == "random_forest":
        params = dict(hyperparams)
        params.setdefault("random_state", random_seed)
        return RandomForestClassifier(**params)

    if name == "extra_trees":
        params = dict(hyperparams)
        params.setdefault("random_state", random_seed)
        return ExtraTreesClassifier(**params)

    if name == "hist_gradient_boosting":
        params = dict(hyperparams)
        params.setdefault("random_state", random_seed)
        return HistGradientBoostingClassifier(**params)

    if name == "svc_rbf":
        params = dict(hyperparams)
        params.setdefault("random_state", random_seed)
        return SVC(**params)

    if name == "mlp":
        params = dict(hyperparams)
        backend = str(params.pop("backend", "pytorch")).strip().lower()
        if backend not in {"pytorch", "torch", "regularizedhybridnet"}:
            raise ValueError(
                f"Unsupported mlp backend '{backend}'. "
                "Set hyperparameters.mlp.backend to 'pytorch' to match notebook methods."
            )
        return TorchRegularizedHybridNetClassifier(random_seed=random_seed, **params)

    if name == "residual_mlp":
        params = dict(hyperparams)
        return TorchResidualHybridNetClassifier(random_seed=random_seed, **params)

    if name == "wide_deep_mlp":
        params = dict(hyperparams)
        return TorchWideDeepNetClassifier(random_seed=random_seed, **params)

    raise ValueError(f"Unknown model: {name}")


def predict_positive_proba(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        # Map scores to (0,1) monotonically for ranking metrics.
        return 1.0 / (1.0 + np.exp(-scores))
    raise RuntimeError("Model does not expose predict_proba or decision_function.")
