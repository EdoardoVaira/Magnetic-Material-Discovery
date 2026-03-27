"""Single masked multitask training script for magnetic materials screening."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Subset, WeightedRandomSampler
from torch_geometric.loader import DataLoader

from dataset import (
    CrystalMaskedMagneticDataset,
    GraphConfig,
    MAGNETIC_ORDERING_CLASSES,
    ensure_parent_dir,
    resolve_device,
    seed_everything,
)
from model import MagNet, ModelConfig


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 96
    epochs: int = 80
    log_every: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    patience: int = 12
    num_workers: int = 0
    seed: int = 7
    device: str = "auto"
    grad_clip_norm: float | None = 5.0
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5


@dataclass(frozen=True)
class RegressionTaskSpec:
    name: str
    output_key: str
    target_key: str
    mask_key: str


@dataclass(frozen=True)
class RegressionMetrics:
    loss: float
    mae: float
    rmse: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ClassificationMetrics:
    loss: float
    accuracy: float
    macro_f1: float
    balanced_accuracy: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ClassificationEvaluation:
    metrics: ClassificationMetrics
    confusion_matrix: list[list[int]]
    per_class_metrics: dict[str, dict[str, float | int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "metrics": self.metrics.to_dict(),
            "confusion_matrix": self.confusion_matrix,
            "per_class_metrics": self.per_class_metrics,
        }


@dataclass(frozen=True)
class TargetScaler:
    mean: float
    std: float

    @classmethod
    def fit(cls, targets: torch.Tensor) -> TargetScaler:
        mean = targets.mean().item()
        std = targets.std(unbiased=False).item()
        return cls(mean=mean, std=max(std, 1e-8))

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class EpochReport:
    loss: float
    loss_terms: dict[str, float]
    regression: dict[str, RegressionMetrics | None]
    ordering: ClassificationEvaluation
    magnetic_loss: float | None
    magnetic_accuracy: float | None
    uncertainty_log_vars: dict[str, float] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "loss": self.loss,
            "loss_terms": self.loss_terms,
            "regression": {
                name: None if metrics is None else metrics.to_dict()
                for name, metrics in self.regression.items()
            },
            "ordering": self.ordering.to_dict(),
            "magnetic_loss": self.magnetic_loss,
            "magnetic_accuracy": self.magnetic_accuracy,
            "uncertainty_log_vars": self.uncertainty_log_vars,
        }


GRAPH_REGRESSION_TASKS: tuple[RegressionTaskSpec, ...] = (
    RegressionTaskSpec("energy", "energy", "y_energy", "energy_mask"),
    RegressionTaskSpec("formation_energy", "formation_energy", "y_formation_energy", "formation_energy_mask"),
    RegressionTaskSpec("band_gap", "band_gap", "y_band_gap", "band_gap_mask"),
    RegressionTaskSpec("magnetization", "magnetization", "y_magnetization", "magnetization_mask"),
    RegressionTaskSpec(
        "transition_temperature",
        "transition_temperature",
        "y_transition_temperature",
        "transition_temperature_mask",
    ),
)
SITE_MOMENT_TASK = RegressionTaskSpec("site_moments", "site_moments", "y_site_moments", "site_moment_mask")


class MultitaskUncertainty(nn.Module):
    """Learned homoscedastic uncertainty weights for multitask losses."""

    def __init__(
        self,
        task_names: list[str],
        *,
        init_log_var: float = 0.0,
        clamp_value: float = 8.0,
    ) -> None:
        super().__init__()
        self.clamp_value = float(clamp_value)
        self.log_vars = nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor(float(init_log_var), dtype=torch.float32))
                for name in task_names
            }
        )

    def weighted_loss(self, task_name: str, raw_loss: torch.Tensor, *, base_weight: float) -> torch.Tensor:
        bounded_log_var = self.log_vars[task_name].clamp(min=-self.clamp_value, max=self.clamp_value)
        precision = torch.exp(-bounded_log_var)
        return float(base_weight) * (precision * raw_loss + bounded_log_var)

    def to_dict(self) -> dict[str, float]:
        return {
            name: float(param.detach().cpu().clamp(min=-self.clamp_value, max=self.clamp_value).item())
            for name, param in self.log_vars.items()
        }


def compute_regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss: float,
) -> RegressionMetrics:
    residuals = predictions - targets
    return RegressionMetrics(
        loss=loss,
        mae=residuals.abs().mean().item(),
        rmse=torch.sqrt(torch.mean(residuals.pow(2))).item(),
    )


def build_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int,
) -> torch.Tensor:
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for target, prediction in zip(targets.view(-1).tolist(), predictions.view(-1).tolist()):
        confusion[int(target), int(prediction)] += 1
    return confusion


def compute_per_class_metrics(
    confusion: torch.Tensor,
    *,
    class_names: list[str],
) -> dict[str, dict[str, float | int]]:
    per_class: dict[str, dict[str, float | int]] = {}
    for index, class_name in enumerate(class_names):
        tp = int(confusion[index, index].item())
        fp = int(confusion[:, index].sum().item()) - tp
        fn = int(confusion[index, :].sum().item()) - tp
        support = int(confusion[index, :].sum().item())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0.0 else (2.0 * precision * recall) / (precision + recall)
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return per_class


def summarize_classification(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss: float,
    class_names: list[str],
) -> ClassificationEvaluation:
    confusion = build_confusion_matrix(predictions, targets, num_classes=len(class_names))
    total = max(int(confusion.sum().item()), 1)
    accuracy = torch.trace(confusion).item() / total
    per_class_metrics = compute_per_class_metrics(confusion, class_names=class_names)
    macro_f1 = sum(metrics["f1"] for metrics in per_class_metrics.values()) / len(class_names)
    balanced_accuracy = sum(metrics["recall"] for metrics in per_class_metrics.values()) / len(class_names)
    return ClassificationEvaluation(
        metrics=ClassificationMetrics(
            loss=loss,
            accuracy=accuracy,
            macro_f1=macro_f1,
            balanced_accuracy=balanced_accuracy,
        ),
        confusion_matrix=confusion.tolist(),
        per_class_metrics=per_class_metrics,
    )


def split_indices(
    dataset_size: int,
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    if dataset_size < 3:
        raise ValueError("Need at least 3 graphs to create train/validation/test splits.")
    if not 0.0 < val_fraction < 1.0 or not 0.0 < test_fraction < 1.0:
        raise ValueError("Validation and test fractions must be in (0, 1).")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("Validation + test fraction must be < 1.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    test_size = max(1, int(round(dataset_size * test_fraction)))
    val_size = max(1, int(round(dataset_size * val_fraction)))
    if test_size + val_size >= dataset_size:
        overflow = (test_size + val_size) - (dataset_size - 1)
        if overflow > 0:
            if test_size >= val_size:
                test_size -= overflow
            else:
                val_size -= overflow

    train_size = dataset_size - val_size - test_size
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]
    return train_indices, val_indices, test_indices


def _masked_graph_targets(
    dataset: CrystalMaskedMagneticDataset,
    indices: list[int],
    task: RegressionTaskSpec,
) -> torch.Tensor:
    values = []
    for index in indices:
        graph = dataset[index]
        mask = getattr(graph, task.mask_key).view(-1)
        if bool(mask.item()):
            values.append(float(getattr(graph, task.target_key).view(-1).item()))
    if not values:
        return torch.empty(0, dtype=torch.float32)
    return torch.tensor(values, dtype=torch.float32)


def _masked_site_targets(dataset: CrystalMaskedMagneticDataset, indices: list[int]) -> torch.Tensor:
    values = []
    for index in indices:
        graph = dataset[index]
        mask = graph.site_moment_mask.view(-1)
        if mask.any():
            values.append(graph.y_site_moments[mask].to(torch.float32))
    if not values:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(values)


def _resolve_material_id_splits(
    dataset: CrystalMaskedMagneticDataset,
    split_path: Path,
) -> tuple[list[int], list[int], list[int], dict[str, int]]:
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = payload.get("train_material_ids")
    val_ids = payload.get("val_material_ids", payload.get("val"))
    test_ids = payload.get("test_material_ids", payload.get("test"))
    if val_ids is None or test_ids is None:
        raise ValueError("Split JSON must define val_material_ids/test_material_ids (or val/test).")

    material_id_to_index: dict[str, int] = {}
    for index in range(len(dataset)):
        material_id = str(dataset[index].material_id)
        if material_id in material_id_to_index:
            raise ValueError(f"Duplicate material_id found in dataset: {material_id}")
        material_id_to_index[material_id] = index

    def _resolve(ids: list[str], label: str) -> list[int]:
        missing = sorted({str(material_id) for material_id in ids} - material_id_to_index.keys())
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"{len(missing)} {label} material IDs were not found in the dataset. First few: {preview}"
            )
        return [material_id_to_index[str(material_id)] for material_id in ids]

    val_indices = _resolve(val_ids, "validation")
    test_indices = _resolve(test_ids, "test")
    val_set = set(val_indices)
    test_set = set(test_indices)
    if val_set & test_set:
        raise ValueError("Validation/test material-ID split overlaps.")

    if train_ids is not None:
        train_indices = _resolve(train_ids, "train")
        train_set = set(train_indices)
        if train_set & val_set or train_set & test_set:
            raise ValueError("Train split overlaps with validation/test holdouts.")
    else:
        train_indices = [index for index in range(len(dataset)) if index not in val_set and index not in test_set]

    if not train_indices:
        raise ValueError("Custom holdout split produced an empty training set.")

    split_stats = {
        "train": len(train_indices),
        "val": len(val_indices),
        "test": len(test_indices),
    }
    return train_indices, val_indices, test_indices, split_stats


def _fit_target_scalers(
    dataset: CrystalMaskedMagneticDataset,
    train_indices: list[int],
) -> dict[str, TargetScaler | None]:
    scalers: dict[str, TargetScaler | None] = {}
    for task in GRAPH_REGRESSION_TASKS:
        values = _masked_graph_targets(dataset, train_indices, task)
        scalers[task.name] = None if values.numel() == 0 else TargetScaler.fit(values)
    site_values = _masked_site_targets(dataset, train_indices)
    scalers[SITE_MOMENT_TASK.name] = None if site_values.numel() == 0 else TargetScaler.fit(site_values)
    return scalers


def _loss_weight_map(args) -> dict[str, float]:
    return {
        "energy": args.energy_loss_weight,
        "formation_energy": args.formation_energy_loss_weight,
        "band_gap": args.band_gap_loss_weight,
        "magnetization": args.magnetization_loss_weight,
        "site_moments": args.site_moment_loss_weight,
        "ordering": args.ordering_loss_weight,
        "magnetic": args.magnetic_loss_weight,
        "transition_temperature": args.transition_temperature_loss_weight,
        "moment_consistency": args.moment_consistency_loss_weight,
    }


def _task_metric(report: EpochReport, name: str) -> float:
    metrics = report.regression.get(name)
    return float("inf") if metrics is None else metrics.mae


def _selection_score(report: EpochReport, selection_metric: str) -> tuple[float, ...]:
    if selection_metric == "loss":
        return (-report.loss,)
    if selection_metric == "ordering_macro_f1":
        return (report.ordering.metrics.macro_f1, -report.loss)
    if selection_metric == "ordering_macro_f1_then_magnetization":
        return (
            report.ordering.metrics.macro_f1,
            -_task_metric(report, "magnetization"),
            -report.loss,
        )
    if selection_metric == "transition_temperature_mae":
        return (-_task_metric(report, "transition_temperature"), report.ordering.metrics.macro_f1)
    raise ValueError(f"Unsupported selection metric: {selection_metric}")


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    scalers: dict[str, TargetScaler | None],
    ordering_criterion: torch.nn.Module,
    magnetic_criterion: torch.nn.Module | None,
    class_names: list[str],
    loss_weights: dict[str, float],
    uncertainty_module: MultitaskUncertainty | None,
    device: torch.device,
    training: bool,
    grad_clip_norm: float | None,
) -> EpochReport:
    if training:
        model.train()
        if uncertainty_module is not None:
            uncertainty_module.train()
    else:
        model.eval()
        if uncertainty_module is not None:
            uncertainty_module.eval()

    total_loss = 0.0
    total_graphs = 0
    total_term_values = {name: 0.0 for name in loss_weights}
    regression_predictions: dict[str, list[torch.Tensor]] = {
        task.name: [] for task in GRAPH_REGRESSION_TASKS + (SITE_MOMENT_TASK,)
    }
    regression_targets: dict[str, list[torch.Tensor]] = {
        task.name: [] for task in GRAPH_REGRESSION_TASKS + (SITE_MOMENT_TASK,)
    }
    ordering_predictions: list[torch.Tensor] = []
    ordering_targets: list[torch.Tensor] = []
    magnetic_predictions: list[torch.Tensor] = []
    magnetic_targets: list[torch.Tensor] = []

    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch)
        batch_loss = torch.zeros((), device=device)

        for task in GRAPH_REGRESSION_TASKS:
            scaler = scalers.get(task.name)
            if scaler is None:
                continue
            target = getattr(batch, task.target_key).view(-1)
            mask = getattr(batch, task.mask_key).view(-1).bool()
            if not mask.any():
                continue
            prediction = outputs[task.output_key].view(-1)
            task_loss = F.smooth_l1_loss(
                scaler.transform(prediction[mask]),
                scaler.transform(target[mask]),
            )
            weighted_task_loss = (
                uncertainty_module.weighted_loss(task.name, task_loss, base_weight=loss_weights[task.name])
                if uncertainty_module is not None
                else loss_weights[task.name] * task_loss
            )
            batch_loss = batch_loss + weighted_task_loss
            total_term_values[task.name] += task_loss.item() * batch.num_graphs
            regression_predictions[task.name].append(prediction[mask].detach().cpu())
            regression_targets[task.name].append(target[mask].detach().cpu())

        site_scaler = scalers.get(SITE_MOMENT_TASK.name)
        if site_scaler is not None:
            site_target = batch.y_site_moments.view(-1)
            site_mask = batch.site_moment_mask.view(-1).bool()
            if site_mask.any():
                site_prediction = outputs["site_moments"].view(-1)
                site_loss = F.smooth_l1_loss(
                    site_scaler.transform(site_prediction[site_mask]),
                    site_scaler.transform(site_target[site_mask]),
                )
                weighted_site_loss = (
                    uncertainty_module.weighted_loss(
                        "site_moments",
                        site_loss,
                        base_weight=loss_weights["site_moments"],
                    )
                    if uncertainty_module is not None
                    else loss_weights["site_moments"] * site_loss
                )
                batch_loss = batch_loss + weighted_site_loss
                total_term_values["site_moments"] += site_loss.item() * batch.num_graphs
                regression_predictions["site_moments"].append(site_prediction[site_mask].detach().cpu())
                regression_targets["site_moments"].append(site_target[site_mask].detach().cpu())

        ordering_target = batch.y_ordering.view(-1).long()
        ordering_mask = batch.ordering_mask.view(-1).bool()
        if ordering_mask.any():
            ordering_logits = outputs["ordering_logits"]
            ordering_loss = ordering_criterion(ordering_logits[ordering_mask], ordering_target[ordering_mask])
            weighted_ordering_loss = (
                uncertainty_module.weighted_loss(
                    "ordering",
                    ordering_loss,
                    base_weight=loss_weights["ordering"],
                )
                if uncertainty_module is not None
                else loss_weights["ordering"] * ordering_loss
            )
            batch_loss = batch_loss + weighted_ordering_loss
            total_term_values["ordering"] += ordering_loss.item() * batch.num_graphs
            ordering_predictions.append(ordering_logits[ordering_mask].argmax(dim=-1).detach().cpu())
            ordering_targets.append(ordering_target[ordering_mask].detach().cpu())

        magnetic_target = batch.y_is_magnetic.view(-1)
        magnetic_mask = batch.magnetic_mask.view(-1).bool()
        if magnetic_criterion is not None and magnetic_mask.any():
            magnetic_logits = outputs["magnetic_logits"].view(-1)
            magnetic_loss = magnetic_criterion(magnetic_logits[magnetic_mask], magnetic_target[magnetic_mask])
            weighted_magnetic_loss = (
                uncertainty_module.weighted_loss(
                    "magnetic",
                    magnetic_loss,
                    base_weight=loss_weights["magnetic"],
                )
                if uncertainty_module is not None
                else loss_weights["magnetic"] * magnetic_loss
            )
            batch_loss = batch_loss + weighted_magnetic_loss
            total_term_values["magnetic"] += magnetic_loss.item() * batch.num_graphs
            magnetic_predictions.append((torch.sigmoid(magnetic_logits[magnetic_mask]) >= 0.5).detach().cpu())
            magnetic_targets.append(magnetic_target[magnetic_mask].detach().cpu())

        if "magnetization_from_sites" in outputs and loss_weights["moment_consistency"] > 0.0:
            consistency_loss = F.smooth_l1_loss(outputs["magnetization"], outputs["magnetization_from_sites"])
            weighted_consistency_loss = (
                uncertainty_module.weighted_loss(
                    "moment_consistency",
                    consistency_loss,
                    base_weight=loss_weights["moment_consistency"],
                )
                if uncertainty_module is not None
                else loss_weights["moment_consistency"] * consistency_loss
            )
            batch_loss = batch_loss + weighted_consistency_loss
            total_term_values["moment_consistency"] += consistency_loss.item() * batch.num_graphs

        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        total_loss += batch_loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs

    if scheduler is not None and not training:
        scheduler.step(total_loss / max(total_graphs, 1))

    mean_loss = total_loss / max(total_graphs, 1)
    regression_report: dict[str, RegressionMetrics | None] = {}
    for task in GRAPH_REGRESSION_TASKS + (SITE_MOMENT_TASK,):
        preds = regression_predictions[task.name]
        targets = regression_targets[task.name]
        if preds and targets:
            regression_report[task.name] = compute_regression_metrics(
                torch.cat(preds),
                torch.cat(targets),
                loss=mean_loss,
            )
        else:
            regression_report[task.name] = None

    if not ordering_predictions or not ordering_targets:
        raise RuntimeError("No labeled ordering targets were available in this split.")
    ordering_report = summarize_classification(
        torch.cat(ordering_predictions),
        torch.cat(ordering_targets),
        loss=mean_loss,
        class_names=class_names,
    )

    magnetic_loss_value = None
    magnetic_accuracy = None
    if magnetic_predictions and magnetic_targets:
        predicted = torch.cat(magnetic_predictions).to(torch.long)
        target = torch.cat(magnetic_targets).to(torch.long)
        magnetic_accuracy = (predicted == target).to(torch.float32).mean().item()
        magnetic_loss_value = total_term_values["magnetic"] / max(total_graphs, 1)

    mean_term_values = {
        name: value / max(total_graphs, 1)
        for name, value in total_term_values.items()
    }
    return EpochReport(
        loss=mean_loss,
        loss_terms=mean_term_values,
        regression=regression_report,
        ordering=ordering_report,
        magnetic_loss=magnetic_loss_value,
        magnetic_accuracy=magnetic_accuracy,
        uncertainty_log_vars=None if uncertainty_module is None else uncertainty_module.to_dict(),
    )


def fit_multitask(
    *,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scalers: dict[str, TargetScaler | None],
    ordering_criterion: torch.nn.Module,
    magnetic_criterion: torch.nn.Module | None,
    class_names: list[str],
    loss_weights: dict[str, float],
    uncertainty_module: MultitaskUncertainty | None,
    device: torch.device,
    epochs: int,
    patience: int,
    log_every: int,
    grad_clip_norm: float | None,
    selection_metric: str,
    checkpoint_path: Path,
) -> tuple[list[dict[str, float | None]], dict[str, object]]:
    best_score: tuple[float, ...] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | None]] = []
    best_state: dict[str, object] | None = None

    for epoch in range(1, epochs + 1):
        train_report = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=None,
            scalers=scalers,
            ordering_criterion=ordering_criterion,
            magnetic_criterion=magnetic_criterion,
            class_names=class_names,
            loss_weights=loss_weights,
            uncertainty_module=uncertainty_module,
            device=device,
            training=True,
            grad_clip_norm=grad_clip_norm,
        )
        val_report = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            scheduler=scheduler,
            scalers=scalers,
            ordering_criterion=ordering_criterion,
            magnetic_criterion=magnetic_criterion,
            class_names=class_names,
            loss_weights=loss_weights,
            uncertainty_module=uncertainty_module,
            device=device,
            training=False,
            grad_clip_norm=None,
        )

        history_row: dict[str, float | None] = {
            "epoch": float(epoch),
            "train_loss": train_report.loss,
            "val_loss": val_report.loss,
            "train_ordering_macro_f1": train_report.ordering.metrics.macro_f1,
            "val_ordering_macro_f1": val_report.ordering.metrics.macro_f1,
            "train_ordering_accuracy": train_report.ordering.metrics.accuracy,
            "val_ordering_accuracy": val_report.ordering.metrics.accuracy,
            "train_ordering_balanced_accuracy": train_report.ordering.metrics.balanced_accuracy,
            "val_ordering_balanced_accuracy": val_report.ordering.metrics.balanced_accuracy,
        }
        for task_name in ["energy", "formation_energy", "band_gap", "magnetization", "site_moments", "transition_temperature"]:
            train_metrics = train_report.regression.get(task_name)
            val_metrics = val_report.regression.get(task_name)
            history_row[f"train_{task_name}_mae"] = None if train_metrics is None else train_metrics.mae
            history_row[f"val_{task_name}_mae"] = None if val_metrics is None else val_metrics.mae
        if uncertainty_module is not None:
            for name, value in uncertainty_module.to_dict().items():
                history_row[f"uncertainty_log_var_{name}"] = value
        history.append(history_row)

        if log_every > 0 and epoch % log_every == 0:
            transition_segment = ""
            val_transition = val_report.regression.get("transition_temperature")
            if val_transition is not None:
                transition_segment = f" val_Tc_mae={val_transition.mae:.4f}"
            print(
                (
                    f"Epoch {epoch:03d} | "
                    f"train_loss={train_report.loss:.4f} "
                    f"val_loss={val_report.loss:.4f} "
                    f"val_E_mae={_task_metric(val_report, 'energy'):.4f} "
                    f"val_M_mae={_task_metric(val_report, 'magnetization'):.4f} "
                    f"val_site_mae={_task_metric(val_report, 'site_moments'):.4f} "
                    f"val_F1={val_report.ordering.metrics.macro_f1:.4f} "
                    f"val_acc={val_report.ordering.metrics.accuracy:.4f} "
                    f"val_bal_acc={val_report.ordering.metrics.balanced_accuracy:.4f}"
                    f"{transition_segment} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                ),
                flush=True,
            )

        selection_score = _selection_score(val_report, selection_metric)
        if best_score is None or selection_score > best_score:
            best_score = selection_score
            epochs_without_improvement = 0
            best_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "target_scalers": {
                    name: None if scaler is None else scaler.to_dict()
                    for name, scaler in scalers.items()
                },
                "selection_metric": selection_metric,
                "selection_score": list(selection_score),
                "history": history,
            }
            if uncertainty_module is not None:
                best_state["uncertainty_state_dict"] = uncertainty_module.state_dict()
                best_state["uncertainty_log_vars"] = uncertainty_module.to_dict()
            ensure_parent_dir(checkpoint_path)
            torch.save(best_state, checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training finished without producing a checkpoint.")
    return history, best_state


@torch.no_grad()
def evaluate_multitask(
    *,
    model: torch.nn.Module,
    loader,
    scalers: dict[str, TargetScaler | None],
    ordering_criterion: torch.nn.Module,
    magnetic_criterion: torch.nn.Module | None,
    class_names: list[str],
    loss_weights: dict[str, float],
    uncertainty_module: MultitaskUncertainty | None,
    device: torch.device,
) -> EpochReport:
    return _run_epoch(
        model=model,
        loader=loader,
        optimizer=None,
        scheduler=None,
        scalers=scalers,
        ordering_criterion=ordering_criterion,
        magnetic_criterion=magnetic_criterion,
        class_names=class_names,
        loss_weights=loss_weights,
        uncertainty_module=uncertainty_module,
        device=device,
        training=False,
        grad_clip_norm=None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the single multitask magnetic materials model."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data"))
    parser.add_argument("--raw-filename", type=str, default="magnetic_unified.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/magnetic_model"))
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--holdout-material-ids-json", type=Path, default=None)

    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--num-radial", type=int, default=64)
    parser.add_argument("--radial-width", type=float, default=0.5)
    parser.add_argument("--feature-version", type=str, default=None)

    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--vector-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--graph-pooling",
        type=str,
        choices=("mean", "mean_max", "attention", "mean_max_attention"),
        default="mean_max",
    )
    parser.add_argument(
        "--ordering-pooling",
        type=str,
        choices=("mean", "mean_max", "attention", "mean_max_attention"),
        default="mean_max_attention",
    )

    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--lr-scheduler-patience", type=int, default=5)
    parser.add_argument(
        "--selection-metric",
        type=str,
        choices=(
            "loss",
            "ordering_macro_f1",
            "ordering_macro_f1_then_magnetization",
            "transition_temperature_mae",
        ),
        default="loss",
    )
    parser.add_argument("--balanced-sampler", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--uncertainty-weighting", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--uncertainty-init-log-var", type=float, default=0.0)

    parser.add_argument("--energy-loss-weight", type=float, default=1.0)
    parser.add_argument("--formation-energy-loss-weight", type=float, default=0.5)
    parser.add_argument("--band-gap-loss-weight", type=float, default=0.5)
    parser.add_argument("--magnetization-loss-weight", type=float, default=2.0)
    parser.add_argument("--site-moment-loss-weight", type=float, default=1.0)
    parser.add_argument("--ordering-loss-weight", type=float, default=1.5)
    parser.add_argument("--magnetic-loss-weight", type=float, default=0.25)
    parser.add_argument("--transition-temperature-loss-weight", type=float, default=0.25)
    parser.add_argument("--moment-consistency-loss-weight", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)

    graph_config = GraphConfig(
        cutoff=args.cutoff,
        num_radial=args.num_radial,
        radial_width=args.radial_width,
        feature_version=args.feature_version or "chem_node_geo_v2",
    )
    model_config = ModelConfig(
        hidden_dim=args.hidden_dim,
        vector_dim=args.vector_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        graph_pooling=args.graph_pooling,
        ordering_pooling=args.ordering_pooling,
    )
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        log_every=args.log_every,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        patience=args.patience,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        grad_clip_norm=args.grad_clip_norm,
        lr_scheduler_factor=args.lr_scheduler_factor,
        lr_scheduler_patience=args.lr_scheduler_patience,
    )
    loss_weights = _loss_weight_map(args)

    dataset = CrystalMaskedMagneticDataset(
        root=args.dataset_root,
        raw_filename=args.raw_filename,
        graph_config=graph_config,
        load_processed=not args.preprocess_only,
    )
    if args.preprocess_only:
        print(
            json.dumps(
                {
                    "processed_path": str(dataset.processed_paths[0]),
                    "graph_config": asdict(graph_config),
                },
                indent=2,
            )
        )
        return

    split_source = "random"
    split_material_id_path = None
    if args.holdout_material_ids_json is not None:
        train_indices, val_indices, test_indices, split_stats = _resolve_material_id_splits(
            dataset,
            args.holdout_material_ids_json,
        )
        split_source = "material_id_holdout"
        split_material_id_path = str(args.holdout_material_ids_json)
        print(
            "Using leak-free holdout split from "
            f"{args.holdout_material_ids_json}: {json.dumps(split_stats)}"
        )
    else:
        train_indices, val_indices, test_indices = split_indices(
            len(dataset),
            val_fraction=training_config.val_fraction,
            test_fraction=training_config.test_fraction,
            seed=training_config.seed,
        )

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    scalers = _fit_target_scalers(dataset, train_indices)
    if scalers["energy"] is None or scalers["magnetization"] is None:
        raise RuntimeError("Training split is missing core energy or magnetization labels.")

    class_names = list(MAGNETIC_ORDERING_CLASSES)
    ordering_labels = []
    ordering_label_indices = []
    for dataset_index in train_indices:
        graph = dataset[dataset_index]
        if bool(graph.ordering_mask.view(-1).item()):
            ordering_labels.append(int(graph.y_ordering.item()))
            ordering_label_indices.append(dataset_index)
    if not ordering_labels:
        raise RuntimeError("No labeled ordering targets were found in the training split.")

    ordering_targets = torch.tensor(ordering_labels, dtype=torch.long)
    class_counts = torch.bincount(ordering_targets, minlength=len(class_names))
    class_weights = ordering_targets.numel() / (len(class_names) * class_counts.clamp(min=1).float())
    device = resolve_device(training_config.device)
    ordering_criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    magnetic_labels = []
    for dataset_index in train_indices:
        graph = dataset[dataset_index]
        if bool(graph.magnetic_mask.view(-1).item()):
            magnetic_labels.append(float(graph.y_is_magnetic.view(-1).item()))

    magnetic_criterion = None
    magnetic_pos_weight = None
    if magnetic_labels:
        magnetic_targets = torch.tensor(magnetic_labels, dtype=torch.float32)
        positive = float(magnetic_targets.sum().item())
        negative = float(magnetic_targets.numel() - positive)
        if positive > 0.0 and negative > 0.0:
            magnetic_pos_weight = negative / positive
            magnetic_criterion = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(magnetic_pos_weight, dtype=torch.float32, device=device)
            )
        else:
            magnetic_criterion = torch.nn.BCEWithLogitsLoss()

    train_sampler = None
    shuffle_train = True
    if args.balanced_sampler:
        sample_weights = torch.ones(len(train_indices), dtype=torch.double)
        index_lookup = {dataset_index: offset for offset, dataset_index in enumerate(train_indices)}
        for dataset_index, label in zip(ordering_label_indices, ordering_labels):
            sample_weights[index_lookup[dataset_index]] = class_weights[label].double()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_subset),
            replacement=True,
        )
        shuffle_train = False
        ordering_criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(
        train_subset,
        batch_size=training_config.batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=training_config.num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
    )

    model = MagNet(
        edge_dim=graph_config.num_radial,
        num_classes=len(class_names),
        config=model_config,
    ).to(device)
    uncertainty_module = None
    if args.uncertainty_weighting:
        uncertainty_module = MultitaskUncertainty(
            task_names=list(loss_weights.keys()),
            init_log_var=args.uncertainty_init_log_var,
        ).to(device)
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if uncertainty_module is not None and "uncertainty_state_dict" in checkpoint:
            uncertainty_module.load_state_dict(checkpoint["uncertainty_state_dict"], strict=False)

    optimizer_parameters: list[dict[str, object]] = [{"params": list(model.parameters())}]
    if uncertainty_module is not None:
        optimizer_parameters.append(
            {
                "params": list(uncertainty_module.parameters()),
                "weight_decay": 0.0,
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=training_config.lr_scheduler_factor,
        patience=training_config.lr_scheduler_patience,
    )

    checkpoint_path = args.output_dir / "best_model.pt"
    history, best_state = fit_multitask(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scalers=scalers,
        ordering_criterion=ordering_criterion,
        magnetic_criterion=magnetic_criterion,
        class_names=class_names,
        loss_weights=loss_weights,
        uncertainty_module=uncertainty_module,
        device=device,
        epochs=training_config.epochs,
        patience=training_config.patience,
        log_every=training_config.log_every,
        grad_clip_norm=training_config.grad_clip_norm,
        selection_metric=args.selection_metric,
        checkpoint_path=checkpoint_path,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if uncertainty_module is not None and "uncertainty_state_dict" in checkpoint:
        uncertainty_module.load_state_dict(checkpoint["uncertainty_state_dict"], strict=False)
    test_report = evaluate_multitask(
        model=model,
        loader=test_loader,
        scalers=scalers,
        ordering_criterion=ordering_criterion,
        magnetic_criterion=magnetic_criterion,
        class_names=class_names,
        loss_weights=loss_weights,
        uncertainty_module=uncertainty_module,
        device=device,
    )

    summary = {
        "device": str(device),
        "dataset": {
            "root": str(args.dataset_root),
            "raw_filename": args.raw_filename,
            "metadata": dataset.metadata,
        },
        "num_graphs": len(dataset),
        "split_sizes": {
            "train": len(train_subset),
            "val": len(val_subset),
            "test": len(test_subset),
        },
        "split_indices": {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        },
        "split_strategy": split_source,
        "split_material_ids_json": split_material_id_path,
        "graph_config": asdict(graph_config),
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "selection_metric": args.selection_metric,
        "loss_weights": loss_weights,
        "uncertainty_weighting": args.uncertainty_weighting,
        "uncertainty_init_log_var": args.uncertainty_init_log_var,
        "uncertainty_log_vars": (
            None
            if uncertainty_module is None
            else checkpoint.get("uncertainty_log_vars", uncertainty_module.to_dict())
        ),
        "class_names": class_names,
        "class_weights": {
            label: float(class_weights[index].item())
            for index, label in enumerate(class_names)
        },
        "train_sampling": {
            "balanced_sampler": args.balanced_sampler,
            "ordering_labeled_train_graphs": len(ordering_labels),
            "magnetic_labeled_train_graphs": len(magnetic_labels),
        },
        "magnetic_pos_weight": magnetic_pos_weight,
        "target_scalers": {
            name: None if scaler is None else scaler.to_dict()
            for name, scaler in scalers.items()
        },
        "best_checkpoint_selection_score": best_state.get("selection_score"),
        "test_report": test_report.to_dict(),
        "history_length": len(history),
    }

    ensure_parent_dir(args.output_dir / "placeholder.txt")
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps(summary["test_report"], indent=2))


if __name__ == "__main__":
    main()
