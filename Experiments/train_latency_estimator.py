import argparse
import os
import sys
from dataclasses import dataclass
from typing import Tuple

from loguru import logger
import torch
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from MAR.InfraMind.latency_estimator import (  # noqa: E402
    LatencyEstimator,
    LatencyEstimatorBundle,
    LatencyEstimatorConfig,
    prepare_latency_estimator_dataset,
    save_latency_estimator,
    train_latency_estimator,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train latency estimator from InfraMind CSV.")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to InfraMind training CSV.")
    parser.add_argument("--record-type", type=str, default="role_step", help="CSV record_type filter.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--min-ttft", type=float, default=1e-6, help="Skip samples below this ttft.")
    parser.add_argument("--min-tpot", type=float, default=1e-6, help="Skip samples below this tpot.")
    parser.add_argument("--train-split", type=float, default=0.8, help="Fraction of data for training.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of data for validation.")
    parser.add_argument("--test-split", type=float, default=0.1, help="Fraction of data for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset split.")
    parser.add_argument(
        "--output-path",
        type=str,
        default="checkpoints/latency_estimator.pt",
        help="Checkpoint path to save the estimator.",
    )
    return parser


@dataclass(frozen=True)
class _LatencyMetrics:
    mse: float
    mae: float
    rmse: float
    r2: float


@dataclass(frozen=True)
class _LatencyMetricsBundle:
    ttft: _LatencyMetrics
    tpot: _LatencyMetrics


def _compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> _LatencyMetrics:
    targets = targets.float()
    preds = preds.float()
    mse = torch.mean((preds - targets) ** 2).item()
    mae = torch.mean(torch.abs(preds - targets)).item()
    rmse = mse**0.5
    target_mean = torch.mean(targets).item()
    ss_tot = torch.sum((targets - target_mean) ** 2).item()
    ss_res = torch.sum((targets - preds) ** 2).item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return _LatencyMetrics(mse=mse, mae=mae, rmse=rmse, r2=r2)


def _evaluate_latency_estimator(
    model: LatencyEstimator,
    dataset: Dataset,
    *,
    batch_size: int,
    device: torch.device,
) -> _LatencyMetricsBundle:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    ttft_preds = []
    tpot_preds = []
    ttft_targets = []
    tpot_targets = []
    with torch.no_grad():
        for x_num, strategy_id, role_id, model_id, ttft_target, tpot_target in loader:
            x_num = x_num.to(device)
            strategy_id = strategy_id.to(device)
            role_id = role_id.to(device)
            model_id = model_id.to(device)
            ttft_pred, tpot_pred = model(x_num, strategy_id, role_id, model_id)
            ttft_preds.append(ttft_pred.cpu())
            tpot_preds.append(tpot_pred.cpu())
            ttft_targets.append(ttft_target.cpu())
            tpot_targets.append(tpot_target.cpu())
    if not ttft_preds:
        empty = _LatencyMetrics(mse=0.0, mae=0.0, rmse=0.0, r2=0.0)
        return _LatencyMetricsBundle(ttft=empty, tpot=empty)
    return _LatencyMetricsBundle(
        ttft=_compute_metrics(torch.cat(ttft_preds), torch.cat(ttft_targets)),
        tpot=_compute_metrics(torch.cat(tpot_preds), torch.cat(tpot_targets)),
    )


def _split_dataset(
    dataset: Dataset,
    *,
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int,
) -> Tuple[Dataset, Dataset, Dataset]:
    total = len(dataset)
    if total == 0:
        raise ValueError("No training samples found in the CSV.")
    split_sum = train_split + val_split + test_split
    if split_sum <= 0:
        raise ValueError("Split fractions must sum to a positive value.")
    if split_sum > 1.0001:
        raise ValueError("Split fractions must sum to <= 1.0.")

    train_len = int(total * train_split)
    val_len = int(total * val_split)
    test_len = int(total * test_split)
    remaining = total - train_len - val_len - test_len
    if remaining > 0:
        train_len += remaining

    if train_len == 0:
        raise ValueError("Train split produced 0 samples. Increase train_split.")

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, val_len, test_len], generator=generator)


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, metadata = prepare_latency_estimator_dataset(
        args.csv_path,
        record_type=args.record_type,
        min_ttft=args.min_ttft,
        min_tpot=args.min_tpot,
    )

    train_set, val_set, test_set = _split_dataset(
        dataset,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )
    logger.info(
        "Dataset split sizes -> train: {}, val: {}, test: {}",
        len(train_set),
        len(val_set),
        len(test_set),
    )

    config = LatencyEstimatorConfig(num_numerical_features=metadata.num_numerical_features)
    model = LatencyEstimator(
        num_numerical_features=config.num_numerical_features,
        num_models=len(metadata.model_vocab),
        num_roles=len(metadata.role_vocab),
        num_strategies=len(metadata.strategy_vocab),
        embedding_dim=config.embedding_dim,
        hidden_dims=list(config.hidden_dims),
        dropout=config.dropout,
    )

    losses = train_latency_estimator(
        model,
        train_set,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )
    logger.info("Training complete. Final loss: {:.4f}", losses[-1] if losses else 0.0)

    train_metrics = _evaluate_latency_estimator(
        model,
        train_set,
        batch_size=args.batch_size,
        device=device,
    )
    logger.info(
        "Train TTFT -> mse: {:.4f}, rmse: {:.4f}, mae: {:.4f}, r2: {:.4f}",
        train_metrics.ttft.mse,
        train_metrics.ttft.rmse,
        train_metrics.ttft.mae,
        train_metrics.ttft.r2,
    )
    logger.info(
        "Train TPOT -> mse: {:.4f}, rmse: {:.4f}, mae: {:.4f}, r2: {:.4f}",
        train_metrics.tpot.mse,
        train_metrics.tpot.rmse,
        train_metrics.tpot.mae,
        train_metrics.tpot.r2,
    )
    if len(val_set) > 0:
        val_metrics = _evaluate_latency_estimator(
            model,
            val_set,
            batch_size=args.batch_size,
            device=device,
        )
        logger.info(
            "Validation TTFT -> mse: {:.4f}, rmse: {:.4f}, mae: {:.4f}, r2: {:.4f}",
            val_metrics.ttft.mse,
            val_metrics.ttft.rmse,
            val_metrics.ttft.mae,
            val_metrics.ttft.r2,
        )
        logger.info(
            "Validation TPOT -> mse: {:.4f}, rmse: {:.4f}, mae: {:.4f}, r2: {:.4f}",
            val_metrics.tpot.mse,
            val_metrics.tpot.rmse,
            val_metrics.tpot.mae,
            val_metrics.tpot.r2,
        )
    else:
        logger.warning("Validation split is empty; skipping validation metrics.")
    if len(test_set) > 0:
        test_metrics = _evaluate_latency_estimator(
            model,
            test_set,
            batch_size=args.batch_size,
            device=device,
        )
        logger.info(
            "Test TTFT -> mse: {:.4f}, rmse: {:.4f}, mae: {:.4f}, r2: {:.4f}",
            test_metrics.ttft.mse,
            test_metrics.ttft.rmse,
            test_metrics.ttft.mae,
            test_metrics.ttft.r2,
        )
        logger.info(
            "Test TPOT -> mse: {:.4f}, rmse: {:.4f}, mae: {:.4f}, r2: {:.4f}",
            test_metrics.tpot.mse,
            test_metrics.tpot.rmse,
            test_metrics.tpot.mae,
            test_metrics.tpot.r2,
        )
    else:
        logger.warning("Test split is empty; skipping test metrics.")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    save_latency_estimator(args.output_path, model, metadata, config)
    logger.info("Saved estimator to {}", args.output_path)

    bundle = LatencyEstimatorBundle(model=model, metadata=metadata)
    logger.info("Bundle ready: {}", bundle)


if __name__ == "__main__":
    main()
