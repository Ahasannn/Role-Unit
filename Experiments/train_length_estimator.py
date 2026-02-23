import argparse
import os
import sys
from dataclasses import dataclass
from typing import Tuple

from loguru import logger
import torch
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from MAR.InfraMind.length_estimator import (  # noqa: E402
    LengthEstimator,
    LengthEstimatorBundle,
    LengthEstimatorConfig,
    prepare_length_estimator_dataset,
    save_length_estimator,
    train_length_estimator,
)
from MAR.InfraMind.inframind_router import SemanticEncoder  # noqa: E402


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train output length estimator from InfraMind CSV.")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to InfraMind training CSV.")
    parser.add_argument("--record-type", type=str, default="role_step", help="CSV record_type filter.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--embed-batch-size", type=int, default=64, help="Prompt embedding batch size.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--min-length", type=int, default=1, help="Skip samples shorter than this length.")
    parser.add_argument("--train-split", type=float, default=0.8, help="Fraction of data for training.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of data for validation.")
    parser.add_argument("--test-split", type=float, default=0.1, help="Fraction of data for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset split.")
    parser.add_argument(
        "--output-path",
        type=str,
        default="checkpoints/length_estimator.pt",
        help="Checkpoint path to save the estimator.",
    )
    return parser


@dataclass(frozen=True)
class _RegressionMetrics:
    mse: float
    mae: float
    rmse: float
    r2: float


def _compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> _RegressionMetrics:
    targets = targets.float()
    preds = preds.float()
    mse = torch.mean((preds - targets) ** 2).item()
    mae = torch.mean(torch.abs(preds - targets)).item()
    rmse = mse**0.5
    target_mean = torch.mean(targets).item()
    ss_tot = torch.sum((targets - target_mean) ** 2).item()
    ss_res = torch.sum((targets - preds) ** 2).item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return _RegressionMetrics(mse=mse, mae=mae, rmse=rmse, r2=r2)


def _evaluate_length_estimator(
    model: LengthEstimator,
    dataset: Dataset,
    *,
    batch_size: int,
    device: torch.device,
) -> _RegressionMetrics:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for semantic, strategy_id, role_id, model_id, target in loader:
            semantic = semantic.to(device)
            strategy_id = strategy_id.to(device)
            role_id = role_id.to(device)
            model_id = model_id.to(device)
            pred = model(semantic, strategy_id, role_id, model_id)
            preds.append(pred.cpu())
            targets.append(target.cpu())
    if not preds:
        return _RegressionMetrics(mse=0.0, mae=0.0, rmse=0.0, r2=0.0)
    return _compute_metrics(torch.cat(preds), torch.cat(targets))


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
    encoder = SemanticEncoder(device=device)
    dataset, metadata = prepare_length_estimator_dataset(
        args.csv_path,
        encoder=encoder,
        record_type=args.record_type,
        batch_size=args.embed_batch_size,
        min_length=args.min_length,
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

    config = LengthEstimatorConfig(semantic_dim=metadata.semantic_dim)
    model = LengthEstimator(
        config,
        num_models=len(metadata.model_vocab),
        num_roles=len(metadata.role_vocab),
        num_strategies=len(metadata.strategy_vocab),
    )

    losses = train_length_estimator(
        model,
        train_set,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )
    logger.info("Training complete. Final loss: {:.4f}", losses[-1] if losses else 0.0)

    train_metrics = _evaluate_length_estimator(
        model,
        train_set,
        batch_size=args.batch_size,
        device=device,
    )
    logger.info(
        "Train metrics -> mse: {:.4f}, rmse: {:.4f}, mae: {:.4f}, r2: {:.4f}",
        train_metrics.mse,
        train_metrics.rmse,
        train_metrics.mae,
        train_metrics.r2,
    )
    if len(val_set) > 0:
        val_metrics = _evaluate_length_estimator(
            model,
            val_set,
            batch_size=args.batch_size,
            device=device,
        )
        logger.info(
            "Validation metrics -> mse: {:.4f}, rmse: {:.4f}, mae: {:.4f}, r2: {:.4f}",
            val_metrics.mse,
            val_metrics.rmse,
            val_metrics.mae,
            val_metrics.r2,
        )
    else:
        logger.warning("Validation split is empty; skipping validation metrics.")
    if len(test_set) > 0:
        test_metrics = _evaluate_length_estimator(
            model,
            test_set,
            batch_size=args.batch_size,
            device=device,
        )
        logger.info(
            "Test metrics -> mse: {:.4f}, rmse: {:.4f}, mae: {:.4f}, r2: {:.4f}",
            test_metrics.mse,
            test_metrics.rmse,
            test_metrics.mae,
            test_metrics.r2,
        )
    else:
        logger.warning("Test split is empty; skipping test metrics.")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    save_length_estimator(args.output_path, model, metadata, config)
    logger.info("Saved estimator to {}", args.output_path)

    bundle = LengthEstimatorBundle(model=model, metadata=metadata, encoder=encoder)
    logger.info("Bundle ready: {}", bundle)


if __name__ == "__main__":
    main()
