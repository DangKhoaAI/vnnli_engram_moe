from __future__ import annotations

import argparse
import json
from typing import Sequence

from vnnli_engram_moe.config import load_config
from vnnli_engram_moe.models.registry import get_checkpoint
from vnnli_engram_moe.utils.logging import configure_logging


def _build_common_overrides(args: argparse.Namespace) -> list[str]:
    overrides = list(args.override or [])
    if getattr(args, "model_key", None):
        overrides.append(f"model.model_key={args.model_key}")
        overrides.append(f"model.checkpoint={json.dumps(get_checkpoint(args.model_key))}")
    if getattr(args, "train_file", None):
        overrides.append(f"data.train_file={args.train_file}")
    if getattr(args, "validation_file", None):
        overrides.append(f"data.validation_file={args.validation_file}")
    if getattr(args, "test_file", None):
        overrides.append(f"data.test_file={args.test_file}")
    if getattr(args, "output_dir", None):
        overrides.append(f"project.output_dir={json.dumps(args.output_dir)}")
    if getattr(args, "epochs", None) is not None:
        overrides.append(f"training.num_train_epochs={args.epochs}")
    if getattr(args, "batch_size", None) is not None:
        overrides.append(f"training.per_device_train_batch_size={args.batch_size}")
        overrides.append(f"training.per_device_eval_batch_size={args.batch_size}")
    if getattr(args, "learning_rate", None) is not None:
        overrides.append(f"training.learning_rate={args.learning_rate}")
    if getattr(args, "max_steps", None) is not None:
        overrides.append(f"training.max_steps={args.max_steps}")
    if getattr(args, "run_name", None):
        overrides.append(f"project.run_name={json.dumps(args.run_name)}")
    if getattr(args, "device", None):
        overrides.append(f"training.device={json.dumps(args.device)}")
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vietnamese NLI fine-tuning toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train an NLI model.")
    train_parser.add_argument("--config", required=True, help="Base YAML config.")
    train_parser.add_argument("--model-config", help="Optional model override YAML.")
    train_parser.add_argument("--user-config", help="Optional experiment override YAML.")
    train_parser.add_argument("--override", action="append", help="Extra KEY=VALUE override.")
    train_parser.add_argument("--model-key")
    train_parser.add_argument("--train-file")
    train_parser.add_argument("--validation-file")
    train_parser.add_argument("--test-file")
    train_parser.add_argument("--output-dir")
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--batch-size", type=int)
    train_parser.add_argument("--learning-rate", type=float)
    train_parser.add_argument("--max-steps", type=int)
    train_parser.add_argument("--run-name")
    train_parser.add_argument("--device")
    train_parser.add_argument(
        "--random-init",
        action="store_true",
        help="Build the architecture from config without loading pretrained weights.",
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a saved training run.")
    evaluate_parser.add_argument("--run-dir", required=True, help="Run directory containing config.resolved.yaml.")
    evaluate_parser.add_argument("--checkpoint", help="Optional checkpoint directory override.")
    evaluate_parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    evaluate_parser.add_argument("--data-file", help="Optional split file override.")

    predict_parser = subparsers.add_parser("predict", help="Predict a label for one premise/hypothesis pair.")
    predict_parser.add_argument("--run-dir", required=True, help="Run directory containing config.resolved.yaml.")
    predict_parser.add_argument("--checkpoint", help="Optional checkpoint directory override.")
    predict_parser.add_argument("--premise", required=True)
    predict_parser.add_argument("--hypothesis", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        from vnnli_engram_moe.training.trainer import train

        overrides = _build_common_overrides(args)
        config = load_config(
            args.config,
            model_config_path=args.model_config,
            user_config_path=args.user_config,
            overrides=overrides,
        )
        run_dir = train(config, pretrained=not args.random_init)
        print(run_dir)
        return 0

    if args.command == "evaluate":
        from vnnli_engram_moe.training.trainer import evaluate_saved_run

        metrics = evaluate_saved_run(
            run_dir=args.run_dir,
            checkpoint=args.checkpoint,
            split=args.split,
            data_file=args.data_file,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0

    if args.command == "predict":
        from vnnli_engram_moe.training.trainer import predict_text_pair

        prediction = predict_text_pair(
            run_dir=args.run_dir,
            checkpoint=args.checkpoint,
            premise=args.premise,
            hypothesis=args.hypothesis,
        )
        print(json.dumps(prediction, ensure_ascii=False, indent=2))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
