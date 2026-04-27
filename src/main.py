"""ML Intern - A lightweight ML training and inference assistant.

This module serves as the main entry point for the ml-intern application,
coordinating model loading, training pipelines, and inference workflows.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.config import Config
from src.trainer import Trainer
from src.inference import InferencePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ML Intern: training and inference assistant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Fine-tune a model")
    train_parser.add_argument(
        "--config", type=Path, required=True, help="Path to training config YAML"
    )
    train_parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"), help="Directory to save checkpoints"
    )
    train_parser.add_argument(
        "--resume", type=Path, default=None, help="Resume from checkpoint path"
    )

    # Infer subcommand
    infer_parser = subparsers.add_parser("infer", help="Run inference on a trained model")
    infer_parser.add_argument(
        "--model-path", type=Path, required=True, help="Path to model checkpoint or HF model id"
    )
    infer_parser.add_argument(
        "--prompt", type=str, default=None, help="Single prompt string for quick inference"
    )
    infer_parser.add_argument(
        "--input-file", type=Path, default=None, help="JSONL file with prompts to process"
    )
    infer_parser.add_argument(
        "--output-file", type=Path, default=Path("predictions.jsonl"), help="Output JSONL file"
    )
    infer_parser.add_argument(
        "--max-new-tokens", type=int, default=256, help="Maximum tokens to generate"
    )

    return parser.parse_args()


def run_train(args: argparse.Namespace) -> None:
    """Execute the training pipeline."""
    logger.info("Loading config from %s", args.config)
    config = Config.from_yaml(args.config)
    config.output_dir = args.output_dir
    config.output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(config)
    if args.resume:
        logger.info("Resuming training from %s", args.resume)
        trainer.load_checkpoint(args.resume)

    logger.info("Starting training run")
    trainer.train()
    logger.info("Training complete. Checkpoints saved to %s", config.output_dir)


def run_infer(args: argparse.Namespace) -> None:
    """Execute the inference pipeline."""
    if args.prompt is None and args.input_file is None:
        logger.error("Provide either --prompt or --input-file for inference.")
        sys.exit(1)

    pipeline = InferencePipeline(
        model_path=str(args.model_path),
        max_new_tokens=args.max_new_tokens,
    )

    if args.prompt:
        result = pipeline.generate(args.prompt)
        print(result)
    else:
        logger.info("Processing prompts from %s", args.input_file)
        pipeline.process_file(args.input_file, args.output_file)
        logger.info("Predictions written to %s", args.output_file)


def main() -> None:
    """Main entry point."""
    args = parse_args()

    dispatch = {
        "train": run_train,
        "infer": run_infer,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        logger.error("Unknown command: %s", args.command)
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
