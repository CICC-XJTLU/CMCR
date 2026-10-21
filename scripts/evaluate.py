"""Evaluate a saved CMCR classifier on a JSONL split."""

import argparse
import json
from pathlib import Path

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from cmcr.data import Collator, read_rows
from cmcr.model import build_model
from scripts.train import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--base-model", help="Override the base-model path saved in the checkpoint.")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    config = json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
    if args.base_model is not None:
        config["base_model"] = args.base_model
    data_path = args.data if args.data is not None else Path(config["test_data"])
    processor = AutoProcessor.from_pretrained(config["base_model"], local_files_only=True)
    device = torch.device(args.device)
    model = build_model(config).to(device)
    set_peft_model_state_dict(
        model.backbone,
        load_file(str(args.checkpoint / "adapter" / "adapter_model.safetensors")),
    )
    model.load_state_dict(
        torch.load(args.checkpoint / "module.pt", map_location="cpu", weights_only=True),
        strict=False,
    )
    loader = DataLoader(
        read_rows(data_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=Collator(processor),
    )
    result = evaluate(model, loader, device, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
