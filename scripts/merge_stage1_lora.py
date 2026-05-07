#!/usr/bin/env python3
r"""Merge a Stage-1 LoRA adapter checkpoint into a full Alpamayo checkpoint.

Stage 1 training in this workspace can be run with PEFT/LoRA, which saves an
adapter checkpoint (for example ``adapter_model.safetensors`` +
``adapter_config.json``). Stage 2, however, expects a full checkpoint directory
that contains ``model.safetensors.index.json`` and sharded ``vlm.*`` weights.

This script:
  1. Loads the base Alpamayo checkpoint directory.
  2. Attaches the LoRA adapter from a Stage-1 output directory.
  3. Merges the adapter weights into the base model.
  4. Saves a full HuggingFace-style checkpoint directory with sharded
     ``*.safetensors`` files.
  5. Copies non-weight files such as tokenizer / processor assets from the base
     checkpoint directory.

Example::

    python scripts/merge_stage1_lora.py \
      --base-model-dir /home/yutotakeuchi/alpamayo/models \
      --adapter-dir /home/yutotakeuchi/alpamayo/output_stage1/checkpoint-114 \
      --output-dir /home/yutotakeuchi/alpamayo/output_stage1/merged_stage1

The resulting directory can be passed to Stage 2 as
``model.stage1_vlm_checkpoint_path``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from peft import PeftModel

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from finetune.sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1


_WEIGHT_SUFFIXES = (".bin", ".safetensors")


def _is_weight_file(name: str) -> bool:
    if name.endswith(_WEIGHT_SUFFIXES):
        return True
    return name.endswith(".index.json") and ("safetensors" in name or "pytorch_model" in name)


def _copy_non_weight_files(src_dir: Path, dst_dir: Path) -> list[str]:
    copied: list[str] = []
    for path in sorted(src_dir.iterdir()):
        if not path.is_file():
            continue
        if _is_weight_file(path.name):
            continue
        shutil.copy2(path, dst_dir / path.name)
        copied.append(path.name)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-dir", required=True, help="Base Alpamayo checkpoint dir")
    parser.add_argument("--adapter-dir", required=True, help="Stage-1 LoRA adapter dir")
    parser.add_argument("--output-dir", required=True, help="Merged checkpoint output dir")
    parser.add_argument(
        "--max-shard-size",
        default="4GB",
        help="Maximum safetensors shard size when saving the merged checkpoint.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing files in the output directory.",
    )
    args = parser.parse_args()

    base_model_dir = Path(args.base_model_dir)
    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (base_model_dir / "config.json").exists():
        raise FileNotFoundError(f"Missing config.json in base_model_dir: {base_model_dir}")
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing adapter_config.json in adapter_dir: {adapter_dir}")

    print(f"Loading base model from {base_model_dir} ...")
    model = TrainableAlpamayoR1.from_pretrained(
        str(base_model_dir),
        dtype="auto",
        cotrain_vlm=False,
        stage1_vlm_checkpoint_path=None,
    )

    print(f"Loading LoRA adapter from {adapter_dir} ...")
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    print("Merging adapter into base model ...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged checkpoint to {output_dir} ...")
    merged_model.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    copied = _copy_non_weight_files(base_model_dir, output_dir)
    print(f"Copied {len(copied)} non-weight files from base checkpoint.")
    print(
        "Done. Use this directory as model.stage1_vlm_checkpoint_path for stage 2: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
