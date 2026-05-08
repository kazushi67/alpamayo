import json
from pathlib import Path
from typing import List, Any

import torch
from torch.utils.data import Dataset


class SavedSFTDataset(Dataset):
    """Load saved `.pt` samples produced for SFT/LoRA training.

    Behavior:
    - If `manifest` is provided and contains a file list under keys
      `files`, `samples` or `items`, those paths (relative to
      `dataset_dir`) are used in that order.
    - Otherwise falls back to glob pattern (default: `sample_*.pt`).

    Each sample is loaded with `torch.load(..., map_location='cpu')`
    and returned as-is (a dictionary). A helper `default_collate`
    is provided to assemble batches for training.
    """

    def __init__(self, dataset_dir: str, pattern: str = "sample_*.pt", manifest: str = None):
        self.dataset_dir = Path(dataset_dir)
        self.pattern = pattern
        self.manifest = manifest

        if manifest:
            try:
                with open(manifest, 'r') as f:
                    m = json.load(f)
                files = m.get('files') or m.get('samples') or m.get('items')
                if files and isinstance(files, list):
                    self.files = [self.dataset_dir / f for f in files]
                else:
                    self.files = sorted(self.dataset_dir.glob(self.pattern))
            except Exception:
                self.files = sorted(self.dataset_dir.glob(self.pattern))
        else:
            self.files = sorted(self.dataset_dir.glob(self.pattern))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Any:
        path = self.files[idx]
        sample = torch.load(path, map_location='cpu')
        return sample


def default_collate(batch: List[Any]):
    """A permissive collate that stacks tensors when possible.

    - If batch elements are dicts: collects all keys and for each key
      stacks tensors (torch.stack) when all values are tensors and
      shapes are compatible; otherwise keeps list of values.
    - If elements are tensors: returns torch.stack(batch, dim=0).
    - Otherwise returns the list as-is.
    """
    import torch

    if len(batch) == 0:
        return {}

    first = batch[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(batch, dim=0)

    if isinstance(first, dict):
        keys = set().union(*(b.keys() for b in batch))
        out = {}
        for k in keys:
            vals = [b.get(k) for b in batch]
            if all(isinstance(v, torch.Tensor) for v in vals):
                try:
                    out[k] = torch.stack(vals, dim=0)
                except Exception:
                    out[k] = vals
            elif all(isinstance(v, (int, float)) for v in vals):
                out[k] = torch.tensor(vals)
            else:
                out[k] = vals
        return out

    return batch


if __name__ == '__main__':
    # Quick local smoke test
    from torch.utils.data import DataLoader
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='206_dataset_ver1')
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()

    ds = SavedSFTDataset(args.dataset_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=default_collate)

    batch = next(iter(loader))
    if isinstance(batch, dict):
        print('Batch keys:', list(batch.keys()))
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f" - {k}: tensor {v.shape} {v.dtype}")
            else:
                print(f" - {k}: {type(v)} len={len(v) if hasattr(v,'__len__') else 'N/A'}")
    else:
        print('Batch type:', type(batch))
