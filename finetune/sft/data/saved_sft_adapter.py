from typing import Any, Dict

try:
    from hydra.utils import instantiate
except Exception:  # pragma: no cover - fallback for smoke tests without hydra installed
    instantiate = None

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - fallback for smoke tests without omegaconf installed
    OmegaConf = None
from torch.utils.data import Dataset
import torch

from finetune.sft.data.saved_sft_dataset import SavedSFTDataset


class SavedSFTAdapter(Dataset):
    """Adapter that maps saved sample dicts to the keys expected by the SFT trainer.

    It wraps `SavedSFTDataset` and in `__getitem__` returns a dictionary containing at least
    the commonly required keys. Missing keys are left out but the adapter will convert
    common alternatives and ensure tensors are of torch.Tensor type.
    """

    EXPECTED_KEYS = [
        'image_frames',
        'camera_indices',
        'ego_history_xyz',
        'ego_history_rot',
        'ego_future_xyz',
        'ego_future_rot',
        'absolute_timestamps',
        'relative_timestamps',
        't0_us',
        'clip_id',
        'text',
        'tokenized_text',
    ]

    def __init__(
        self,
        dataset_dir: str | None = None,
        local_dir: str | None = None,
        pattern: str = "sample_*.pt",
        manifest: str = None,
        file_start: int | None = None,
        file_end: int | None = None,
        model_config: Any | None = None,
        vla_preprocess_args: dict | None = None,
        chunk_ids: Any | None = None,
        use_default_keyframe: bool | None = None,
        **_: Any,
    ):
        del chunk_ids, use_default_keyframe

        dataset_root = dataset_dir or local_dir
        if dataset_root is None:
            raise ValueError("Either dataset_dir or local_dir must be provided")

        self.inner = SavedSFTDataset(dataset_root, pattern=pattern, manifest=manifest)
        files = list(self.inner.files)
        if file_start is not None or file_end is not None:
            self.inner.files = files[file_start:file_end]
        # simple on-the-fly tokenizer state
        self._vocab = {"<pad>": 0, "<unk>": 1}
        self._vocab_next = 2

        self.vla_preprocess_func = None
        if model_config is not None and isinstance(model_config, dict) and OmegaConf is not None:
            model_config = OmegaConf.create(model_config)
        if vla_preprocess_args is not None:
            if instantiate is None:
                raise ImportError("hydra is required when vla_preprocess_args is provided")
            self.vla_preprocess_func = instantiate(vla_preprocess_args, model_config=model_config)

    def __len__(self):
        return len(self.inner)

    def _to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x
        try:
            return torch.as_tensor(x)
        except Exception:
            return x

    def _simple_tokenize(self, text: str):
        """Very small whitespace tokenizer that assigns incremental ids.

        Keeps a small vocabulary in memory for deterministic mapping during a run.
        """
        if not isinstance(text, str):
            return None
        toks = text.strip().split()
        ids = []
        for t in toks:
            if t in self._vocab:
                ids.append(self._vocab[t])
            else:
                self._vocab[t] = self._vocab_next
                ids.append(self._vocab_next)
                self._vocab_next += 1
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx) -> Dict[str, Any]:
        s = self.inner[idx]
        out = {}

        # Direct pass-through for common keys
        for k in self.EXPECTED_KEYS:
            if k in s:
                out[k] = self._to_tensor(s[k]) if k not in ('clip_id', 'text', 'tokenized_text') else s[k]

        # Fallback name mappings
        if 'images' in s and 'image_frames' not in out:
            out['image_frames'] = self._to_tensor(s['images'])
        if 'frames' in s and 'image_frames' not in out:
            out['image_frames'] = self._to_tensor(s['frames'])

        if 'clip_id' not in out:
            if 'id' in s:
                out['clip_id'] = s['id']
            elif 'clip' in s:
                out['clip_id'] = s['clip']

        # Ensure camera_indices exists as tensor
        if 'camera_indices' in out and not isinstance(out['camera_indices'], torch.Tensor):
            out['camera_indices'] = self._to_tensor(out['camera_indices']).long()

        # Ensure types for motion tensors
        for k in ('ego_history_xyz', 'ego_future_xyz'):
            if k in out and isinstance(out[k], torch.Tensor):
                out[k] = out[k].float()

        for k in ('ego_history_rot', 'ego_future_rot'):
            if k in out and isinstance(out[k], torch.Tensor):
                out[k] = out[k].float()

        # image_frames as uint8 -> keep as is or convert to float later in preprocessing
        if 'image_frames' in out and isinstance(out['image_frames'], torch.Tensor):
            # keep dtype as-is; convert to float32 in model preprocess if needed
            pass

        # Create tokenized_text if text exists but tokenized_text missing
        if 'tokenized_text' not in out:
            if 'text' in out and isinstance(out['text'], str):
                tok = self._simple_tokenize(out['text'])
                if tok is not None:
                    out['tokenized_text'] = tok

        # Create a numeric clip id index for bookkeeping (stable within this process)
        if 'clip_id' in out and 'clip_id_idx' not in out:
            cid = out['clip_id']
            # if clip_id is a list (batch) keep as-is, else compute single id
            try:
                if isinstance(cid, list):
                    # compute per-sample hashed index
                    idxs = []
                    for c in cid:
                        idxs.append(torch.tensor([(hash(c) & ((1 << 63) - 1))], dtype=torch.long))
                    out['clip_id_idx'] = torch.stack(idxs, dim=0).squeeze(-1)
                else:
                    out['clip_id_idx'] = torch.tensor([(hash(cid) & ((1 << 63) - 1))], dtype=torch.long)
            except Exception:
                pass

        if self.vla_preprocess_func is not None:
            # Match PAIDataset behavior: preprocess mutates the sample dict and returns tokenized_data
            out['tokenized_data'] = self.vla_preprocess_func(data=out)

        return out


if __name__ == '__main__':
    # Smoke test
    from torch.utils.data import DataLoader
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='206_dataset_ver1')
    args = parser.parse_args()

    ds = SavedSFTAdapter(args.dataset_dir)
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    print('Adapter batch keys:', list(batch.keys()))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f" - {k}: {v.shape} {v.dtype}")
        else:
            print(f" - {k}: {type(v)}")
