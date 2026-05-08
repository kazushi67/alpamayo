from torch.utils.data import DataLoader
from saved_sft_dataset import SavedSFTDataset, default_collate


def demo(dataset_dir: str = "../206_dataset_ver1"):
    ds = SavedSFTDataset(dataset_dir)
    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=default_collate)
    batch = next(iter(loader))
    print('Loaded 1 batch from', dataset_dir)
    if isinstance(batch, dict):
        for k, v in batch.items():
            if hasattr(v, 'shape'):
                print(f" - {k}: {v.shape} {getattr(v,'dtype',None)}")
            else:
                print(f" - {k}: {type(v)} len={len(v) if hasattr(v,'__len__') else 'N/A'}")


if __name__ == '__main__':
    demo()
