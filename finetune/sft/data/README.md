SavedSFTDataset
================

このディレクトリには、外部で作成した `sample_*.pt` ファイルをそのまま読み込むための
`SavedSFTDataset` とサンプルローダーが入っています。

推奨する起点は次の 2 つです。

- [saved_sft_stage1.yaml](../configs/saved_sft_stage1.yaml): stage1 SFT 用。`sample_00000.pt` 〜 `sample_00059.pt` を train、`sample_00060.pt` 〜 `sample_00075.pt` を valid に分割します。
- [saved_sft_stage2.yaml](../configs/saved_sft_stage2.yaml): stage2 SFT 用。分割ルールは stage1 と同じです。

もし独自に組み替える場合は、`SavedSFTAdapter` の `local_dir` と `file_start` / `file_end` を変えるだけで分割できます。

LoRA を使って stage1 を回した場合は、stage2 の前に adapter をマージしてください。

```bash
cd /home/yutotakeuchi/alpamayo
./ar1_venv/bin/python scripts/merge_stage1_lora.py \
  --base-model-dir /absolute/path/to/Alpamayo-R1-10B \
  --adapter-dir /absolute/path/to/output_stage1_saved/checkpoint-xxx \
  --output-dir /absolute/path/to/output_stage1_saved/merged_stage1
```

そのうえで、[saved_sft_stage2.yaml](../configs/saved_sft_stage2.yaml) の `model.stage1_vlm_checkpoint_path` をマージ済みディレクトリに置き換えてから stage2 を起動してください。

実行コマンド例

1GPU で stage1 を回す場合:

```bash
cd /home/yutotakeuchi/alpamayo
./ar1_venv/bin/torchrun --nproc_per_node 1 -m finetune.sft.train_hf \
  --config-path pkg://finetune/sft/configs \
  --config-name saved_sft_stage1
```

1GPU で stage2 を回す場合:

```bash
cd /home/yutotakeuchi/alpamayo
./ar1_venv/bin/torchrun --nproc_per_node 1 -m finetune.sft.train_hf \
  --config-path pkg://finetune/sft/configs \
  --config-name saved_sft_stage2
```

データローダーだけ確認したい場合:

```bash
cd /home/yutotakeuchi/alpamayo
PYTHONPATH=. python3 finetune/sft/data/saved_sft_dataset.py --dataset_dir 206_dataset_ver1 --batch_size 2
PYTHONPATH=. python3 finetune/sft/data/saved_sft_adapter.py --dataset_dir 206_dataset_ver1
```

使い方（簡易）:

- 単体実行:

```bash
python finetune/sft/data/saved_sft_dataset.py --dataset_dir path/to/206_dataset_ver1 --batch_size 2
```

- 例: `finetune/sft/configs` にある設定を上書きしてトレーニングに使う場合は、`data.train_dataset` を次のように設定してください。

```yaml
data:
  train_dataset:
    _recursive_: false
    _target_: finetune.sft.data.saved_sft_adapter.SavedSFTAdapter
    local_dir: /absolute/path/to/206_dataset_ver1
    file_start: 0
    file_end: 60
    use_default_keyframe: true
```

注意:
- `sample_*.pt` の中身（辞書のキー構成）によってはトレーナー側で期待されるキーに変換が必要です。
- まずは `python finetune/sft/data/saved_sft_dataset.py` で中身を確認してください。
- config をそのまま使うなら、`saved_sft_stage1.yaml` と `saved_sft_stage2.yaml` を使うのが最短です。
