# Med-FSRU

**A 2D-FFT-based cross-modal fusion mechanism for medical Visual Question Answering.**

Med-FSRU filters early- and late-layer visual features in the frequency domain and fuses them with text representations for generative medical VQA. This repository contains the training, fine-tuning, and evaluation code accompanying our AAAI submission.

## Overview

- **Vision encoder:** BiomedCLIP (ViT-B/16), with features tapped at both an early intermediate block and the final layer via forward hooks.
- **Frequency fusion (FSRU):** a 2D real FFT (`rfft2`) is applied over the 14×14 ViT patch grid, followed by a learned complex filter bank that mixes early- and late-frequency components before an inverse FFT projects back to the spatial/token domain.
- **Text decoder:** BioBARTv2, conditioned on the fused visual representation to generate free-form answers.
- **Optional retrieval augmentation (CosineRAG):** a cosine-similarity retrieval module over a PubMed-derived knowledge base (disabled by default in the released configs — see [Notes](#notes)).
- **Training:** a four-stage curriculum (projection warm-up → FSRU + decoder → RAG + encoder unfreeze → full fine-tune), driven by epoch thresholds in `argument.py`.

## Repository structure

```
Med-FSRU/
├── model.py              # VisionEncoder, FSRU integration, QFSRUModel (full architecture)
├── 2d_fsru.py             # 2D-FFT frequency filtering / gating modules (FSRU core)
├── cosine.py              # CosineRAG retrieval module
├── loss.py                # Focal loss / answer-type loss components
├── dataset.py              # MedicalVQAGenerativeDataset and dataloader construction
├── argument.py            # Config class, CLI parsing, benchmark-specific config presets
├── train.py                # Pretraining loop (PMC-VQA), curriculum staging, checkpointing
├── finetune/
│   ├── finetune_vqa-rad.py # Benchmark fine-tuning + multi-seed evaluation on VQA-RAD
│   └── finetune_slake.py   # Multi-seed fine-tuning driver for SLAKE
├── datasets/
│   ├── download_pmc_vqa.py     # Download + preprocess PMC-VQA (pretraining corpus)
│   ├── download_vqa_rad.py     # Download VQA-RAD via HuggingFace `datasets`
│   └── datasets/download_slake.py  # Download SLAKE images + splits
├── case study               # Qualitative analysis script (success/failure case figures)
└── eval_results.json        # Reported evaluation outputs (EM / F1, overall + open/closed splits)
```

## Setup

```bash
pip install torch torchvision transformers open_clip_torch datasets \
            huggingface_hub pandas numpy tqdm matplotlib requests
```

Tested with Python 3.10+ and PyTorch 2.x. A CUDA GPU is strongly recommended for training; `train.py` and the fine-tuning scripts auto-select `cuda` when available and fall back to CPU otherwise.

## Data

Three datasets are used: **PMC-VQA** (pretraining), **VQA-RAD**, and **SLAKE** (fine-tuning/evaluation benchmarks). Download scripts are provided under `datasets/`:

```bash
python datasets/download_pmc_vqa.py
python datasets/download_vqa_rad.py
python "datasets/datasets/download_slake.py"
```

By default these scripts write to `/content/...` paths (Colab/Kaggle conventions). If running locally, edit the `ROOT_DIR` / output paths at the top of each script, and update the corresponding paths in `argument.py` (`pmc_csv_path`, `pmc_image_root`, and the `vqarad_*` / `slake_*` fields in `get_vqarad_finetune_config` / `get_slake_finetune_config`).

## Training

Pretraining on PMC-VQA (four-stage curriculum, stage boundaries configured in `argument.py`):

```bash
python train.py --dataset_name pmc-vqa --batch_size 16 --max_epochs 20 --output_path ./checkpoints
```

Key flags (see `argument.py:parse_args` for the full list): `--lr`, `--training_stage`, `--resume_checkpoint`, `--eval_only`, `--freeze_vision`, `--seed`. Any field on `Config` can also be set via a JSON file passed to `--config`.

## Fine-tuning / benchmark evaluation

The fine-tuning scripts (`finetune/finetune_vqa-rad.py`, `finetune/finetune_slake.py`) run multi-seed (42 / 123 / 2024) fine-tuning from a pretrained checkpoint and report Exact Match (overall / open / closed) on the held-out test split. They are written to be run from the same process/notebook as `train.py` (they import shared classes via `from __main__ import Config`), so the intended workflow is:

1. Run pretraining with `train.py` to produce `pretrain_epoch_20.pt` (or chosen checkpoint).
2. Update the `checkpoint` path in `finetune_vqa-rad.py` / `finetune_slake.py` to point at that checkpoint.
3. Run the fine-tuning script in the same environment where `Config` and the model/dataloader code are already defined (e.g. execute `train.py`'s definitions first, or adapt the `from __main__ import Config` import to `from argument import Config` for standalone execution).

```bash
python finetune/finetune_vqa-rad.py
python finetune/finetune_slake.py
```


## Notes

- Retrieval augmentation (`CosineRAG`, `use_rag`) and the alternative prefix-fusion path (`use_prefix_fusion`) are implemented but disabled in the released configs; the reported results use the FSRU fusion path only.
- Paths in `argument.py` (`Config._detect_data_path` / `_detect_output_path`) auto-switch between Kaggle, Colab, and local conventions — override `--output_path` / config fields as needed for your environment.

## Citation

If you use this code, please cite our paper (details to be added upon publication).
