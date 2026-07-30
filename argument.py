
import argparse
import json
import os
import torch
from typing import Dict, Any

class Config:
    """Unified configuration manager for MED-FSRU"""

    def __init__(self, **kwargs):
        self._set_defaults()
        for key, value in kwargs.items():

            setattr(self, key, value)

    def _set_defaults(self):
        # Data
        self.batch_size = 16
        self.accumulation_steps = 2
        self.image_size = 224
        self.num_workers = 2
        self.training_stage = "stage0"
        self.dataset_name = "pmc-vqa"
        self.debug_magnitudes = False


        # Model
        self.d_model = 768
        self.num_heads = 8
        self.num_layers = 12
        self.dropout = 0.1
        self.fsru_num_filter = 2
        self.fusion_type = "fsru"
        self.max_answer_len = 128
        self.vocab_size = 50265
        self.vision_backbone = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        self.early_vision_dim = 768
        self.early_hook_layer = 3
        self.text_backbone = "GanjinZero/biobart-v2-base"
        self.pmc_csv_path = "/content/pmc_vqa_full_with_images.csv"
        self.pmc_image_root =  "/content/pmc_vqa/images"
        self.max_epochs = 20

        self.lr = 1e-5
        self.finetune_lr = 5e-6
        self.warmup_epochs = 2
        self.patience = 5

        self.freeze_vision = True
        self.stage0_end_epoch = 1
        self.stage1_end_epoch      = 7
        self.unfreeze_encoder_epoch = 8
        self.rag_start_epoch        = 11
        self.finetune_epoch         = 15
        self.seed = 42
        self.use_amp = True
        self.use_rag = False    # RAG disabled for this run
        self.use_multimodal_rag_query = False    # unused while use_rag=False
        self.rag_query_image_weight   = 0.2      # unused while use_rag=False
        self.rag_lambda = 0.3                    # unused while use_rag=False           
        self.resume_checkpoint = None
        self.eval_only = False
        self.use_prefix_fusion = False           # prefix fusion disabled for this run
        self.prefix_start_epoch = None           # unused while use_prefix_fusion=False
        self.prefix_length = 16                  # unused while use_prefix_fusion=False
        self.prefix_dropout = 0.1                # unused while use_prefix_fusion=False
        self.retrieval_encoder_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"      # unused, reserved for future RAG
        self.rag_min_similarity = 0.35           # unused while use_rag=False


        self.top_k = 3                         # unused, RAG-only param
        self.max_rag_embeddings = 500          # unused, RAG-only param
        self.retrieval_metric = 'cosine'       # unused, RAG-only param
        self.rag_temperature = 0.5             # unused, RAG-only param
        self.rag_embeddings_path = "/content/pubmed_embeddings.npy"   # unused, RAG-only path
        self.rag_texts_path = "/content/pubmed_texts.pkl"             # unused, RAG-only path


        # Loss weights
        self.alpha = 0.1
        self.beta = 0.1

        # Paths (auto-detect environment)
        self.data_path = self._detect_data_path()
        self.output_path = self._detect_output_path()

    def _detect_data_path(self):
        """Auto-detect data path based on environment"""
        if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
            return '/kaggle/input'
        elif os.path.exists('/content'):
            return '/content/drive/MyDrive/visualquesansforqfsru'
        else:
            return './data/vqa-rad'

    def _detect_output_path(self):
        """Auto-detect output path"""
        if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
            return '/kaggle/working/checkpoints'
        elif os.path.exists('/content'):
            return '/content/drive/MyDrive/qfsru_checkpoints'
        else:
            return './checkpoints'

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        cfg = cls(**data)

        if hasattr(cfg, 'device') and cfg.device is not None:
            cfg.device = torch.device(cfg.device)
        return cfg

def parse_args() -> Config:
    """Parse command line arguments and return Config object"""
    parser = argparse.ArgumentParser(
        description='Q-FSRU: Medical VQA with Quantum RAG'
    )


    parser.add_argument('--config', type=str, help='Load config from JSON')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--accumulation_steps', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--max_epochs', type=int)
    parser.add_argument('--device', type=str)
    parser.add_argument('--top_k', type=int)      # RAG-only, unused
    parser.add_argument('--rag_embeddings_path', type=str)   # RAG-only, unused
    parser.add_argument('--training_stage', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    parser.add_argument('--eval_only',   action='store_true', default=None)
    parser.add_argument('--use_rag',     action='store_true', default=None)     # kept for future RAG re-enablement
    parser.add_argument('--freeze_vision', action='store_true', default=None)
    parser.add_argument('--seed',        type=int)
    parser.add_argument('--output_path', type=str)


    cmd_args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[parse_args] WARNING: unknown arguments ignored: {unknown}")

    if cmd_args.config:
        config = Config.load(cmd_args.config)
    else:
        config = Config()

    for key, value in vars(cmd_args).items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)

    # Set device
    if not hasattr(config, 'device') or config.device is None:
        config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        config.device = torch.device(config.device)

    return config
def get_vqarad_finetune_config(pretrained_checkpoint_path: str) -> Config:

    cfg = Config()

    # ── Stage control: skip warmup, go straight to full fine-tune ──
    cfg.stage0_end_epoch     = 0   # no projection warmup needed (already aligned)
    cfg.stage1_end_epoch     = 1   # no FSRU-only phase needed (already trained)
    cfg.rag_start_epoch      = 1   # RAG on from epoch 1
    cfg.finetune_epoch       = 3   # full fine-tune from epoch 1
    cfg.unfreeze_encoder_epoch = 1 # encoder unfrozen from epoch 1
    cfg.freeze_vision = True

    # ── LR: much lower — protect PMC-VQA learned representations ──
    cfg.lr           = 5e-6
    cfg.finetune_lr  = 1e-6

    # ── Training length: short — VQA-RAD is small (3515 samples) ──
    cfg.max_epochs   = 10
    cfg.patience     = 3
    cfg.batch_size   = 8     # smaller, VQA-RAD is tiny, avoid overfitting
    cfg.accumulation_steps = 4

    # ── Dataset: point to VQA-RAD ──
    cfg.dataset_name   = "vqa-rad"
    cfg.training_stage = "finetune"

    # ── Paths ──
    cfg.resume_checkpoint  = pretrained_checkpoint_path
    cfg.vqarad_csv_path    = "/content/vqa_rad/vqa_rad_train.json"
    cfg.vqarad_image_root  = "/content/vqa_rad/images"
    cfg.vqarad_test_path   = "/content/vqa_rad/vqa_rad_test.json"

    cfg.use_rag = False

    cfg.output_path = cfg.output_path + "/vqarad_finetune"

    return cfg


def get_slake_finetune_config(pretrained_checkpoint_path: str) -> Config:
    """
    Config for fine-tuning on SLAKE (secondary benchmark).
    SLAKE has 14k samples — slightly longer training than VQA-RAD.
    """
    cfg = Config()

    cfg.stage0_end_epoch     = 0
    cfg.stage1_end_epoch     = 0
    cfg.rag_start_epoch      = 1    #unused since rag is off
    cfg.finetune_epoch       = 1
    cfg.unfreeze_encoder_epoch = 1

    cfg.lr           = 5e-6
    cfg.finetune_lr  = 1e-6
    cfg.max_epochs   = 15
    cfg.patience     = 4
    cfg.batch_size   = 16
    cfg.accumulation_steps = 2

    cfg.dataset_name   = "slake"
    cfg.training_stage = "finetune"

    cfg.resume_checkpoint = pretrained_checkpoint_path
    cfg.slake_csv_path    = "/content/slake/train.json"
    cfg.slake_image_root  = "/content/slake/imgs"
    cfg.slake_test_path   = "/content/slake/test.json"

    # See note in get_vqarad_finetune_config — RAG defaults off.
    cfg.use_rag    = False
    cfg.output_path = cfg.output_path + "/slake_finetune"

    return cfg
# For direct import
if __name__ == "__main__":
    args = parse_args()
