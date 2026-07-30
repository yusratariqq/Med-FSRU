
import numpy as np
import os
import json
import torch
import pandas as pd
import re
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Dict, List
from transformers import AutoTokenizer
import torchvision.transforms as T
import random
from transformers import AutoImageProcessor
ANSWER_TYPE_MAP ={
    "yes/no": 0,
    "open": 1,
    "other": 2
}

def get_transforms(is_training: bool):

    if is_training:
        return T.Compose([
            T.RandomRotation(10),
        ])
    else:
        return None  # processor handles resize deterministically


# ============================================================
# Dataset
# ============================================================

class MedicalVQAGenerativeDataset(Dataset):

    def __init__(self, args, split: str):
        """
        Args:
            args : Config object from arguments.py
            split: 'train', 'val', or 'test'
        """
        self.args = args
        self.split = split
        self.max_answer_len = args.max_answer_len
        self.augment = get_transforms(is_training=(split == 'train'))
        if args.dataset_name == "pmc-vqa":
            self.data = self._load_pmc_annotations()
        elif args.dataset_name == "vqa-rad":
            self.data = self._load_vqa_rad_annotations()
        elif args.dataset_name == "slake":
            self.data = self._load_slake_annotations()
        else:
            raise ValueError(
                f"Unknown dataset_name: '{args.dataset_name}'. "
                f"Choose from: pmc-vqa, vqa-rad, slake"
            )
        all_valid = []
        missing_count = 0
        for i, item in enumerate(self.data):
            img_path = self._resolve_image_path(item)
            exists = img_path is not None and os.path.exists(img_path)
            if exists:
                all_valid.append(i)
            elif args.dataset_name != "pmc-vqa":
                all_valid.append(i)          # keep with fallback gray image
                missing_count += 1
            else:
                missing_count += 1
        pct = 100.0 * missing_count / max(len(self.data), 1)
        if missing_count > 0:
            print(f"[Dataset WARNING] {missing_count} missing images ({pct:.1f}%) in split='{split}'")
            if args.dataset_name == "pmc-vqa" and pct > 20.0:
                raise RuntimeError(
                    f"[Dataset Error] {pct:.1f}% of PMC-VQA images could not be resolved. "
                    f"Check pmc_image_root='{args.pmc_image_root}' before pretraining."
                )
        if split != "train" and pct > 5.0:
            raise RuntimeError(
                f"[Dataset Error] {pct:.1f}% of {split} images are missing. "
                f"Benchmark metrics would be invalid. Fix image paths before evaluating."
            )
        if split == "train" and pct > 20.0:
            raise RuntimeError(
                f"[Dataset Error] {pct:.1f}% of training images are missing. "
                f"This looks like a misconfigured image_root path, not a few "
                f"isolated missing files. Fix the path before training."
            )

        # PMC-VQA: split 95/5 train/val here (annotations file has no preset split)
        # VQA-RAD and SLAKE: split already done in annotation loaders
        if args.dataset_name == "pmc-vqa":
            split_point = int(0.95 * len(all_valid))
            if split == "train":
                self.valid_indices = all_valid[:split_point]
            elif split == "val":
                self.valid_indices = all_valid[split_point:]
            else:
                self.valid_indices = all_valid
        else:
            # VQA-RAD / SLAKE already split in loader — use all valid
            self.valid_indices = all_valid
        if len(self.valid_indices) == 0:
          raise RuntimeError(
              f"[Dataset Error] No valid samples found for split='{self.split}'. "
              f"Check PMC image paths and CSV consistency."
          )
        print(
            f"[Dataset] {self.split}: using {len(self.valid_indices)} / {len(all_valid)} valid samples"
        )
        # At the end of __init__(), after building valid_indices:
        if len(self.valid_indices) > 10:
            # Sample 5 paths and verify they exist
            import random as _rnd
            sample_idx = _rnd.sample(self.valid_indices[:100], min(5, len(self.valid_indices)))
            missing = 0
            for idx in sample_idx:
                path = self._resolve_image_path(self.data[idx])
                if path and not os.path.exists(path):
                    missing += 1
            if missing > 2:
                print(f"[Dataset WARNING] {missing}/5 sampled image paths do not exist.")
                print(f"  Example: {self._resolve_image_path(self.data[sample_idx[0]])}")
                print(f"  Check pmc_image_root and Figure_path prefix.")

        text_backbone = getattr(args, 'text_backbone', "GanjinZero/biobart-v2-base")
        if not hasattr(args, '_tokenizer') or args._tokenizer is None:
            args._tokenizer = AutoTokenizer.from_pretrained(text_backbone)
            if args._tokenizer.pad_token is None:
                args._tokenizer.add_special_tokens({"pad_token": "<pad>"})
            # Set vocab_size once from the authoritative tokenizer
            args.vocab_size = len(args._tokenizer)

        # Processor assignment used to happen only inside the
        # tokenizer-creation branch above. If args._tokenizer already
        # existed, args._processor was never set, and
        # self.processor = args._processor below would raise
        # AttributeError. This check is now separate and unconditional.
        if not hasattr(args, '_processor') or args._processor is None:
            if not hasattr(args, '_image_preprocess') or args._image_preprocess is None:
                raise RuntimeError(
                    "args._image_preprocess is not set. Construct VisionEncoder "
                    "and assign args._image_preprocess = vision_encoder.processor "
                    "before building the dataset."
                )
            args._processor = args._image_preprocess

        self.tokenizer = args._tokenizer
        self.processor = args._processor

    def _load_vqa_rad_annotations(self):
        """
        Load VQA-RAD using the OFFICIAL train/test split.

        VQA-RAD was constructed so that test questions come from different
        images than training questions. Random re-splitting breaks this
        guarantee and inflates val metrics by leaking image context.

        Expected file layout (set in get_vqarad_finetune_config):
            vqarad_csv_path  → trainset.json   (train split, official)
            vqarad_test_path → testset.json    (test split, official)

        Val is carved from train (15%) using image-level grouping to prevent
        questions from the same image appearing in both train and val.
        """
        if self.split == "test":
            json_path = getattr(self.args, "vqarad_test_path", None)
            if json_path is None:
                raise ValueError("vqarad_test_path not set in args")
            with open(json_path, "r") as f:
                data = json.load(f)
            return self._normalize_vqarad(data)

        # Train or val: load from the official train file
        json_path = getattr(self.args, "vqarad_csv_path", None)
        if json_path is None:
            raise ValueError("vqarad_csv_path not set in args")
        with open(json_path, "r") as f:
            data = json.load(f)
        data = self._normalize_vqarad(data)

        # Image-level split: group questions by image, split images 85/15.
        # All questions from the same image go to the SAME split — this is
        # the key guarantee that prevents val contamination.
        from collections import defaultdict
        image_groups = defaultdict(list)
        for item in data:
            image_groups[item["image_name"]].append(item)

        image_names = sorted(image_groups.keys())   # sort for reproducibility
        _rng = random.Random(self.args.seed)
        _rng.shuffle(image_names)

        split_point  = int(0.85 * len(image_names))
        train_images = set(image_names[:split_point])
        val_images   = set(image_names[split_point:])

        if self.split == "train":
            result = [item for name in train_images for item in image_groups[name]]
            print(f"[VQA-RAD] Train: {len(train_images)} images, {len(result)} questions")
            self._log_answer_type_distribution(result, "train")
            return result
        elif self.split == "val":
            result = [item for name in val_images for item in image_groups[name]]
            print(f"[VQA-RAD] Val: {len(val_images)} images, {len(result)} questions")
            self._log_answer_type_distribution(result, "val")
            return result
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def _normalize_vqarad(self, data):
        """Normalize VQA-RAD JSON keys to the pipeline's expected format."""
        normalized = []
        for d in data:
            answer_type = d.get("answer_type", "open").lower()
            if answer_type in ("closed", "yes/no"):
                answer_type = "yes/no"
            elif answer_type not in ANSWER_TYPE_MAP:
                answer_type = "open"
            normalized.append({
                "image_name":  d.get("image_name", d.get("image_org_id", "")),
                "question":    d.get("question", ""),
                "answer":      str(d.get("answer", "")),
                "answer_type": answer_type,
            })
        return normalized
    def _normalize_slake(self, data):
        normalized = []
        for d in data:
            normalized.append({
                "image_name":  d.get("img_name", d.get("image_name", "")),
                "question":    d.get("question", ""),
                "answer":      str(d.get("answer", "")),
                "answer_type": d.get("answer_type", "open").lower(),
            })
        return normalized
    def _log_answer_type_distribution(self, data: list, split_name: str):
        """
        Logs answer-type counts and percentages for a split.

        Expected SLAKE English distribution: closed ~40-45%, open ~55-60%.
        Expected VQA-RAD distribution:       yes/no ~57%, open ~43%.

        A split deviating more than 10pp from the full-dataset ratio
        suggests a stratification problem worth investigating before training.
        Call this after any split construction to verify balance.
        """
        from collections import Counter
        type_counts = Counter(
            item.get("answer_type", "open") for item in data
        )
        total = max(len(data), 1)

        print(f"[{self.args.dataset_name.upper()} {split_name}] "
              f"Answer-type distribution ({total} samples):")
        for atype, count in sorted(type_counts.items()):
            pct = 100.0 * count / total
            print(f"  {atype:<12}: {count:>5}  ({pct:.1f}%)")

        # Warn if closed/open ratio is severely skewed
        closed = type_counts.get("closed", 0) + type_counts.get("yes/no", 0)
        open_  = type_counts.get("open", 0)
        if total > 50 and open_ > 0:
            ratio = closed / open_
            if ratio < 0.3 or ratio > 2.0:
                print(
                    f"  ⚠️  closed/open ratio = {ratio:.2f} — "
                    f"distribution is heavily skewed. "
                    f"Check that stratified split ran correctly."
                )
            else:
                print(f"  ✅ closed/open ratio = {ratio:.2f} — distribution looks balanced.")
    def _load_slake_annotations(self):
        """
        Load SLAKE dataset annotations.
        SLAKE JSON format: list of dicts with keys:
          img_name, question, answer, answer_type, q_lang
        We use English questions only (q_lang == 'en').
        """
        if self.split == "test":
            json_path = getattr(self.args, "slake_test_path", None)
        else:
            json_path = getattr(self.args, "slake_csv_path", None)

        if json_path is None:
            raise ValueError("slake_csv_path / slake_test_path not set in args")

        with open(json_path, "r") as f:
            data = json.load(f)

        # SLAKE is bilingual — keeping English only
        data = [d for d in data if d.get("q_lang", "en") == "en"]

        # Normalize keys to match our pipeline
        normalized = []
        for d in data:
            normalized.append({
                "image_name":   d.get("img_name", d.get("image_name", "")),
                "question":     d.get("question", ""),
                "answer":       str(d.get("answer", "")),
                "answer_type":  d.get("answer_type", "open").lower(),
            })

        # SLAKE ships three separate JSON files: train.json, validate.json, test.json.
        # Using them directly, NOT re-splitting the training file.
        # The finetune config should set:
        #   slake_csv_path  → train.json
        #   slake_val_path  → validate.json  
        #   slake_test_path → test.json
        if self.split == "test":
            self._log_answer_type_distribution(normalized, "test")
            return normalized

        if self.split == "val":
            # Load from the official val file if it exists.
            # If found: return immediately — official split is already balanced.
            # If not found: fall through to stratified split below.
            val_path = getattr(self.args, "slake_val_path", None)
            if val_path is not None and os.path.exists(val_path):
                with open(val_path, "r") as f:
                    val_data = json.load(f)
                val_data = [d for d in val_data if d.get("q_lang", "en") == "en"]
                normalized_val = self._normalize_slake(val_data)
                self._log_answer_type_distribution(normalized_val, "val")
                return normalized_val
            # No return here — fall through to stratified split below
            print("[SLAKE] slake_val_path not set — carving val from train (stratified)")

        # ── Stratified split by answer type + image grouping ─────────────
        # Groups by (answer_type, image_name) so:
        #   1. Questions from the same image stay in the same split
        #   2. Train and val have the same closed/open ratio as full dataset
        from collections import defaultdict

        type_image_groups = defaultdict(lambda: defaultdict(list))
        for item in normalized:
            atype = item.get("answer_type", "open")
            iname = item.get("image_name", "unknown")
            type_image_groups[atype][iname].append(item)

        train_items = []
        val_items   = []

        for atype, image_groups in type_image_groups.items():
            image_names  = sorted(image_groups.keys())
            _rng         = random.Random(self.args.seed)
            _rng.shuffle(image_names)
            split_point  = int(0.85 * len(image_names))
            train_images = set(image_names[:split_point])
            val_images   = set(image_names[split_point:])

            for iname in train_images:
                train_items.extend(image_groups[iname])
            for iname in val_images:
                val_items.extend(image_groups[iname])

        _rng_final = random.Random(self.args.seed + 1)
        _rng_final.shuffle(train_items)
        _rng_final.shuffle(val_items)

        if self.split == "train":
            self._log_answer_type_distribution(train_items, "train")
            return train_items
        else:
            self._log_answer_type_distribution(val_items, "val")
            return val_items

    def _load_pmc_annotations(self):
      df = pd.read_csv(self.args.pmc_csv_path, low_memory=False)
      def has_valid_answer(row):
        answer = str(row["Answer"]).strip()
        if answer in ["A", "B", "C", "D"]:
          choice_col = f"Choice {answer}"
          if choice_col in df.columns:
            return pd.notna(row[choice_col]) and len(str(row[choice_col]).strip()) > 0
          return False
        return True  # Non-letter answers are valid
      before_count = len(df)
      df = df[df.apply(has_valid_answer, axis=1)].reset_index(drop=True)
      print(f"[Dataset] Filtered PMC-VQA: {before_count} → {len(df)} rows (removed {before_count - len(df)} with missing choices)")

      df = df.sample(frac=1, random_state=self.args.seed).reset_index(drop=True)
      return df.to_dict("records")
    def _resolve_answer_and_type(self, item):
        """Shared by __getitem__ and get_dataloaders' sampler-weight builder,
        so the two can never diverge on what counts as yes/no vs open vs other."""
        if self.args.dataset_name == "pmc-vqa":
            answer_raw = item["Answer"]
            if answer_raw in ["A", "B", "C", "D"]:
                choice_col = f"Choice {answer_raw}"
                if choice_col in item and item[choice_col] is not None:
                    answer = re.sub(r'^[A-D]\s*:\s*', '', str(item[choice_col]).strip())
                else:
                    answer = answer_raw
            else:
                answer = re.sub(r'^[A-D]\s*:\s*', '', str(answer_raw).strip())

            answer_lower = answer.lower().strip()
            _YES_TOKENS = {"yes", "true", "correct", "present", "positive",
                           "increased", "elevated", "normal"}
            _NO_TOKENS  = {"no", "false", "absent", "negative", "decreased",
                           "not present", "none", "unremarkable"}
            if answer_lower in _YES_TOKENS or answer_lower in _NO_TOKENS:
                answer_type = "yes/no"
            elif len(answer_lower.split()) <= 3:
                answer_type = "open"
            else:
                answer_type = "other"
        else:
            answer = str(item["answer"]).strip()
            answer_type = item.get("answer_type", "open").lower()
            if answer_type in ("closed", "yes/no"):
                answer_type = "yes/no"
            elif answer_type not in ANSWER_TYPE_MAP:
                answer_type = "open"
        return answer, answer_type
    # --------------------------------------------------------
    # Image loading
    # --------------------------------------------------------
    def _resolve_image_path(self, item: dict) -> str:
        """
        Returns the full image path for any dataset format.
        Returns None if the required key is missing.
        """
        if self.args.dataset_name == "pmc-vqa":
            fig = item.get("Figure_path")
            if fig is None:
                return None
            fig_clean = os.path.basename(fig)
            return os.path.join(self.args.pmc_image_root, fig_clean)

        elif self.args.dataset_name == "vqa-rad":
            img_root = getattr(
                self.args, "vqarad_image_root",
                os.path.join(self.args.data_path, "VQA_RAD Image Folder")
            )
            return os.path.join(img_root, item.get("image_name", ""))

        elif self.args.dataset_name == "slake":
            img_root = getattr(self.args, "slake_image_root", self.args.data_path)
            return os.path.join(img_root, item.get("image_name", ""))

        return None

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    def _tokenize_question(self, question: str) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0)
        }

    def _tokenize_answer(self, answer):
        """Safely tokenize answer text."""
        if answer is None:
            answer = ""
        elif isinstance(answer, (int, float)):
            answer = str(answer)
        elif isinstance(answer, list):
            answer = " ".join([str(a) for a in answer])
        else:
            answer = str(answer)

        enc = self.tokenizer(
            answer,
            max_length=self.max_answer_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        labels = enc["input_ids"].clone()  # [1, L] — keep BOS; HF's internal
        # shift_tokens_right() builds decoder_input_ids from this correctly,
        # and safely replaces -100 with pad_token_id before shifting.

        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        # Safety: ensure at least one token is unmasked
        if (labels == -100).all():
            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None:
                eos_pos = (enc["input_ids"][0] == eos_id).nonzero()
                if len(eos_pos) > 0:
                    pos = max(0, eos_pos[0].item() - 1)
                    labels[0, pos] = enc["input_ids"][0, eos_pos[0].item()]

        return {
            "input_ids":      labels.squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

    # --------------------------------------------------------
    # PyTorch hooks
    # --------------------------------------------------------

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        real_idx = self.valid_indices[idx]
        item     = self.data[real_idx]

        # ── Resolve image path ───────────────────────────────────────────
        img_path = self._resolve_image_path(item)

        question = item["Question"] if self.args.dataset_name == "pmc-vqa" else item["question"]
        answer, answer_type = self._resolve_answer_and_type(item)

        try:
            img = Image.open(img_path).convert("RGB")
            if self.split == "train" and self.augment is not None:
                img = self.augment(img)  # flip + rotate only; processor handles resize
        except Exception as e:
            print(f"[WARN] Image load failed: {img_path} | {e}")
            img = Image.new("RGB", (224, 224), (128, 128, 128))

        pixel_values = self.processor(img)

        # ── Tokenize ─────────────────────────────────────────────────────
        q = self._tokenize_question(question)
        a = self._tokenize_answer(answer)

        answer_type_id = ANSWER_TYPE_MAP.get(answer_type, ANSWER_TYPE_MAP["other"])

        return {
            "images":          pixel_values,
            "question_ids":    q["input_ids"],
            "question_mask":   q["attention_mask"],
            "answer_ids":      a["input_ids"],
            "answer_mask":     a["attention_mask"],
            "answer_text":     answer,
            "question_text":   question,
            "answer_type":     answer_type,
            "answer_type_id":  answer_type_id,
            # finetune.py run_benchmark_evaluation uses this key
            "question_type":   answer_type,
        }

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {
        "images":          torch.stack([b["images"] for b in batch]),
        "question_ids":    torch.stack([b["question_ids"] for b in batch]),
        "question_mask":   torch.stack([b["question_mask"] for b in batch]),
        "answer_ids":      torch.stack([b["answer_ids"] for b in batch]),
        "answer_mask":     torch.stack([b["answer_mask"] for b in batch]),
        "answer_texts":    [b["answer_text"]   for b in batch],
        "question_texts":  [b["question_text"] for b in batch],
        "answer_type":     [b["answer_type"]   for b in batch],
        # Used by finetune.py run_benchmark_evaluation
        "question_type":   [b["question_type"] for b in batch],
        "answer_type_ids": torch.tensor(
            [b["answer_type_id"] for b in batch],
            dtype=torch.long
        ),
    }


# ============================================================
# Dataloader Factory
# ============================================================

def get_dataloaders(args):
    # PMC-VQA has no preset test split (used for pretraining only)
    # VQA-RAD and SLAKE have test splits for benchmark evaluation
    if args.dataset_name == "pmc-vqa":
        splits = ["train", "val"]
    else:
        splits = ["train", "val", "test"]

    datasets = {}
    # AFTER
    for split in splits:
        try:
            datasets[split] = MedicalVQAGenerativeDataset(args, split)
        except Exception as e:
            if split in ("train", "val"):
                raise   # train/val are required — don't limp forward silently
            print(f"[DataLoader] ⚠️  Could not load split='{split}': {e}")
    from torch.utils.data import WeightedRandomSampler

    def _make_loader(split, ds, args):
        if split != "train" or args.dataset_name not in ("vqa-rad", "slake", "pmc-vqa"):
            return DataLoader(
                ds,
                batch_size=max(1, args.batch_size // 2) if split != "train" else args.batch_size,
                shuffle=(split == "train"),
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
                drop_last=(split == "train"),
            )

        # Per-sample answer_type via the same heuristic __getitem__ uses —
        # no image loading, just metadata + regex, so this is cheap even
        # over the full ~294k-row PMC-VQA train split.
        type_list = [
            ds._resolve_answer_and_type(ds.data[ds.valid_indices[i]])[1]
            for i in range(len(ds))
        ]
        from collections import Counter
        type_counts = Counter(type_list)
        print(f"[Sampler] {args.dataset_name} train type counts: {dict(type_counts)}")

        # Inverse-frequency weight per class, capped the same way the loss
        # weighting is capped, so exposure and loss weighting pull in the
        # same direction rather than fighting each other.
        max_count   = max(type_counts.values())
        cap_ratio   = 20.0
        # simplify: cap each weight at cap_ratio × the majority class's weight (1.0)
        type_weight = {t: min(max_count / c, cap_ratio) for t, c in type_counts.items()}
        weights = [type_weight[t] for t in type_list]
        print(f"[Sampler] Per-type weights: {type_weight}")

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
        )

    loaders = {split: _make_loader(split, ds, args) for split, ds in datasets.items()}

    print(f"[DataLoader] Splits loaded: {list(loaders.keys())}")
    for split, loader in loaders.items():
        print(f"  {split}: {len(loader.dataset)} samples, "
              f"{len(loader)} batches (batch_size={loader.batch_size})")

    return loaders
