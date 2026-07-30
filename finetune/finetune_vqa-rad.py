
# BENCHMARK FINE-TUNING: VQA-RAD / SLAKE

import os, sys, json, torch, argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from __main__ import Config

def ft_set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_vqarad_finetune_config(checkpoint_path):
    cfg = Config()
    cfg.stage0_end_epoch       = 0
    cfg.stage1_end_epoch       = 1
    cfg.rag_start_epoch        = 1
    cfg.finetune_epoch         = 3
    cfg.unfreeze_encoder_epoch = 2
    cfg.freeze_vision          = True
    cfg.lr                     = 5e-6
    cfg.finetune_lr            = 1e-6
    cfg.max_epochs             = 10
    cfg.patience               = 3
    cfg.batch_size             = 8
    cfg.accumulation_steps     = 4
    cfg.dataset_name           = "vqa-rad"
    cfg.training_stage         = "finetune"
    cfg.resume_checkpoint      = checkpoint_path
    cfg.vqarad_csv_path        = "/content/vqa_rad/vqa_rad_train.json"
    cfg.vqarad_image_root      = "/content/vqa_rad/images"
    cfg.vqarad_test_path       = "/content/vqa_rad/vqa_rad_test.json"
    # RAG defaults off see arguments.py get_vqarad_finetune_config note.
    cfg.use_rag                = False
    cfg.output_path            = "/content/drive/MyDrive/qfsru_checkpoints/vqarad_finetune"
    return cfg


def get_slake_finetune_config(checkpoint_path):
    cfg = Config()
    cfg.stage0_end_epoch       = 0
    cfg.stage1_end_epoch       = 0
    cfg.rag_start_epoch        = 1
    cfg.finetune_epoch         = 1
    cfg.unfreeze_encoder_epoch = 1
    cfg.freeze_vision          = True
    cfg.lr                     = 5e-6
    cfg.finetune_lr            = 1e-6
    cfg.max_epochs             = 15
    cfg.patience               = 4
    cfg.batch_size             = 16
    cfg.accumulation_steps     = 2
    cfg.dataset_name           = "slake"
    cfg.training_stage         = "finetune"
    cfg.resume_checkpoint      = checkpoint_path
    cfg.slake_csv_path         = "/content/slake/train.json"
    cfg.slake_image_root       = "/content/slake/imgs"
    cfg.slake_test_path        = "/content/slake/test.json"
    cfg.slake_val_path         = "/content/slake/validate.json"  
    cfg.use_rag                = False
    cfg.output_path            = "/content/drive/MyDrive/qfsru_checkpoints/slake_finetune"
    return cfg

def ft_apply_freezing(model, optimizer, args, epoch):

    vision_backbone_frozen = getattr(args, "freeze_vision", True)
    for p in model.vision_encoder.model.parameters():
        p.requires_grad = not vision_backbone_frozen

    if epoch == 1:
        
        for p in model.bart.model.encoder.parameters(): p.requires_grad = False
        for p in model.bart.model.decoder.parameters(): p.requires_grad = True
        for p in model.bart.lm_head.parameters(): p.requires_grad = True
        for m in ("fsru", "text_to_fsru", "fsru_to_text", "answer_type_head", "rag_bridge"):
            mod = getattr(model, m, None)
            if mod:
                for p in mod.parameters(): p.requires_grad = True
        # FIX: keep the projection bridges trainable during fine-tuning too.
        for p in model.vision_encoder.proj.parameters(): p.requires_grad = True
        for p in model.vision_encoder.early_proj.parameters(): p.requires_grad = True
        print(f"[Epoch {epoch}] → FT Stage A: encoder frozen, decoder+FSRU+RAG trainable")

    elif epoch < args.finetune_epoch:
       
        for p in model.bart.model.encoder.parameters(): p.requires_grad = True
        for p in model.bart.model.decoder.parameters(): p.requires_grad = True
        for p in model.bart.lm_head.parameters(): p.requires_grad = True
        for m in ("fsru", "text_to_fsru", "fsru_to_text", "answer_type_head", "rag_bridge"):
            mod = getattr(model, m, None)
            if mod:
                for p in mod.parameters(): p.requires_grad = True
        
        for p in model.vision_encoder.proj.parameters(): p.requires_grad = True
        for p in model.vision_encoder.early_proj.parameters(): p.requires_grad = True
        print(f"[Epoch {epoch}] → FT Stage B: encoder unfrozen @0.1x LR")

   
    else:
        for name, p in model.named_parameters():
            if "vision_encoder.model.conv1" in name:
                p.requires_grad = False
            
            elif "retrieval_encoder" in name:
                p.requires_grad = False
            elif "vision_encoder" not in name:
                p.requires_grad = True
        for p in model.vision_encoder.proj.parameters(): p.requires_grad = True
        for p in model.vision_encoder.early_proj.parameters(): p.requires_grad = True
        print(f"[Epoch {epoch}] → FT Stage C: full unfreeze — LR={optimizer.param_groups[0]['lr']:.1e}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

def ft_train_one_epoch(model, optimizer, scaler, dataloader,
                       device, epoch, criterion, args):
    model.train()
    running_loss = 0.0
    running_gen_loss = 0.0
    correct_type = 0
    total_type   = 0
    accum_steps  = getattr(args, "accumulation_steps", 1)

    
    fsru_params = (
        list(model.fsru.parameters())
        + list(model.text_to_fsru.parameters())
        + list(model.fsru_to_text.parameters())
    )
    fsru_param_ids = {id(p) for p in fsru_params}
    other_params = [
        p for n, p in model.named_parameters()
        if id(p) not in fsru_param_ids
    ]

    loop = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for step, batch in enumerate(loop):
        is_last_batch     = (step == len(dataloader) - 1)
        is_accum_boundary = ((step + 1) % accum_steps == 0) or is_last_batch

        images          = batch["images"].to(device)
        question_ids    = batch["question_ids"].to(device)
        question_mask   = batch["question_mask"].to(device)
        answer_ids      = batch["answer_ids"].to(device)
        answer_mask     = batch["answer_mask"].to(device)
        answer_type_ids = batch.get("answer_type_ids")
        if answer_type_ids is not None:
            answer_type_ids = answer_type_ids.to(device)

        if step % accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        with autocast('cuda'):
            outputs = model(
                images=images,
                question_ids=question_ids,
                question_mask=question_mask,
                answer_ids=answer_ids,
                question_texts=batch["question_texts"],
            )
            loss, loss_components = criterion(
                outputs,
                answer_ids,
                answer_mask=answer_mask,
                answer_type_ids=answer_type_ids,
                model=model,
                images=images,
                text_feats=outputs.get("text_feats"),
                training_stage=args.training_stage
            )
            loss = loss / accum_steps
            if is_last_batch:
                print(f"[Epoch {epoch} | Step {step}] Loss components: {loss_components}")

        if not torch.isfinite(loss):
            print(f"[Epoch {epoch} | Step {step}] ⚠️ NaN — skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()

        if is_accum_boundary:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(fsru_params, max_norm=2.0)
            torch.nn.utils.clip_grad_norm_(other_params, max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

        if outputs.get("answer_type_logits") is not None and answer_type_ids is not None:
            preds = outputs["answer_type_logits"].argmax(dim=-1)
            correct_type += (preds == answer_type_ids).sum().item()
            total_type   += answer_type_ids.size(0)

        running_loss += loss.item() * accum_steps
        loop.set_postfix(loss=loss.item() * accum_steps)

    avg_loss = running_loss / len(dataloader)
    type_acc = (correct_type / total_type) if total_type > 0 else 0.0
    return avg_loss, type_acc

def ft_evaluate(model, dataloader, device, criterion, args, epoch=0):
    model.eval()
    running_loss = 0.0
    correct_type = 0
    total_type   = 0

    with torch.no_grad():
        loop = tqdm(dataloader, desc="  [Validation]", leave=False)
        for batch in loop:
            # AFTER
            images          = batch["images"].to(device)
            question_ids    = batch["question_ids"].to(device)
            question_mask   = batch["question_mask"].to(device)
            answer_ids      = batch["answer_ids"].to(device)
            answer_mask     = batch["answer_mask"].to(device)
            answer_type_ids = batch.get("answer_type_ids")
            if answer_type_ids is not None:
                answer_type_ids = answer_type_ids.to(device)

            with autocast('cuda'):
                outputs = model(
                    images=images,
                    question_ids=question_ids,
                    question_mask=question_mask,
                    answer_ids=answer_ids,
                    question_texts=batch["question_texts"],
                )
                loss, loss_components = criterion(
                    outputs, answer_ids,
                    answer_mask=answer_mask,
                    answer_type_ids=answer_type_ids,
                    model=None, images=None,
                    text_feats=None,
                    training_stage=args.training_stage
                )

            running_loss += loss.item()
            if outputs.get("answer_type_logits") is not None and answer_type_ids is not None:
                preds = outputs["answer_type_logits"].argmax(dim=-1)
                correct_type += (preds == answer_type_ids).sum().item()
                total_type   += answer_type_ids.size(0)
            loop.set_postfix(val_loss=loss.item())

    avg_loss = running_loss / len(dataloader)
    type_acc = (correct_type / total_type) if total_type > 0 else 0.0
    return avg_loss, type_acc

def load_pretrained(model, checkpoint_path, device):
    print(f"📂 Loading pretrained weights from: {checkpoint_path}")
    ckpt     = torch.load(checkpoint_path, map_location=device)
    state    = ckpt.get("model_state_dict", ckpt)
    model_sd = model.state_dict()

    filtered = {
        k: v for k, v in state.items()
        if k in model_sd and v.shape == model_sd[k].shape
    }
    skipped = len(state) - len(filtered)
    if skipped:
        print(f"  ⚠️  Skipped {skipped} mismatched key(s)")

    model.load_state_dict(filtered, strict=False)
    print(f"✅ Loaded {len(filtered)} keys (epoch {ckpt.get('epoch', '?')})")
    return model

import re

def _normalize_answer(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    # Remove BART special tokens
    for tok in ("</s>", "<s>", "<pad>", "<unk>", "<mask>"):
        text = text.replace(tok, "")
    # Strip trailing punctuation only (internal hyphens in 'ring-enhancing' must survive)
    text = re.sub(r"[.,!?;:]+$", "", text)
    # Article removal with word boundary — the ONLY safe method
    text = re.sub(r"\b(a|an|the)\b\s*", "", text)
    # Collapse multiple spaces left by article removal
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _exact_match(pred: str, gt: str) -> bool:
    """Strict exact match after normalization. No substring tricks."""
    return _normalize_answer(pred) == _normalize_answer(gt)


def _token_f1(pred: str, gt: str) -> float:
    
    pred_tokens = _normalize_answer(pred).split()
    gt_tokens   = _normalize_answer(gt).split()
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gt_tokens)
    return 2.0 * precision * recall / (precision + recall)


def run_benchmark_evaluation(model, test_loader, device, args):
    model.eval()

    em_open = em_closed = 0
    f1_open = f1_closed = 0.0
    total_open = total_closed = 0
    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images        = batch["images"].to(device)
            question_ids  = batch["question_ids"].to(device)
            question_mask = batch["question_mask"].to(device)
            gt_answers    = batch["answer_texts"]
            q_types       = batch.get("question_type", ["open"] * len(gt_answers))


            preds = model.generate(images, question_ids, question_mask,
                                    question_texts=batch["question_texts"])

            for i in range(images.size(0)):
                pred      = preds[i]
                pred_norm = _normalize_answer(pred)
                gt_norm   = _normalize_answer(str(gt_answers[i]))

                em = int(_exact_match(pred, str(gt_answers[i])))
                f1 = _token_f1(pred, str(gt_answers[i]))

                qtype = str(q_types[i]).lower()
                if qtype in ("closed", "yes/no"):
                    em_closed    += em
                    f1_closed    += f1
                    total_closed += 1
                else:
                    em_open    += em
                    f1_open    += f1
                    total_open += 1

                results.append({
                    "pred": pred_norm,
                    "gt":   gt_norm,
                    "em":   bool(em),
                    "f1":   round(f1, 4),
                    "type": qtype,
                })

    total      = total_open + total_closed
    overall_em = (em_open + em_closed) / total      if total        > 0 else 0.0
    open_em    = em_open   / total_open              if total_open   > 0 else 0.0
    closed_em  = em_closed / total_closed            if total_closed > 0 else 0.0
    overall_f1 = (f1_open + f1_closed) / total      if total        > 0 else 0.0
    open_f1    = f1_open   / total_open              if total_open   > 0 else 0.0
    closed_f1  = f1_closed / total_closed            if total_closed > 0 else 0.0

    print("\n" + "="*65)
    print("📊 BENCHMARK EVALUATION RESULTS")
    print("="*65)
    print(f"  Metric          Overall      Open         Closed")
    print(f"  Exact Match     {overall_em*100:6.2f}%    {open_em*100:6.2f}%    {closed_em*100:6.2f}%")
    print(f"  Token F1        {overall_f1*100:6.2f}%    {open_f1*100:6.2f}%    {closed_f1*100:6.2f}%")
    print(f"  Sample count    {total:6d}     {total_open:6d}     {total_closed:6d}")
    print("="*65)

    os.makedirs(args.output_path, exist_ok=True)
    out = os.path.join(args.output_path, "eval_results.json")
    with open(out, "w") as f:
        json.dump({
            "overall_em": overall_em, "open_em": open_em, "closed_em": closed_em,
            "overall_f1": overall_f1, "open_f1": open_f1, "closed_f1": closed_f1,
            "total": total, "total_open": total_open, "total_closed": total_closed,
            "per_sample": results,
        }, f, indent=2)
    print(f"💾 Results saved to: {out}")
    return overall_em, open_em, closed_em

def run_single_seed(cmd, seed):
    # Config
    if cmd.benchmark == "vqa-rad":
        args = get_vqarad_finetune_config(cmd.checkpoint)
    else:
        args = get_slake_finetune_config(cmd.checkpoint)

    args.seed = seed
    args.output_path = args.output_path + f"_seed{seed}"

    ft_set_seed(args.seed)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    os.makedirs(args.output_path, exist_ok=True)
    print(f"🖥️  Device: {device} | 📁 Output: {args.output_path}")
    from transformers import AutoTokenizer as _AutoTokenizer
    args._tokenizer = _AutoTokenizer.from_pretrained(args.text_backbone)
    if args._tokenizer.pad_token is None:
        args._tokenizer.add_special_tokens({"pad_token": "<pad>"})
    args.vocab_size = len(args._tokenizer)
    print(f"[TOKENIZER] Shared tokenizer ready: {args.text_backbone} | vocab_size={args.vocab_size}")

    # Model
    model = QFSRUModel(args).to(device)
    args._image_preprocess = model.vision_encoder.processor
    model = load_pretrained(model, cmd.checkpoint, device)

    if getattr(args, "use_rag", False):
        model.enable_rag()
    else:
        print("ℹ️  args.use_rag is False — running without retrieval augmentation")

    # Data
    dataloaders  = get_dataloaders(args)
    train_loader = dataloaders["train"]
    val_loader   = dataloaders.get("val")
    test_loader  = dataloaders.get("test")
    print(f"📊 Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset) if val_loader else 0} | "
          f"Test: {len(test_loader.dataset) if test_loader else 0}")

    encoder_params = list(model.bart.model.encoder.parameters())

    proj_params = [
        model.vision_encoder.proj.weight,
        model.vision_encoder.proj.bias,
        model.vision_encoder.early_proj.weight,
        model.vision_encoder.early_proj.bias,
        model.fsru.img_filter_2d.filter_bank,
        model.fsru.txt_filter_1d.filter_bank,
        model.fsru.img_pos_embed,
    ]
    proj_params = [p for p in proj_params if p is not None]

    backbone_params = list(model.vision_encoder.model.parameters())
    _claimed_ids = (
        {id(p) for p in encoder_params}
        | {id(p) for p in proj_params}
        | {id(p) for p in backbone_params}
    )
    main_params = [
        p for _, p in model.named_parameters()
        if id(p) not in _claimed_ids
    ]

    optimizer = AdamW(main_params, lr=args.lr, weight_decay=0.01)
    optimizer.add_param_group({"params": encoder_params, "lr": args.lr * 0.1})
    optimizer.add_param_group({"params": proj_params, "lr": args.lr, "weight_decay": 0.0})
    optimizer.add_param_group({"params": backbone_params, "lr": args.lr * 0.1})
    print(f"✅ FT optimizer built with 4 param groups: "
          f"{ [len(g['params']) for g in optimizer.param_groups]}")

    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-7)
    scaler    = GradScaler('cuda')
    loss_weights = get_loss_weights_for_stage("finetune")
    criterion    = QFSRULoss(**loss_weights)
    best_val_loss = float("inf")

    print(f"\n🚀 Fine-tuning on {cmd.benchmark.upper()} "
          f"for {args.max_epochs} epochs (seed={seed})")
    print(f"   LR={args.lr:.1e} | batch={args.batch_size} | "
          f"accum={args.accumulation_steps}\n")

    for epoch in range(1, args.max_epochs + 1):
        ft_apply_freezing(model, optimizer, args, epoch)
        train_loss, train_acc = ft_train_one_epoch(
            model, optimizer, scaler, train_loader,
            device, epoch, criterion, args
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()

        if val_loader is not None:
            val_loss, val_acc = ft_evaluate(
                model, val_loader, device, criterion, args, epoch
            )
            print(f"[Epoch {epoch}] Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | Type Acc: {val_acc:.4f} | "
                  f"LR: {current_lr[0]:.2e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(args.output_path, "best_finetune.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "benchmark": cmd.benchmark,
                    "config": args.to_dict()
                }, best_path)
                print(f"🏆 Best model saved (val_loss={val_loss:.4f})")
        else:
            print(f"[Epoch {epoch}] Train: {train_loss:.4f} | "
                  f"LR: {current_lr[0]:.2e}")

    # Final test evaluation
    if test_loader is not None:
        best_path = os.path.join(args.output_path, "best_finetune.pt")
        if os.path.exists(best_path):
            print("\n📊 Loading best model for final test evaluation...")
            best_ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(best_ckpt["model_state_dict"])
        else:
            print("⚠️  best_finetune.pt not found — evaluating with current weights")
        overall_em, open_em, closed_em = run_benchmark_evaluation(model, test_loader, device, args)
        return {"seed": seed, "overall_em": overall_em, "open_em": open_em, "closed_em": closed_em}
    else:
        print("⚠️  No test loader — skipping final evaluation")
        return {"seed": seed, "overall_em": None, "open_em": None, "closed_em": None}


def run_multiseed(seeds=(42, 123, 2024)):
    
    cmd = argparse.Namespace(
        benchmark="vqa-rad",
        checkpoint="/content/drive/MyDrive/qfsru_checkpoints/pretrain_epoch_20.pt"
    )

    all_results = []
    for seed in seeds:
        print(f"\n{'='*80}\n🌱 Starting run with seed={seed}\n{'='*80}")
        result = run_single_seed(cmd, seed)
        all_results.append(result)

    import statistics as stats
    ems = [r["overall_em"] for r in all_results if r["overall_em"] is not None]
    print(f"\n{'='*80}")
    print(f"📊 MULTI-SEED SUMMARY ({len(seeds)} seeds)")
    for r in all_results:
        print(f"  seed={r['seed']:<6} overall_em={r['overall_em']}")
    if ems:
        print(f"  MEAN EM: {stats.mean(ems):.4f}  ±  STD: {stats.stdev(ems):.4f}")
    print(f"{'='*80}")
    return all_results


if __name__ == "__main__":
    run_multiseed(seeds=(42, 123, 2024))
