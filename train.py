

import os
import glob
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, SequentialLR

def set_seed(seed: int):
    import random as _random
    import numpy as _np
    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def _build_scheduler_with_warmup(optimizer, total_epochs: int, warmup_epochs: int = 2):
    
    warmup_epochs = max(0, min(warmup_epochs, max(total_epochs - 1, 0)))
    if warmup_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=max(total_epochs, 1), eta_min=1e-7)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(total_epochs - warmup_epochs, 1), eta_min=1e-7)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
def compare_fusion_param_counts(d_model=768, img_seq=196, txt_seq=128, num_filter=2, num_heads=8):

    fsru = FSRU(d_model=d_model, img_seq=img_seq, txt_seq=txt_seq, num_filter=num_filter)
    cross_attn = CrossAttnFusion(d_model=d_model, num_heads=num_heads)
    fsru_params = sum(p.numel() for p in fsru.parameters())
    cross_attn_params = sum(p.numel() for p in cross_attn.parameters())
    print(f"[Fusion param comparison] FSRU: {fsru_params:,} | "
          f"CrossAttn: {cross_attn_params:,} | ratio: {fsru_params/cross_attn_params:.2f}x")

# DIAGNOSTIC UTILITIES

def log_gradient_norms(model, epoch, step=None):
    
    norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            norms[name] = p.grad.norm().item()

    fsru_norm     = sum(v for k, v in norms.items() if "fsru"    in k.lower())
    decoder_norm  = sum(v for k, v in norms.items() if "decoder" in k.lower())
    encoder_norm  = sum(v for k, v in norms.items() if "bart.model.encoder" in k)
    vision_proj_norm = sum(v for k, v in norms.items()
                            if "vision_encoder.proj" in k or "vision_encoder.early_proj" in k)
    rag_norm       = sum(v for k, v in norms.items() if "rag" in k.lower())
    type_head_norm = sum(v for k, v in norms.items() if "answer_type_head" in k)

    tag = f"[Epoch {epoch}]" if step is None else f"[Epoch {epoch} | Step {step}]"
    parts = [f"FSRU: {fsru_norm:.4f}", f"Decoder: {decoder_norm:.4f}", f"Encoder: {encoder_norm:.4f}"]
    if getattr(model, "use_rag", False):
        parts.append(f"RAG: {rag_norm:.4f}")
    parts.append(f"TypeHead: {type_head_norm:.4f}")
    print(f"{tag} Grad Norms — " + " | ".join(parts))

    if decoder_norm > 0:
        ratio = fsru_norm / decoder_norm
        if ratio > 5.0:
            print(f"  ⚠️  FSRU/Decoder grad ratio = {ratio:.2f} — FFT gradients may be exploding.")
        elif ratio < 0.1:
            print(f"  ⚠️  FSRU/Decoder grad ratio = {ratio:.2f} — FFT branch may be dead.")

    return norms


def log_fsru_gate_saturation(model, epoch):
    """
    Method 1: FSRU gate saturation check.

    Reads gate_activation_mean attribute written by FSRU.forward() during the
    last batch of the epoch. This is the mean ABSOLUTE VALUE of the raw
    (unbounded, non-sigmoided) text2img_gate / img2text_gate Conv1d outputs —
    NOT a bounded [0,1] saturation fraction. There is no principled "healthy
    range" for this quantity in absolute terms; instead, track it across
    epochs and watch for it collapsing toward exactly 0 (gate dead / FFT
    branch muted) or growing without bound (potential instability, cross-
    check against log_gradient_norms for the FSRU parameter group).
    """
    if not hasattr(model, "fsru"):
        return
    fsru = model.fsru
    if not hasattr(fsru, "gate_activation_mean"):
        print(f"[Epoch {epoch}] ⚠️  FSRU has no gate_activation_mean attribute. "
              "Add `self.gate_activation_mean = ...` inside FSRU.forward().")
        return

    g   = fsru.gate_activation_mean
    gs  = getattr(fsru, "gate_activation_std", None)
    if g < 1e-4:
        status = "🔴 DEAD ..."
    else:
        status = "ℹ️  nonzero ..."
    print(f"[Epoch {epoch}] FSRU Gate Mean |Activation|: {g:.4f}  "
          f"Std: {gs:.4f}  {status}" if gs is not None else
          f"[Epoch {epoch}] FSRU Gate Mean |Activation|: {g:.4f}  {status}")
@torch.no_grad()
def diagnose_alignment(img, img_early, enc_hidden):
    img_pooled   = img.float().mean(dim=1)
    early_pooled = img_early.float().mean(dim=1)
    txt_pooled   = enc_hidden.float().mean(dim=1)

    def cos_sim(a, b):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return (a * b).sum(dim=-1)

    orig_late  = cos_sim(img_pooled, txt_pooled)
    orig_early = cos_sim(early_pooled, txt_pooled)

    img_c   = img_pooled   - img_pooled.mean(dim=0, keepdim=True)
    early_c = early_pooled - early_pooled.mean(dim=0, keepdim=True)
    txt_c   = txt_pooled   - txt_pooled.mean(dim=0, keepdim=True)

    cent_late  = cos_sim(img_c, txt_c)
    cent_early = cos_sim(early_c, txt_c)

    print(f"ORIG   late: mean={orig_late.mean():.4f}  std={orig_late.std():.4f}")
    print(f"ORIG  early: mean={orig_early.mean():.4f}  std={orig_early.std():.4f}")
    print(f"CENT   late: mean={cent_late.mean():.4f}  std={cent_late.std():.4f}")
    print(f"CENT  early: mean={cent_early.mean():.4f}  std={cent_early.std():.4f}")

    for name, raw in [("img", img_pooled), ("early", early_pooled), ("txt", txt_pooled)]:
        dc_norm  = raw.mean(dim=0).norm()
        res_norm = (raw - raw.mean(dim=0, keepdim=True)).norm(dim=-1).mean()
        print(f"{name}: ||batch_mean||={dc_norm:.4f}  avg||residual||={res_norm:.4f}  ratio={dc_norm/res_norm:.4f}")
    # ADD at the end of diagnose_alignment(), after the existing prints
    with torch.no_grad():
        def top1_acc(a, b):
            a = F.normalize(a, dim=-1); b = F.normalize(b, dim=-1)
            sims = a @ b.t()
            preds = sims.argmax(dim=-1)
            labels = torch.arange(sims.size(0), device=sims.device)
            return (preds == labels).float().mean().item()
        print(f"[InfoNCE CHECK] late  in-batch top-1 acc: {top1_acc(img_pooled, txt_pooled):.3f}")
        print(f"[InfoNCE CHECK] early in-batch top-1 acc: {top1_acc(early_pooled, txt_pooled):.3f}")
        print(f"  (chance level = 1/{img_pooled.size(0)} = {1.0/img_pooled.size(0):.3f}; "
              f"well above chance ⇒ no collapse)")
# ============================================================
# CKA REPRESENTATION SIMILARITY 
# ============================================================

def _center_kernel_matrix(K: torch.Tensor) -> torch.Tensor:
    """
    Double-centers a kernel matrix K of shape (n, n).
    Formula: Kc = H @ K @ H  where H = I - (1/n) * ones
    """
    row_mean   = K.mean(dim=1, keepdim=True)
    col_mean   = K.mean(dim=0, keepdim=True)
    total_mean = K.mean()
    return K - row_mean - col_mean + total_mean


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Computes linear CKA between two representation matrices.

    Args:
        X : (n_samples, d1)  FSRU output features
        Y : (n_samples, d2)  spatial-only (pre-FFT projection) features

    Returns:
        CKA score in [0, 1].
          1.0  -> representations identical up to rotation (FFT adds nothing)
          0.0  -> representations completely orthogonal (FFT maximally different)
        Threshold from review: CKA > 0.95 means FFT branch is redundant.

    Uses linear kernel (K = X @ X.T) so no hyperparameter is needed.
    Runs on CPU float32 — FFT in float16 is numerically unsafe.
    Memory cost: O(n^2). For n=500 val samples ~1 MB, negligible.
    """
    X = X.detach().cpu().float()
    Y = Y.detach().cpu().float()

    # L2-normalise rows for stability (CKA is scale-invariant so this is safe)
    X = X / (X.norm(dim=1, keepdim=True).clamp(min=1e-8))
    Y = Y / (Y.norm(dim=1, keepdim=True).clamp(min=1e-8))

    Kx  = X @ X.T                          # (n, n) Gram matrix
    Ky  = Y @ Y.T
    Kxc = _center_kernel_matrix(Kx)
    Kyc = _center_kernel_matrix(Ky)

    # HSIC estimator (Frobenius inner product of centred Gram matrices)
    hsic_xy = (Kxc * Kyc).sum()
    hsic_xx = (Kxc * Kxc).sum()
    hsic_yy = (Kyc * Kyc).sum()

    denom = (hsic_xx * hsic_yy).sqrt().clamp(min=1e-10)
    return float(hsic_xy / denom)


class FSRURepresentationHook:
    """
    Captures FSRU's image-branch input and output to compute CKA.

    The correct comparison for measuring FFT contribution is:
      - spatial_buf: img_early AFTER early_proj, BEFORE FSRU
                     This is the raw patch-embed projection — what the
                     image looks like entering the FFT branch.
      - fsru_buf:    freq_tokens image slice AFTER fsru_to_text + layer_norm
                     This is what the image looks like leaving the FFT branch.

    Both are image-branch-only, mean-pooled over 196 patch tokens.
    Text tokens are explicitly excluded from both buffers so CKA measures
    only the FFT transformation of visual features, not the modality gap.

    CKA > 0.95 → 2D FFT leaves image representations essentially unchanged
    CKA 0.50–0.80 → FFT meaningfully transforms image representations
    CKA < 0.50 → FFT produces very different representations (investigate)
    """

    def __init__(self, model: torch.nn.Module):
        self.model         = model
        self._fsru_buf     : list = []
        self._spatial_buf  : list = []
        self._handles      : list = []
        self._n_img_tokens : int  = 196

    def register(self, n_img_tokens: int = 196):
        
        self._fsru_buf     = []
        self._spatial_buf  = []
        self._n_img_tokens = n_img_tokens

        def _capture_fsru_output(module, inp, out):
            feat = out[0] if isinstance(out, tuple) else out
            if feat.dim() != 3:
                print(f"[FSRURepresentationHook] Unexpected fsru_to_text "
                      f"output dim: {feat.dim()} — expected 3. Skipping batch.")
                return
            # Slice image branch only, text tokens are excluded
            img_feat = feat[:, :self._n_img_tokens, :]   
            pooled   = img_feat.mean(dim=1)              
            self._fsru_buf.append(pooled.detach().cpu().float())
        def _capture_early_proj_output(module, inp, out):
            feat = out[0] if isinstance(out, tuple) else out
            if feat.dim() != 3:
                print(f"[FSRURepresentationHook] Unexpected early_proj "
                      f"output dim: {feat.dim()} — expected 3. Skipping batch.")
                return
            
            pooled = feat.mean(dim=1)                     
            self._spatial_buf.append(pooled.detach().cpu().float())

        if hasattr(self.model, "fsru_to_text"):
            self._handles.append(
                self.model.fsru_to_text.register_forward_hook(_capture_fsru_output)
            )
        else:
            print("[FSRURepresentationHook] WARNING: model.fsru_to_text not found. "
                  "CKA will not be computed. Check model attribute name.")

        vision_enc = getattr(self.model, "vision_encoder", None)
        if vision_enc is not None and hasattr(vision_enc, "early_proj"):
            self._handles.append(
                vision_enc.early_proj.register_forward_hook(
                    _capture_early_proj_output
                )
            )
        else:
            print("[FSRURepresentationHook] WARNING: vision_encoder.early_proj "
                  "not found. Spatial baseline will be empty. "
                  "CKA cannot be computed without both hooks firing.")

        registered = len(self._handles)
        if registered < 2:
            print(f"[FSRURepresentationHook] Only {registered}/2 hooks registered. "
                  f"CKA results will be invalid.")
        else:
            print(f"[FSRURepresentationHook] Both hooks registered — "
                  f"comparing FSRU image-branch input vs output over "
                  f"{n_img_tokens} patch tokens.")

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def compute_cka(self, epoch: int, max_samples: int = 2000) -> float:
        """
        Computes CKA(FSRU_image_output, early_proj_image_input).

        Interpretation:
          High CKA (> 0.95): FSRU's 2D FFT filtering leaves image
            representations nearly unchanged. FFT branch is redundant —
            consider whether FSRU is actually training (check gate norms).
          Medium CKA (0.50–0.80): FFT produces meaningfully different
            image representations. This is the target range — FFT is
            transforming but not destroying spatial information.
          Low CKA (< 0.50): FFT produces very different representations.
            May indicate FFT instability or exploding gate activations.
            Cross-check with log_fsru_gate_saturation().
        """
        if not self._fsru_buf:
            print(f"[Epoch {epoch}] CKA: ⚠️  FSRU output buffer empty — "
                  f"fsru_to_text hook did not fire. Check model.fsru_to_text exists.")
            return -1.0

        if not self._spatial_buf:
            print(f"[Epoch {epoch}] CKA: ⚠️  Spatial input buffer empty — "
                  f"early_proj hook did not fire. Check vision_encoder.early_proj exists.")
            return -1.0

        fsru_feats    = torch.cat(self._fsru_buf,    dim=0)  # [N, d_model]
        spatial_feats = torch.cat(self._spatial_buf, dim=0)  # [N, d_model]

        n_fsru    = fsru_feats.shape[0]
        n_spatial = spatial_feats.shape[0]

        if abs(n_fsru - n_spatial) > 5:
            
            print(f"[Epoch {epoch}] CKA: ⚠️  Buffer size mismatch — "
                  f"FSRU buffer: {n_fsru}, spatial buffer: {n_spatial}. "
                  f"Truncating to minimum.")

        n   = min(n_fsru, n_spatial, max_samples)
        cka = linear_cka(fsru_feats[:n], spatial_feats[:n])

        if cka > 0.95:
            verdict = "🔴 REDUNDANT  — 2D FFT leaves image representations unchanged"
        elif cka > 0.80:
            verdict = "🟡 MARGINAL   — 2D FFT adds limited transformation"
        elif cka > 0.50:
            verdict = "🟢 USEFUL     — 2D FFT meaningfully transforms image features"
        else:
            verdict = "🟡 INVESTIGATE — large transformation, check gate saturation"

        print(
            f"[Epoch {epoch}] CKA(FSRU_img_out ∥ early_proj_img_in) = {cka:.4f}  "
            f"{verdict}  [n={n}]"
        )
        return cka

    def compute_output_diversity(self, epoch: int, max_samples: int = 500) -> float:
            """
            Cross-sample similarity of FSRU image-branch OUTPUTS — not
            input-vs-output like compute_cka(). This disambiguates the CKA
            plateau: a low input-output CKA is consistent with BOTH (a) FSRU
            learning a rich, per-image-dependent transform (healthy), and
            (b) FSRU collapsing toward a near-constant output that is simply
            unlike the input for every image (pathological — the low CKA
            would then be a false positive for "useful transformation").

            Computes average pairwise cosine similarity BETWEEN DIFFERENT
            samples' post-FSRU outputs (same buffer compute_cka already fills):
              > 0.90  -> 🔴 outputs barely vary across different images —
                        likely collapsed to a near-constant vector
              0.60-0.90 -> 🟡 low diversity, worth watching
              < 0.60  -> 🟢 outputs are genuinely image-specific
            """
            if not self._fsru_buf:
                print(f"[Epoch {epoch}] Output-diversity: ⚠️ FSRU output buffer empty.")
                return -1.0

            feats = torch.cat(self._fsru_buf, dim=0)
            n = min(feats.size(0), max_samples)
            feats = feats[:n]
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)

            sim_matrix = feats @ feats.t()                       # [n, n]
            off_diag_mask = ~torch.eye(n, dtype=torch.bool)
            avg_cross_sim = sim_matrix[off_diag_mask].mean().item()

            if avg_cross_sim > 0.90:
                verdict = "🔴 COLLAPSED — outputs nearly identical across different images"
            elif avg_cross_sim > 0.60:
                verdict = "🟡 LOW DIVERSITY — outputs cluster tightly, investigate"
            else:
                verdict = "🟢 DIVERSE — outputs vary meaningfully across images"

            print(f"[Epoch {epoch}] FSRU output cross-sample similarity = "
                  f"{avg_cross_sim:.4f}  {verdict}  [n={n}]")
            return avg_cross_sim
# ============================================================
# FREEZING STRATEGY
# ============================================================

_encoder_group_added = False


def apply_freezing_strategy(model, optimizer, args, epoch):
    global _encoder_group_added

    vision_backbone_frozen = getattr(args, "freeze_vision", True)
    for p in model.vision_encoder.model.parameters():
        p.requires_grad = not vision_backbone_frozen

    if epoch <= args.stage0_end_epoch:
        for p in model.parameters():
            p.requires_grad = False

        for p in model.vision_encoder.proj.parameters():
            p.requires_grad = True
        for p in model.vision_encoder.early_proj.parameters():
            p.requires_grad = True
        if hasattr(model, "stage0_logit_scale"):
            model.stage0_logit_scale.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Epoch {epoch}] 🔵 STAGE 0: Training proj + early_proj only "
              f"({trainable:,} params) — align loss only, no CE")

    # ── STAGE 1: FSRU + decoder ──────────────────────────────────────────────
    elif epoch <= args.stage1_end_epoch:
        # Keep BART encoder frozen
        for p in model.bart.model.encoder.parameters():
            p.requires_grad = False
        # Unfreeze decoder
        for p in model.bart.model.decoder.parameters():
            p.requires_grad = True
        for p in model.bart.lm_head.parameters():
            p.requires_grad = True
        # Unfreeze FSRU and projections
        for module_name in ("fsru", "text_to_fsru", "fsru_to_text", "answer_type_head"):
            module = getattr(model, module_name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = True
       
        for p in model.vision_encoder.proj.parameters():
            p.requires_grad = True
        for p in model.vision_encoder.early_proj.parameters():
            p.requires_grad = True
        
        for p in model.source_type_embed.parameters():
            p.requires_grad = True
        for p in model.visual_out_norm.parameters():
            p.requires_grad = True
        for p in model.text_out_norm.parameters():
            p.requires_grad = True

        print(f"[Epoch {epoch}] 🟡 STAGE 1: FSRU + decoder training (encoder frozen)")


    # ── STAGE 2:  encoder unfreeze ─────────────────────────────
    elif epoch < args.finetune_epoch:
        
        for p in model.bart.model.encoder.parameters():
            p.requires_grad = True
        for p in model.bart.model.decoder.parameters():
            p.requires_grad = True
        for p in model.bart.lm_head.parameters():
            p.requires_grad = True
        for module_name in ("fsru", "text_to_fsru", "fsru_to_text", "answer_type_head", "rag_bridge"):
            module = getattr(model, module_name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = True
       
        for p in model.vision_encoder.proj.parameters():
            p.requires_grad = True
        for p in model.vision_encoder.early_proj.parameters():
            p.requires_grad = True
        rag_note = "RAG + " if getattr(args, "use_rag", False) else ""
        print(f"[Epoch {epoch}] 🔓 Stage 2: encoder + {rag_note}decoder + FSRU all trainable")

    else:
       
        for name, p in model.named_parameters():
            if "vision_encoder.model.conv1" in name:
                p.requires_grad = False
            
            elif "retrieval_encoder" in name:
                p.requires_grad = False
            elif "vision_encoder" not in name:
                p.requires_grad = True
        
        for p in model.vision_encoder.proj.parameters():
            p.requires_grad = True
        for p in model.vision_encoder.early_proj.parameters():
            p.requires_grad = True

        # LR is managed by the main training loop at the stage boundary.
        # apply_freezing_strategy no longer sets LR avoiding double-set.
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"[Epoch {epoch}] 🟢 STAGE 3 — LR={optimizer.param_groups[0]['lr']:.1e} | "
              f"Trainable: {trainable:,}/{total:,} ({100*trainable/total:.1f}%)")

    if getattr(model, "use_rag", False) and model.cosine_rag is not None:
        for p in model.cosine_rag.parameters():
            p.requires_grad = True

def evaluate(model, dataloader, device, criterion, args, epoch: int = 0):
    """
    Runs validation with no_grad + model.eval().
    Returns (avg_val_loss, avg_type_acc, cka_score).

    NEW-D: CKA is computed here via forward hooks ,zero extra forward passes.
    cka_score == -1.0 means hooks failed (non-fatal, logged as warning).
    """
    model.eval()
    running_loss = 0.0
    running_gen_loss = 0.0
    correct_type = 0
    total_type   = 0

    cka_hook = FSRURepresentationHook(model)
    cka_hook.register(n_img_tokens=model.vision_encoder.num_tokens)

    if model.use_rag and model.cosine_rag is not None and model.cosine_rag.kb_texts is not None:
        sample_batch = next(iter(dataloader))
        with torch.no_grad():
            q_texts = sample_batch["question_texts"][:5]
            _q = model._build_rag_query(q_texts)
            
            _, topk_idx = model.cosine_rag(direct_query=_q)
            retrieved_texts = model.cosine_rag.lookup_texts(topk_idx)
            hits = 0
            for i, q_text in enumerate(sample_batch["question_texts"][:5]):
                retrieved = retrieved_texts[i] if retrieved_texts else []
                print(f"[RAG CHECK] Q: {q_text}")
                for r in retrieved:
                    print(f"  → {r[:100]}")
                
                q_words = set(q_text.lower().split()) - {"is","the","a","in","this","of","are","there","what"}
                if any(any(w in r.lower() for w in q_words) for r in retrieved):
                    hits += 1
            print(f"[RAG CHECK] Topical hit rate: {hits}/5 queries")
    gate_by_qtype = {}   

    TYPE_NAMES = [k.replace("/", "_") for k, _ in sorted(ANSWER_TYPE_MAP.items(), key=lambda kv: kv[1])]
    num_type_classes = len(TYPE_NAMES)
    num_type_classes = len(TYPE_NAMES)
    class_correct    = torch.zeros(num_type_classes)
    class_total      = torch.zeros(num_type_classes)
    class_pred_total = torch.zeros(num_type_classes)

    with torch.no_grad():
        loop = tqdm(dataloader, desc="  [Validation]", leave=False)
        for step, batch in enumerate(loop):
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
                if hasattr(model.fsru, "_last_img_gate"):  
                    gate_per_sample = model.fsru._last_img_gate.abs().mean(dim=(1, 2, 3)).cpu().tolist()   
                    for qtype, gval in zip(batch["question_type"], gate_per_sample):   
                        gate_by_qtype.setdefault(qtype, []).append(gval)  
                if args.training_stage == "stage0" and (step == 0 or step == len(dataloader) - 1):
                    diagnose_alignment(
                        outputs["_debug_img"],
                        outputs["_debug_img_early"],
                        outputs["_debug_enc_hidden"],
                    )
                
                loss, loss_components = criterion(
                    outputs,
                    answer_ids,
                    answer_mask=answer_mask,
                    answer_type_ids=answer_type_ids,
                    model=None,
                    images=None,
                    text_feats=None,
                    training_stage=args.training_stage
                )

            running_loss += loss.item()
            running_gen_loss += (loss_components.get("generation") or 0.0)

            if outputs.get("answer_type_logits") is not None and answer_type_ids is not None:
                preds = outputs["answer_type_logits"].argmax(dim=-1)
                correct_type += (preds == answer_type_ids).sum().item()
                total_type   += answer_type_ids.size(0)

                for c in range(num_type_classes):
                    true_mask = (answer_type_ids == c)
                    pred_mask = (preds == c)
                    class_total[c]      += true_mask.sum().item()
                    class_pred_total[c] += pred_mask.sum().item()
                    class_correct[c]    += (true_mask & pred_mask).sum().item()

            loop.set_postfix(val_loss=loss.item())
    if gate_by_qtype:   
      print(f"[Epoch {epoch}] Gate magnitude by question type:")  
      for qtype, vals in sorted(gate_by_qtype.items()):  
          print(f"  {qtype:<10}: mean={sum(vals)/len(vals):.4f}  n={len(vals)}")

    recall    = class_correct / class_total.clamp(min=1)
    precision = class_correct / class_pred_total.clamp(min=1)
    f1        = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    macro_f1  = f1.mean().item()
    print(f"[Epoch {epoch}] Type head — macro-F1: {macro_f1:.4f}")
    for c, name in enumerate(TYPE_NAMES):
        print(f"  {name:<8}: recall={recall[c]:.4f}  precision={precision[c]:.4f}  "
              f"f1={f1[c]:.4f}  n_true={int(class_total[c])}  n_pred={int(class_pred_total[c])}")

    cka_score      = cka_hook.compute_cka(epoch)
    diversity_score = cka_hook.compute_output_diversity(epoch)  
    cka_hook.remove()
    avg_loss = running_loss / len(dataloader)
    type_acc = (correct_type / total_type) if total_type > 0 else 0.0
    avg_gen_loss = running_gen_loss / len(dataloader)   # isolates generation signal
    return avg_loss, type_acc, cka_score, macro_f1, diversity_score, avg_gen_loss

# ============================================================
# ANSWER-QUALITY EVAL (exact-match / token-F1)
# ============================================================

def _normalize_answer_text(text: str) -> str:
    """Lowercase, strip punctuation-adjacent whitespace, collapse spaces.
    Kept deliberately simple/consistent so EM and F1 use the same
    normalization on both prediction and reference."""
    return " ".join(text.lower().strip().split())


def _token_f1(pred: str, ref: str) -> float:
    """Standard word-overlap F1 (SQuAD-style) between two short strings."""
    pred_tokens = _normalize_answer_text(pred).split()
    ref_tokens  = _normalize_answer_text(ref).split()
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return float(pred_tokens == ref_tokens)

    common = {}
    for t in pred_tokens:
        common[t] = min(pred_tokens.count(t), ref_tokens.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


@torch.no_grad()
def evaluate_answer_quality(model, dataloader, device, args, epoch: int,
                             max_samples: int = 500, max_gen_length: int = 64,
                             fixed_indices: list = None) -> dict:
    
    model.eval()
    seen = 0
    em_total, f1_total = 0.0, 0.0
    by_type = {}   # qtype -> {"em": ..., "f1": ..., "n": ...}

    if fixed_indices is not None:
        from torch.utils.data import Subset, DataLoader as _DataLoader
        eval_ds = Subset(dataloader.dataset, fixed_indices)
        eval_loader = _DataLoader(
            eval_ds, batch_size=dataloader.batch_size,
            shuffle=False, collate_fn=collate_fn,
            num_workers=0,
        )
        max_samples = len(fixed_indices)   # scan the whole fixed set, no early cutoff
    else:
        eval_loader = dataloader

    loop = tqdm(eval_loader, desc="  [Answer-quality eval]", leave=False)
    for batch in loop:
        if seen >= max_samples:
            break

        images        = batch["images"].to(device)
        question_ids  = batch["question_ids"].to(device)
        question_mask = batch["question_mask"].to(device)
        answer_ids    = batch["answer_ids"].to(device)

        generated = model.generate(
            images=images,
            question_ids=question_ids,
            question_mask=question_mask,
            max_length=max_gen_length,
            question_texts=batch["question_texts"],
        )

        pad_id = model.tokenizer.pad_token_id or 0
        ref_ids = answer_ids.clone()
        ref_ids[ref_ids == -100] = pad_id
        references = model.tokenizer.batch_decode(ref_ids, skip_special_tokens=True)

        qtypes = batch.get("question_type", ["unknown"] * len(generated))

        for pred, ref, qtype in zip(generated, references, qtypes):
            if seen >= max_samples:
                break
            em = float(_normalize_answer_text(pred) == _normalize_answer_text(ref))
            f1 = _token_f1(pred, ref)
            em_total += em
            f1_total += f1
            seen += 1

            bucket = by_type.setdefault(qtype, {"em": 0.0, "f1": 0.0, "n": 0})
            bucket["em"] += em
            bucket["f1"] += f1
            bucket["n"]  += 1

    avg_em = em_total / max(seen, 1)
    avg_f1 = f1_total / max(seen, 1)

    print(f"[Epoch {epoch}] Answer quality (n={seen}) — Exact Match: {avg_em:.4f} | Token-F1: {avg_f1:.4f}")
    for qtype, b in sorted(by_type.items()):
        print(f"  {qtype:<10}: EM={b['em']/b['n']:.4f}  F1={b['f1']/b['n']:.4f}  n={b['n']}")

    return {"em": avg_em, "f1": avg_f1, "n": seen, "by_type": by_type}
def build_stratified_eval_subset(dataloader, per_type_target=100, yesno_target=None):
    """
    Builds a fixed list of dataset indices, stratified by answer_type, so
    evaluate_answer_quality always scores the SAME samples each epoch rather
    than whatever the first N in dataloader order happen to be.
    """
    ds = dataloader.dataset
    by_type = {}
    for i in range(len(ds)):
        real_idx = ds.valid_indices[i]
        item = ds.data[real_idx]
        _, atype = ds._resolve_answer_and_type(item)
        by_type.setdefault(atype, []).append(i)

    subset_indices = []
    for atype, indices in by_type.items():
        if atype == "yes/no":
            target = yesno_target if yesno_target is not None else len(indices)
        else:
            target = per_type_target
        subset_indices.extend(indices[:target])

    selected_counts = {}
    for atype, indices in by_type.items():
        target = yesno_target if (atype == "yes/no" and yesno_target is not None) else \
                 (len(indices) if atype == "yes/no" else per_type_target)
        selected_counts[atype] = min(target, len(indices))
    print(f"[Stratified eval subset] built with {len(subset_indices)} total samples "
          f"across types: {selected_counts}  (pool sizes were: "
          f"{ {t: len(v) for t, v in by_type.items()} })")
    return subset_indices
def train_one_epoch(model, optimizer, scaler, dataloader, device, epoch, criterion, args):
    model.train()
    running_loss  = 0.0
    correct_type  = 0
    total_type    = 0
    accum_steps   = getattr(args, "accumulation_steps", 1)
    log_grad_norm_at_end = False
    use_amp = (args.training_stage != "stage0")

    fsru_params = (
        list(model.fsru.parameters())
        + list(model.text_to_fsru.parameters())
        + list(model.fsru_to_text.parameters())
    )
    fsru_param_ids = {id(p) for p in fsru_params}
    blend_params = [model.early_blend_alpha, model.fsru_blend_alpha]
    blend_param_ids = {id(p) for p in blend_params}

    other_params = [
        p for n, p in model.named_parameters()
        if id(p) not in fsru_param_ids and id(p) not in blend_param_ids
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

        with autocast('cuda', enabled=use_amp):
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
            print(f"[Epoch {epoch} | Step {step}] ⚠️ NaN/Inf loss detected — skipping batch")
            optimizer.zero_grad(set_to_none=True)
            continue

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if is_accum_boundary:
            if use_amp:
                scaler.unscale_(optimizer)
                if is_last_batch:
                    log_gradient_norms(model, epoch)
                    log_grad_norm_at_end = True
                
                torch.nn.utils.clip_grad_norm_(fsru_params, max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(other_params, max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(blend_params, max_norm=5.0)  
                scaler.step(optimizer)
                scaler.update()
            else:
                # Stage 0: plain optimizer step, no scaler involved
                if is_last_batch:
                    log_gradient_norms(model, epoch)
                # REPLACE (both occurrences)
                torch.nn.utils.clip_grad_norm_(fsru_params, max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(other_params, max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(blend_params, max_norm=5.0)  
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if outputs.get("answer_type_logits") is not None and answer_type_ids is not None:
            preds = outputs["answer_type_logits"].argmax(dim=-1)
            correct_type += (preds == answer_type_ids).sum().item()
            total_type   += answer_type_ids.size(0)

        running_loss += loss.item() * accum_steps
        loop.set_postfix(loss=loss.item() * accum_steps)

    avg_loss = running_loss / len(dataloader)
    type_acc = (correct_type / total_type) if total_type > 0 else 0.0

    log_fsru_gate_saturation(model, epoch)
    with torch.no_grad():
        w_norm = model.fsru_to_text.weight.norm().item()
        print(f"[Epoch {epoch}] fsru_to_text weight norm: {w_norm:.6f}")
    with torch.no_grad():
        a_early = model._bounded_alpha(model.early_blend_alpha).item()
        a_txt   = model._bounded_alpha(model.fsru_blend_alpha).item()
    print(f"[Epoch {epoch}] Blend alphas — visual: {a_early:.4f} | text: {a_txt:.4f}  "
          f"(bounded range [0.05, 0.95] — watch for persistent drift toward the floor)")

    return avg_loss, type_acc


# ============================================================
# CHECKPOINT UTILITIES
# ============================================================

def find_latest_checkpoint(output_path):
    """Find the most recent epoch checkpoint in output directory."""
    checkpoint_files = glob.glob(os.path.join(output_path, "pretrain_epoch_*.pt"))
    if not checkpoint_files:
        return None, None

    def _parse_epoch(path):
      try:
        return int(path.split("_")[-1].split(".")[0])
      except ValueError:
        return -1

    checkpoint_files = [f for f in checkpoint_files if _parse_epoch(f) > 0]
    if not checkpoint_files:
      return None, None
    checkpoint_files.sort(key=_parse_epoch)
    latest_ckpt  = checkpoint_files[-1]
    latest_epoch = _parse_epoch(latest_ckpt)
    return latest_ckpt, latest_epoch



def is_phase_boundary(epoch, args):
    '''
    Used for logging phase transitions. No longer affects optimizer state.
    '''
    boundaries = {
        args.unfreeze_encoder_epoch - 1,
        args.rag_start_epoch - 1,
    }
    return epoch in boundaries
def _rebuild_kb_with_trained_encoder(model, args, device, batch_size=32):
    """
    Re-encodes the entire KB using the model's current BART encoder weights.
    Called once, if RAG is enabled in future experiments.
    """
    import pickle
    print("🔄 Rebuilding KB with trained encoder (this takes ~30s)...")
    model.eval()

    # Load original texts
    if not hasattr(args, 'rag_texts_path') or not args.rag_texts_path:
        print("⚠️  rag_texts_path not set — cannot rebuild KB. Skipping.")
        return
    with open(args.rag_texts_path, 'rb') as f:
        all_texts = pickle.load(f)

    embeddings = []
    tokenizer  = model.tokenizer

    with torch.no_grad():
        for i in range(0, len(all_texts), batch_size):
            batch  = all_texts[i:i + batch_size]
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(device)

            enc    = model.bart.model.encoder(
                input_ids      = tokens['input_ids'],
                attention_mask = tokens['attention_mask'],
                return_dict    = True
            )
            hidden = enc.last_hidden_state                      # [B, T, d_model]
            mask   = tokens['attention_mask'].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            pooled = F.normalize(pooled, dim=-1)
            embeddings.append(pooled.cpu())

    new_kb = torch.cat(embeddings, dim=0).to(device)           # [N, d_model]

    # Validate dimension before replacing
    if new_kb.shape[1] != model.cosine_rag.kb_embeddings.shape[1]:
        raise RuntimeError(
            f"Rebuilt KB dim {new_kb.shape[1]} != "
            f"existing KB dim {model.cosine_rag.kb_embeddings.shape[1]}. "
            f"Check d_model consistency."
        )

    model.cosine_rag.kb_embeddings.copy_(new_kb)
    print(f"✅ KB rebuilt: {new_kb.shape[0]} entries, dim={new_kb.shape[1]}")
    model.train()
@torch.no_grad()
def verify_patch_ordering(vision_encoder, device):
    
    from PIL import Image
    import numpy as np

    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[:112, :112, :] = 255             # bright top-left quadrant only
    pil_img = Image.fromarray(arr)

    # Hook the patch-embedding layer directly (pre-transformer-block).
    patch_embed = None
    for candidate in ("trunk.patch_embed", "patch_embed"):
        obj = vision_encoder.model
        try:
            for part in candidate.split("."):
                obj = getattr(obj, part)
            patch_embed = obj
            break
        except AttributeError:
            continue
    if patch_embed is None:
        raise RuntimeError(
            "[Patch-order check] Could not resolve patch_embed module — "
            "add the correct path in verify_patch_ordering()."
        )

    def _get_patch_tokens(pil_image):
        captured = {}
        def _capture(module, inp, out):
            captured["out"] = out.detach()
        handle = patch_embed.register_forward_hook(_capture)
        img_t = vision_encoder.processor(pil_image).unsqueeze(0).to(device)
        _, _ = vision_encoder(img_t, return_early=True)
        handle.remove()
        tokens = captured["out"]
        if tokens.dim() == 4:           
            tokens = tokens.flatten(2).transpose(1, 2)  
        return tokens.squeeze(0)        

    
    baseline_arr = np.full((224, 224, 3), 128, dtype=np.uint8)
    baseline_img = Image.fromarray(baseline_arr)

    target_row, target_col = 2, 9   
    perturbed_arr = baseline_arr.copy()
    r0, c0 = target_row * 16, target_col * 16
    perturbed_arr[r0:r0 + 16, c0:c0 + 16, :] = 255
    perturbed_img = Image.fromarray(perturbed_arr)

    baseline_tokens  = _get_patch_tokens(baseline_img)
    perturbed_tokens = _get_patch_tokens(perturbed_img)

    diff = (perturbed_tokens - baseline_tokens).norm(dim=-1)   # [196]
    predicted_idx = diff.argmax().item()
    expected_idx  = target_row * 14 + target_col

    print(f"[Patch-order check] perturbed patch (row={target_row}, col={target_col}) "
          f"expected flat index={expected_idx} | predicted (argmax diff)={predicted_idx}")

    if predicted_idx != expected_idx:
        pred_row, pred_col = predicted_idx // 14, predicted_idx % 14
        raise RuntimeError(
            f"[Patch-order check] FAILED — perturbing grid cell "
            f"(row={target_row}, col={target_col}) [flat index {expected_idx}] produced "
            f"peak response at flat index {predicted_idx} (row={pred_row}, col={pred_col}). "
            f"Patch ordering does not match the [14,14] row-major assumption FSRU depends on."
        )
    print("[Patch-order check] PASSED.")


def main():
    global _encoder_group_added

    args   = parse_args()
    set_seed(args.seed)
    device = args.device

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    
    from transformers import AutoTokenizer as _AutoTokenizer
    args._tokenizer = _AutoTokenizer.from_pretrained(args.text_backbone)
    if args._tokenizer.pad_token is None:
        args._tokenizer.add_special_tokens({"pad_token": "<pad>"})
        print(f"[TOKENIZER] Added <pad> token — vocab size now {len(args._tokenizer)}")
    args.vocab_size = len(args._tokenizer)
    print(f"[TOKENIZER] Shared tokenizer ready: {args.text_backbone} | vocab_size={args.vocab_size}")

    compare_fusion_param_counts(d_model=args.d_model, num_filter=args.fsru_num_filter, num_heads=args.num_heads)
    model = QFSRUModel(args).to(device)
    args._image_preprocess = model.vision_encoder.processor
    print(f"[INIT CHECK] early_proj weight std: {model.vision_encoder.early_proj.weight.std().item():.6f}")
    print(f"[INIT CHECK] proj (late) weight std: {model.vision_encoder.proj.weight.std().item():.6f}")

    verify_patch_ordering(model.vision_encoder, device)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dataloaders = get_dataloaders(args)
    train_loader = dataloaders["train"]
    val_loader   = dataloaders.get("val", None)
    def _is_no_decay_param(name: str) -> bool:
        
        no_decay_names = {"early_blend_alpha", "fsru_blend_alpha", "stage0_logit_scale"}
        short_name = name.split(".")[-1]
        return (
            name.endswith(".bias")
            or ".norm" in name.lower()
            or "layernorm" in name.lower()
            or short_name in no_decay_names
        )

    initial_named = [
        (name, p) for name, p in model.named_parameters()
        if "vision_encoder" not in name
        and "bart.model.encoder" not in name
        and "rag_bridge" not in name
    ]
    initial_decay    = [p for name, p in initial_named if not _is_no_decay_param(name)]
    initial_no_decay = [p for name, p in initial_named if _is_no_decay_param(name)]

    optimizer = AdamW(initial_decay, lr=args.lr, weight_decay=0.05)
    existing_ids = {id(p) for name, p in initial_named}

    proj_params = [
        model.vision_encoder.proj.weight,
        model.vision_encoder.proj.bias,
        model.vision_encoder.early_proj.weight,
        model.vision_encoder.early_proj.bias,
        model.fsru.img_filter_2d.filter_bank,
        model.fsru.txt_filter_1d.filter_bank,
        model.fsru.img_pos_embed,
    ]
    proj_params = [p for p in proj_params if p is not None and id(p) not in existing_ids]
    existing_ids.update(id(p) for p in proj_params)

    backbone_named = [
        (name, p) for name, p in model.vision_encoder.model.named_parameters()
        if id(p) not in existing_ids
    ]
    backbone_decay    = [p for name, p in backbone_named if not _is_no_decay_param(name)]
    backbone_no_decay = [p for name, p in backbone_named if _is_no_decay_param(name)]
    existing_ids.update(id(p) for name, p in backbone_named)

    encoder_named = [
        (name, p) for name, p in model.bart.model.encoder.named_parameters()
        if id(p) not in existing_ids
    ]
    encoder_decay    = [p for name, p in encoder_named if not _is_no_decay_param(name)]
    encoder_no_decay = [p for name, p in encoder_named if _is_no_decay_param(name)]
    existing_ids.update(id(p) for name, p in encoder_named)

    rag_named = (
        [(name, p) for name, p in model.rag_bridge.named_parameters() if id(p) not in existing_ids]
        if model.rag_bridge is not None else []
    )
    rag_decay    = [p for name, p in rag_named if not _is_no_decay_param(name)]
    rag_no_decay = [p for name, p in rag_named if _is_no_decay_param(name)]
    existing_ids.update(id(p) for name, p in rag_named)

    zero_decay_params = proj_params + initial_no_decay + encoder_no_decay + rag_no_decay + backbone_no_decay
    optimizer.add_param_group({
        "params": zero_decay_params,
        "lr": args.lr,
        "weight_decay": 0.0,
    })

    
    optimizer.add_param_group({
        "params": encoder_decay,
        "lr": args.lr * 0.1,
    })


    optimizer.add_param_group({
        "params": rag_decay,
        "lr": args.lr,
    })


    optimizer.add_param_group({
        "params": backbone_decay,
        "lr": args.lr * 0.1,
    })

    _encoder_group_added = True
    print(f"✅ Optimizer built with 5 param groups: {[len(g['params']) for g in optimizer.param_groups]}")

    active_epochs = args.max_epochs - args.stage0_end_epoch
    scheduler = _build_scheduler_with_warmup(
        optimizer, total_epochs=active_epochs, warmup_epochs=getattr(args, "warmup_epochs", 2)
    )

    scaler = GradScaler('cuda', init_scale=256)

    num_classes = 3
    print("[TYPE WEIGHTS] Scanning full train split for exact class counts...")
    from collections import Counter
    type_counter = Counter()
    for i in tqdm(range(len(train_loader.dataset)), desc="[Counting answer types]"):
        real_idx = train_loader.dataset.valid_indices[i]
        item = train_loader.dataset.data[real_idx]
        _, atype = train_loader.dataset._resolve_answer_and_type(item)
        type_counter[atype] += 1

    counts = torch.tensor(
        [type_counter.get(k, 0) for k in ("yes/no", "open", "other")],
        dtype=torch.float
    )
    raw_weights = counts.sum() / counts
    max_weight_ratio = 4.0
    raw_weights = raw_weights.clamp(max=raw_weights.min() * max_weight_ratio)
    type_class_weights = raw_weights / raw_weights.sum() * num_classes
    type_class_weights = type_class_weights.to(device)

    print(f"[TYPE WEIGHTS] counts(exact)={counts.tolist()} → weights={type_class_weights.tolist()}")

    loss_weights = get_loss_weights_for_stage(args.training_stage)
    criterion    = QFSRULoss(**loss_weights, type_class_weights=type_class_weights)
    os.makedirs(args.output_path, exist_ok=True)

    start_epoch   = 1
    best_val_loss = float("inf")

    latest_ckpt, latest_epoch = find_latest_checkpoint(args.output_path)

    if latest_ckpt is not None:
        print(f"🔁 Found checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)

        checkpoint_sd = checkpoint["model_state_dict"]
        model_sd      = model.state_dict()
        filtered_sd ={
            k: v for k, v in checkpoint_sd.items()
            if k in model_sd and v.shape == model_sd[k].shape
        }
        skipped = [k for k in checkpoint_sd if k not in filtered_sd]
        if skipped:
            print(f"⚠️  Skipped {len(skipped)} mismatched key(s) on resume: {skipped}")
        model.load_state_dict(filtered_sd, strict=False)
        print(f"✅ Model weights restored ({len(filtered_sd)} keys loaded)")
        if getattr(args, "reinit_type_head_on_resume", False):
            nn.init.xavier_uniform_(model.answer_type_head.weight, gain=0.1)
            nn.init.zeros_(model.answer_type_head.bias)
            print("🔄 answer_type_head re-initialized after resume (loss-weighting fix)")

            type_head_params = {id(model.answer_type_head.weight), id(model.answer_type_head.bias)}
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if id(p) in type_head_params and p in optimizer.state:
                        optimizer.state[p] = {}
            print("🔄 Cleared Adam state for answer_type_head params")

        if is_phase_boundary(latest_epoch, args):
            print("ℹ️  Resuming at a phase boundary — optimizer state still restored "
                  "(matches live-training behavior; only the LR group values change)")
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("✅ Optimizer state restored")
        except Exception as e:
            print(f"⚠️  Optimizer restore failed ({e}), starting fresh")

        if "scheduler_state_dict" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                print("✅ Scheduler state restored")
            except (KeyError, TypeError) as e:
                print(f"⚠️  Scheduler restore failed ({e}) — likely a scheduler-type "
                      f"mismatch between this checkpoint and current code. "
                      f"Continuing with a freshly initialized scheduler.")

        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            print("✅ GradScaler state restored")

        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        _encoder_group_added = True
        print("✅ Encoder + RAG param groups pre-built — optimizer state will restore cleanly")

        start_epoch = latest_epoch + 1
        print(f"➡️  Starting from epoch {start_epoch}\n")
        if args.stage0_end_epoch < start_epoch <= args.stage1_end_epoch:
            print("🔄 Mid-stage-1 resume detected — restoring full LR for decoder")
            encoder_ratio = getattr(args, "encoder_lr_ratio", 0.1)
            for i, group in enumerate(optimizer.param_groups):
                group["lr"] = args.lr * encoder_ratio if i == 2 else args.lr
                group.pop("initial_lr", None)          # NEW
            remaining = args.max_epochs - start_epoch + 1
            scheduler = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=1e-7)
            print(f"   LR reset — Main/RAG: {args.lr:.1e} | Encoder: {args.lr*encoder_ratio:.1e} | Scheduler T_max={remaining} epochs")

    else:
        print("🆕 No checkpoint found. Starting fresh training.\n")

    if latest_ckpt is not None and start_epoch > args.stage1_end_epoch and start_epoch < args.finetune_epoch:
        print("🔄 Stage 2 resume: repairing discriminative LRs")
        optimizer.param_groups[0]["lr"] = args.lr
        optimizer.param_groups[1]["lr"] = args.lr
        optimizer.param_groups[2]["lr"] = args.lr * 0.1
        optimizer.param_groups[3]["lr"] = args.lr
        remaining = args.finetune_epoch - start_epoch + 6
        scheduler = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=1e-7)
        print(f"   Main/RAG LR: {args.lr:.1e} | Encoder LR: {args.lr*0.1:.1e}")
    if latest_ckpt is not None and start_epoch >= args.finetune_epoch:
        print("🔄 Stage 3 resume: repairing LR after initial_lr fix")
        finetune_lr   = getattr(args, "finetune_lr", 5e-6)
        encoder_ratio = getattr(args, "encoder_lr_ratio", 0.1)
        for i, group in enumerate(optimizer.param_groups):
            group["lr"] = finetune_lr * encoder_ratio if i == 2 else finetune_lr
            group.pop("initial_lr", None)
        remaining = args.max_epochs - start_epoch + 1
        scheduler = CosineAnnealingLR(optimizer, T_max=max(remaining, 1), eta_min=1e-7)
        print(f"   Main/Proj/RAG LR: {finetune_lr:.1e} | Encoder LR: {finetune_lr*encoder_ratio:.1e} | T_max={remaining}")

    
    args.training_stage = (
        "stage0" if start_epoch <= args.stage0_end_epoch else
        "stage1" if start_epoch <= args.stage1_end_epoch else
        "stage2" if start_epoch < args.finetune_epoch    else
        "stage3"
    )
    print(f"ℹ️  training_stage initialized to '{args.training_stage}' for start_epoch={start_epoch}")

    print("=" * 80)
    print("TRAINING STRATEGY: 4-Stage Q-FSRU")
    print("=" * 80)
    print(f"Stage 0  Epochs 1–{args.stage0_end_epoch}:     🔵 Projection warmup (proj layers only)")
    print(f"Stage 1  Epochs {args.stage0_end_epoch+1}–{args.stage1_end_epoch}:     🟡 FSRU + decoder (encoder frozen)")
    print(f"Stage 2  Epochs {args.stage1_end_epoch+1}–{args.finetune_epoch-1}:    🟠 encoder unfrozen (0.1x LR)")
    print(f"Stage 3  Epochs {args.finetune_epoch}–{args.max_epochs}:   🟢 Full fine-tune at LR={getattr(args,'finetune_lr',5e-6):.1e}")
    print(f"Vision Encoder :   🔒 ALWAYS FROZEN")
    print(f"Effective batch size:        {args.batch_size * getattr(args, 'accumulation_steps', 1)}")
    print("=" * 80 + "\n")

    patience_counter = 0
    cka_history  = []
    gate_history = []
    eval_subset_indices = None
    if val_loader is not None:
        eval_subset_indices = build_stratified_eval_subset(
            val_loader, per_type_target=200, yesno_target=None
        )
    for epoch in range(start_epoch, args.max_epochs + 1):

        if (getattr(args, "use_rag", False)
                and epoch >= args.rag_start_epoch
                and not model.use_rag):
            model.enable_rag()

            print(f"🧠 RAG ENABLED at epoch {epoch} — frozen retrieval encoder, "
                  f"query/KB space fixed")
        if (getattr(args, "use_prefix", False)
            and getattr(args, "prefix_start_epoch", None) is not None
            and epoch >= args.prefix_start_epoch
            and not model.use_prefix):

          model.enable_prefix()
          print(f"🧩 Prefix fusion ENABLED at epoch {epoch}")
        current_stage = (
            "stage0" if epoch <= args.stage0_end_epoch else
            "stage1" if epoch <= args.stage1_end_epoch else
            "stage2" if epoch < args.finetune_epoch    else
            "stage3"
        )

        if current_stage != args.training_stage:
            args.training_stage = current_stage
            loss_weights = get_loss_weights_for_stage(current_stage)
            criterion    = QFSRULoss(**loss_weights, type_class_weights=type_class_weights)

            best_val_loss    = float("inf")
            patience_counter = 0
            print(f"[Epoch {epoch}] 🔄 Loss weights updated for {current_stage}: {loss_weights}")
            print(f"[Epoch {epoch}] 🔄 best_val_loss and patience_counter reset for new stage")

        apply_freezing_strategy(model, optimizer, args, epoch)

        if epoch == args.stage0_end_epoch + 1:
            for group in optimizer.param_groups:
                group["lr"] = args.lr
                group.pop("initial_lr", None)         
            remaining = args.max_epochs - epoch
            scheduler = _build_scheduler_with_warmup(
                optimizer, total_epochs=remaining, warmup_epochs=getattr(args, "warmup_epochs", 2)
            )
            print(f"[Epoch {epoch}] 🔄 Scheduler reset for Stage 1 — LR={args.lr:.1e} (+ warmup)")

       
        elif epoch == args.stage1_end_epoch + 1:
          encoder_ratio = getattr(args, "encoder_lr_ratio", 0.1)
          optimizer.param_groups[0]["lr"] = args.lr
          optimizer.param_groups[1]["lr"] = args.lr
          optimizer.param_groups[2]["lr"] = args.lr * encoder_ratio
          optimizer.param_groups[3]["lr"] = args.lr
          remaining = args.max_epochs - epoch
          scheduler = CosineAnnealingLR(optimizer, T_max=max(remaining, 1), eta_min=1e-7)
          print(f"[Epoch {epoch}] 🔄 Stage 2 scheduler reset — Main: {args.lr:.1e} | Encoder: {args.lr*encoder_ratio:.1e} (+ warmup)")

        elif epoch == args.finetune_epoch:
            finetune_lr   = getattr(args, "finetune_lr", 5e-6)
            encoder_ratio = getattr(args, "encoder_lr_ratio", 0.1)
            for i, group in enumerate(optimizer.param_groups):
                if i == 2:
                    group["lr"] = finetune_lr * encoder_ratio
                else:
                    group["lr"] = finetune_lr
                group.pop("initial_lr", None)          
                                                        
            remaining = args.max_epochs - epoch + 1
            scheduler = _build_scheduler_with_warmup(
                optimizer, total_epochs=remaining, warmup_epochs=getattr(args, "warmup_epochs", 2)
            )
            print(f"[Epoch {epoch}] 🔄 Stage 3 scheduler reset — "
                f"Main/Proj/RAG LR={finetune_lr:.1e} | "
                f"Encoder LR={finetune_lr*encoder_ratio:.1e} (+ warmup)")

        train_loss, train_type_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            dataloader=train_loader,
            device=device,
            epoch=epoch,
            criterion=criterion,
            args=args,
        )
        gate_history.append(model.fsru.gate_activation_mean)

        scheduler.step()
        current_lr = scheduler.get_last_lr()
        print(f"[Epoch {epoch}] LR after scheduler step: {current_lr}")

        val_loss        = float("inf")
        val_type_acc    = 0.0
        cka_score       = -1.0
        val_macro_f1    = 0.0
        diversity_score = -1.0
        val_gen_loss    = float("inf")
        answer_quality  = None
        # REPLACE
        if val_loader is not None:
            val_loss, val_type_acc, cka_score, val_macro_f1, diversity_score, val_gen_loss = evaluate(
                model, val_loader, device, criterion, args, epoch=epoch
            )
            print(
                f"[Epoch {epoch}] "
                f"Train Loss: {train_loss:.4f} | Train Type Acc: {train_type_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Type Acc: {val_type_acc:.4f} | "
                f"Val Macro-F1: {val_macro_f1:.4f} | CKA: {cka_score:.4f} | "
                f"Output Diversity: {diversity_score:.4f}"
            )
            answer_eval_every = getattr(args, "answer_eval_every", 3)
            if epoch % answer_eval_every == 0 or epoch == args.max_epochs:
                answer_quality = evaluate_answer_quality(
                    model, val_loader, device, args, epoch=epoch,
                    max_samples=getattr(args, "answer_eval_max_samples", 500),
                    fixed_indices=eval_subset_indices,
                )

            if val_gen_loss < best_val_loss:
                best_val_loss = val_gen_loss
                best_path = os.path.join(args.output_path, "best_model.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_loss": val_loss,
                        "config": args.to_dict(),
                    },
                    best_path,
                )
                print(f"🏆 New best model saved (val_loss={val_loss:.4f}): {best_path}")
                patience_counter = 0
            else:
              patience_counter += 1
              if patience_counter >= args.patience:
                print(f"[Epoch {epoch}] ⏹ Early stopping: no improvement for {args.patience} epochs.")
                break
        else:
            print(
                f"[Epoch {epoch}] "
                f"Train Loss: {train_loss:.4f} | Train Type Acc: {train_type_acc:.4f}"
            )

        ckpt_path = os.path.join(args.output_path, f"pretrain_epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "cka_score": cka_score if val_loader is not None else -1.0,
                "answer_em": answer_quality["em"] if answer_quality is not None else None,
                "answer_f1": answer_quality["f1"] if answer_quality is not None else None,
                "encoder_group_added": _encoder_group_added,
                "config": args.to_dict(),
            },
            ckpt_path,
        )
        print(f"💾 Saved checkpoint: {ckpt_path}\n")


if __name__ == "__main__":
    main()
