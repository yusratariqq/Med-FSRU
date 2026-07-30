import torch
import torch.nn as nn
import torch.nn.functional as F

class _FocalLoss(nn.Module):
    """Focal loss (Lin et al. 2017) down-weights well-classified majority
    examples via (1-p_t)^gamma, on top of the existing class weighting.
    Directly targets the collapse pattern where the model gets 99% recall
    on 'other' by just always predicting it.

    label_smoothing added: once logits get pushed toward extreme confidence
    on the majority class, cross-entropy gradients shrink toward zero (this
    is why TypeHead grad norm was logging as 0.0000 for many epochs even
    though the type loss component was nonzero). Smoothing caps how
    confident/extreme the target distribution can get, keeping gradients
    from vanishing at that saturation point.
    """
    def __init__(self, weight=None, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets, weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none"
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

class QFSRULoss(nn.Module):
    def __init__(
        self,
        beta: float = 0.05,
        lambda_type: float = 0.1,
        lambda_contrast: float = 0.1,
        lambda_diversity: float = 0.1,   
        smoothing: float = 0.1,
        contrastive_margin: float = 0.2,
        type_class_weights: torch.Tensor = None,   
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            label_smoothing=smoothing,
            ignore_index=-100
        )
        self.type_ce = _FocalLoss(weight=type_class_weights, gamma=2.0)

        self.beta            = beta
        self.lambda_type     = lambda_type
        self.lambda_contrast = lambda_contrast
        self.lambda_diversity = lambda_diversity
        self.contrastive_margin = contrastive_margin

    def forward(self, outputs, answer_ids, answer_mask=None, answer_type_ids=None,
            model=None, images=None, text_feats=None, training_stage="stage0"):
        components = {"generation": None, "contrast": None, "type": None, "cos_align": None}

        if training_stage == "stage0":
            align_loss = outputs.get("stage0_align_loss", None)
            if align_loss is None:
                zero = torch.tensor(0.0, device=answer_ids.device)
                return zero, components
            components["cos_align"] = align_loss.item()
            return align_loss, components

        logits = outputs["decoder_logits"]
        loss = self.ce(logits.reshape(-1, logits.size(-1)), answer_ids.reshape(-1))
        components["generation"] = loss.item()
        fsru_feats = outputs.get("fsru_feats", None)
        kagg       = outputs.get("rag_vec",    None)

        if (training_stage in ("stage1", "stage2", "stage3", "finetune")
                and model is not None and images is not None
                and text_feats is not None and fsru_feats is not None):
            N_img_for_loss = outputs["fsru_feats"].size(1) - text_feats.size(1)  
            correct_img_feats = fsru_feats[:, :N_img_for_loss, :]
            correct_txt_feats = fsru_feats[:, N_img_for_loss:, :]
            answer_emb = self._encode_answer(model, answer_ids, answer_mask)
            img_early  = outputs.get("img_early")
            if img_early is None:
                raise RuntimeError(
                    "outputs['img_early'] is missing — model.forward() must "
                    "return img_early in its training-path output dict for "
                    "the contrastive loss to avoid a redundant vision "
                    "encoder forward pass."
                )
            img_early = img_early.detach()  

            img_contrast_loss = self._contrastive_visual_loss(
                model, images, text_feats, answer_emb, correct_img_feats
            )
            txt_contrast_loss = self._contrastive_text_loss(
                model, img_early, text_feats, answer_emb, correct_txt_feats
            )
            contrast_loss = 0.5 * (img_contrast_loss + txt_contrast_loss)
            components["contrast"]      = contrast_loss.item()
            components["contrast_img"]  = img_contrast_loss.item()
            components["contrast_txt"]  = txt_contrast_loss.item()
            loss = loss + self.lambda_contrast * contrast_loss

            if correct_img_feats.size(0) >= 2:   
                img_branch_feats = correct_img_feats.mean(dim=1)             
                img_branch_norm  = F.normalize(img_branch_feats, dim=-1)
                sim_matrix = img_branch_norm @ img_branch_norm.t()          
                off_diag = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
                collapse_penalty = sim_matrix[off_diag].mean().clamp(min=0.0)
                components["collapse_penalty"] = collapse_penalty.item()
                loss = loss + self.lambda_diversity * collapse_penalty

        if (training_stage in ("stage2", "stage3", "finetune")
        and fsru_feats is not None and kagg is not None):
            q = F.normalize(fsru_feats.mean(dim=1), dim=-1)        
            
            k = F.normalize(kagg.mean(dim=1), dim=-1)              
            cos_align_loss = 1.0 - (q * k).sum(dim=-1).mean()
            components["cos_align"] = cos_align_loss.item()
            loss = loss + self.beta * cos_align_loss

        if (training_stage in ("stage1", "stage2", "stage3", "finetune")
                and answer_type_ids is not None
                and outputs.get("answer_type_logits") is not None):
            type_loss = self.type_ce(outputs["answer_type_logits"], answer_type_ids)
            components["type"] = type_loss.item()
            loss = loss + self.lambda_type * type_loss

        return loss, components
    def _encode_answer(self, model, answer_ids: torch.Tensor, answer_mask: torch.Tensor) -> torch.Tensor:
        
        pad_id = model.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        answer_ids_safe = answer_ids.clone()
        answer_ids_safe[answer_ids_safe == -100] = pad_id

        with torch.no_grad():   
            enc_out = model.bart.model.encoder(
                input_ids=answer_ids_safe,
                attention_mask=answer_mask,
                return_dict=True,
            )
            hidden = enc_out.last_hidden_state
        mask       = answer_mask.float().unsqueeze(-1)
        answer_emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        answer_emb = F.normalize(answer_emb, dim=-1)
        return answer_emb
    @staticmethod
    def _make_shuffled_permutation(batch_size: int, device: torch.device, max_attempts: int = 10) -> torch.Tensor:
        """
        Returns a permutation of range(batch_size) with at most 25% fixed
        points, used to build a "wrong" negative batch for contrastive loss.

        """
        perm = torch.randperm(batch_size, device=device)
        identity = torch.arange(batch_size, device=device)
        attempts = 1
        while (perm == identity).sum() > batch_size // 4 and attempts < max_attempts:
            perm = torch.randperm(batch_size, device=device)
            attempts += 1
        return perm
    def _contrastive_visual_loss(
        self, model, images: torch.Tensor, text_feats: torch.Tensor,
        answer_emb: torch.Tensor, correct_fsru_feats: torch.Tensor
    ):
        batch_size = images.size(0)
        if batch_size < 2:
            return torch.zeros((), device=images.device, dtype=correct_fsru_feats.dtype)

        perm = self._make_shuffled_permutation(batch_size, images.device)
        wrong_images = images[perm]
        with torch.no_grad():
            _, wrong_img_early = model.vision_encoder(wrong_images, return_early=True)
            wrong_fsru_output  = model.fsru(wrong_img_early, text_feats)
            wrong_fsru_output = model.fsru_to_text(wrong_fsru_output)
            wrong_fsru_output = F.layer_norm(
                wrong_fsru_output, [wrong_fsru_output.size(-1)]
            )
            N_img = wrong_img_early.size(1)
            wrong_fsru_feats  = wrong_fsru_output[:, :N_img, :]

        # Normalize FSRU features
        correct_feats_norm = F.normalize(correct_fsru_feats.mean(dim=1), dim=-1)
        wrong_feats_norm = F.normalize(wrong_fsru_feats.mean(dim=1), dim=-1)

        # Cosine similarity
        pos_sim = (correct_feats_norm * answer_emb).sum(dim=-1)  # Should be high
        neg_sim = (wrong_feats_norm * answer_emb).sum(dim=-1)    # Should be low

        margin_loss    = F.relu(neg_sim - pos_sim + self.contrastive_margin)
        # Clip pos_attraction to zero once pos_sim ≥ 0.9 (saturates like a hinge)
        pos_attraction = F.relu(0.9 - pos_sim)
        contrast_loss  = (margin_loss + 0.5 * pos_attraction).mean()

        return contrast_loss
    def _contrastive_text_loss(
            self, model, img_early: torch.Tensor, text_feats: torch.Tensor,
            answer_emb: torch.Tensor, correct_fsru_feats: torch.Tensor
        ):
            batch_size = text_feats.size(0)
            if batch_size < 2:
                return torch.zeros((), device=text_feats.device, dtype=correct_fsru_feats.dtype)

            perm = self._make_shuffled_permutation(batch_size, text_feats.device)
            wrong_text_feats = text_feats[perm]

            with torch.no_grad():
                wrong_fsru_output  = model.fsru(img_early, wrong_text_feats)
                wrong_fsru_output  = model.fsru_to_text(wrong_fsru_output)
                wrong_fsru_output  = F.layer_norm(
                    wrong_fsru_output, [wrong_fsru_output.size(-1)]
                )
                N_img = img_early.size(1)
                wrong_fsru_feats  = wrong_fsru_output[:, N_img:, :]

            correct_feats_norm = F.normalize(correct_fsru_feats.mean(dim=1), dim=-1)
            wrong_feats_norm   = F.normalize(wrong_fsru_feats.mean(dim=1), dim=-1)

            pos_sim = (correct_feats_norm * answer_emb).sum(dim=-1)
            neg_sim = (wrong_feats_norm * answer_emb).sum(dim=-1)

            margin_loss    = F.relu(neg_sim - pos_sim + self.contrastive_margin)
            pos_attraction = F.relu(0.9 - pos_sim)
            contrast_loss  = (margin_loss + 0.5 * pos_attraction).mean()

            return contrast_loss


def get_loss_weights_for_stage(training_stage: str) -> dict:
    
    stage_weights = {
        "stage0": dict(
            beta=0.0,
            lambda_type=0.0,
            lambda_contrast=0.15,
            smoothing=0.1
        ),
        "stage1": dict(
            beta=0.0,
            lambda_type=0.05,
            lambda_contrast=0.12,
            smoothing=0.1
        ),
        "stage2": dict(
            beta=0.05,
            lambda_type=0.15,   
            lambda_contrast=0.1,
            smoothing=0.1
        ),
        "stage3": dict(
            beta=0.05,
            lambda_type=0.2, 
            lambda_contrast=0.1,
            smoothing=0.05
        ),
        "finetune": dict(
            beta=0.03,
            lambda_type=0.2,
            lambda_contrast=0.05,
            smoothing=0.05
        ),
    }
    weights = stage_weights.get(training_stage, stage_weights["stage1"])
    return weights
