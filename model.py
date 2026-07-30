# model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from transformers import AutoTokenizer
from transformers import BartForConditionalGeneration, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from transformers import AutoModel, AutoImageProcessor

def _resolve_early_hook_target(visual_model: torch.nn.Module, layer_idx: int = 3) -> torch.nn.Module:
    """
    Resolves an intermediate ViT block (default: block `layer_idx`) as the
    early-frequency hook target. patch_embed output alone has no attention
    context, so an early-but-not-first block carries more useful texture
    information while staying distinct from the late-semantic branch.
    """
    candidates = [
        (f"trunk.blocks.{layer_idx}",       lambda m: m.trunk.blocks[layer_idx]),
        (f"blocks.{layer_idx}",             lambda m: m.blocks[layer_idx]),
        (f"transformer.resblocks.{layer_idx}", lambda m: m.transformer.resblocks[layer_idx]),
    ]

    for path_name, accessor in candidates:
        try:
            target = accessor(visual_model)
            if not isinstance(target, torch.nn.Module):
                continue
            print(f"[VisionEncoder] Early hook resolved → '{path_name}'  "
                  f"({type(target).__name__})  [intermediate-block hook, Issue #1 fix]")
            return target
        except (AttributeError, IndexError, TypeError):
            continue

    tree_lines = []
    for name, mod in visual_model.named_modules():
        depth  = name.count(".")
        indent = "  " * depth
        tree_lines.append(f"{indent}{name or '(root)'}  →  {type(mod).__name__}")
    tree_str = "\n".join(tree_lines)

    raise RuntimeError(
        f"[VisionEncoder] _resolve_early_hook_target: None of the candidate "
        f"paths matched BiomedCLIP's visual module.\n"
        f"Add the correct path to the candidates list in _resolve_early_hook_target().\n"
        f"Full module tree:\n{tree_str}"
    )

class VisionEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        import open_clip

        backbone_id = getattr(
            args, "vision_backbone",
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        full_model, self.processor = open_clip.create_model_from_pretrained(backbone_id)
        full_model.visual.output_tokens = True
        self.model = full_model.visual

        if hasattr(self.model, 'trunk'):
            self.model.trunk.output_tokens = True

        self.num_tokens = 196
        self._output_tokens_ok  = False
        self._late_hook_target  = None
        self._late_buf          = None

        with torch.no_grad():
            dummy    = torch.randn(1, 3, 224, 224)
            test_out = self.model(dummy)
            if isinstance(test_out, tuple) and len(test_out) >= 2:
                print(f"✅ output_tokens working: pooled={test_out[0].shape}, "
                      f"tokens={test_out[1].shape}")
                self._output_tokens_ok = True
            else:
                print(f"❌ output_tokens NOT working — installing late-token hook "
                      f"on final ViT block.")
                self._late_hook_target = self._resolve_late_hook_target()

        #Late-layer (semantic) projection 
        vision_dim = 768
        self.proj  = nn.Linear(vision_dim, args.d_model)
        early_vision_dim    = getattr(args, "early_vision_dim", 768)
        self.early_proj     = nn.Linear(early_vision_dim, args.d_model)
        self._early_buf     = None   
        self._early_hook_handle = None

        early_layer_idx = getattr(args, "early_hook_layer", 3)
        if not isinstance(early_layer_idx, int):
            early_layer_idx = 3
        self._early_hook_target = _resolve_early_hook_target(self.model, layer_idx=early_layer_idx)


        if args.freeze_vision:
            for p in self.model.parameters():
                p.requires_grad = False
        

    @property
    def device(self):
        return next(self.proj.parameters()).device   

    def _register_early_hook(self):
        '''One-shot design: re-registering every call means there is zero risk of a
           stale buffer if forward() runs without return_early, while also avoiding
           duplicate hook accumulation.
        '''
        def _hook(module, inp, out):
            feat = out[0] if isinstance(out, tuple) else out
            if feat.dim() == 3:
                if feat.size(1) == self.num_tokens + 1:
                    feat = feat[:, 1:, :]   
                elif feat.size(1) != self.num_tokens:
                    raise RuntimeError(
                        f"[VisionEncoder] Early hook captured {feat.size(1)} tokens "
                        f"but num_tokens={self.num_tokens}. Cannot establish 1:1 "
                        f"spatial alignment for gated-additive fusion. "
                        f"Verify _resolve_early_hook_target() is hooking the right layer."
                    )
            elif feat.dim() == 2:
                
                B = inp[0].size(0) if isinstance(inp, tuple) else inp.size(0)
                feat = feat.view(B, -1, feat.size(-1))
                if feat.size(1) != self.num_tokens:
                    raise RuntimeError(
                        f"[VisionEncoder] 2D early hook reshaped to "
                        f"[{feat.size(0)}, {feat.size(1)}, {feat.size(2)}] "
                        f"but expected num_tokens={self.num_tokens}."
                    )
            self._early_buf = feat.detach().clone()   # detach: backbone is frozen

        handle = self._early_hook_target.register_forward_hook(_hook)
        return handle
    def _resolve_late_hook_target(self) -> torch.nn.Module:
        """
        Resolves the FINAL ViT block for late-token extraction.
        Only called when output_tokens=True is not respected by open_clip.
        """
        candidates = [
            ("trunk.blocks[-1]",          lambda m: m.trunk.blocks[-1]),
            ("blocks[-1]",                lambda m: m.blocks[-1]),
            ("transformer.resblocks[-1]", lambda m: m.transformer.resblocks[-1]),
        ]
        for path_name, accessor in candidates:
            try:
                target = accessor(self.model)
                if isinstance(target, torch.nn.Module):
                    print(f"[VisionEncoder] Late-token hook resolved → '{path_name}' "
                          f"({type(target).__name__})")
                    return target
            except (AttributeError, IndexError, TypeError):
                continue
        raise RuntimeError(
            "[VisionEncoder] Cannot resolve final ViT block for late-token hook. "
            "Print self.model and add the correct path to _resolve_late_hook_target()."
        )

    def _register_late_hook(self):
        def _hook(module, inp, out):
            feat = out[0] if isinstance(out, tuple) else out
            if feat.dim() == 3:
                if feat.size(1) == self.num_tokens + 1:
                    feat = feat[:, 1:, :]   
            self._late_buf = feat.detach().clone()
        return self._late_hook_target.register_forward_hook(_hook)

    def forward(self, images, return_early: bool = False):
        images = images.to(self.device)

        self._early_buf = None
        early_handle    = self._register_early_hook()
        late_handle = None
        if not self._output_tokens_ok and self._late_hook_target is not None:
            self._late_buf = None
            late_handle    = self._register_late_hook()

        out = self.model(images)  

        early_handle.remove()
        if late_handle is not None:
            late_handle.remove()

        if self._early_buf is None:
            raise RuntimeError(
                "[VisionEncoder] Early hook did not fire. Check "
                "_resolve_early_hook_target()."
            )

        # ── Get late (semantic) tokens ────────────────────────────────────
        if self._output_tokens_ok:
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                tokens = out[1]
            else:
                raise ValueError(
                    f"output_tokens reported OK but got unexpected output: {type(out)}"
                )
        else:
            # Use final ViT block output captured by late hook
            if self._late_buf is None:
                raise RuntimeError("[VisionEncoder] Late hook did not fire.")
            tokens = self._late_buf.to(self.device)

        assert tokens.dim() == 3, (
            f"Expected 3D token tensor [B, N, D], got shape {tokens.shape}"
        )
        if tokens.size(1) == self.num_tokens + 1:
            tokens = tokens[:, 1:, :]   

        feats = self.proj(tokens)   

        if return_early:
            early_raw   = self._early_buf.to(self.device)
            early_feats = self.early_proj(early_raw)        
            assert feats.shape == early_feats.shape, (
                f"[VisionEncoder] Late {feats.shape} != early {early_feats.shape}"
            )
            return feats, early_feats

        return feats

class QFSRUModel(nn.Module):
    """
    NOTE: name is legacy — originally built for quantum-inspired retrieval.
    That approach was dropped; no quantum computation is used in this codebase.
    Not renamed to avoid breaking checkpoints/imports pre-submission.
    """
    def __init__(self, args):
      super().__init__()
      self.args = args
      self.vision_encoder = VisionEncoder(args)
      if hasattr(args, '_tokenizer') and args._tokenizer is not None:
          self.tokenizer = args._tokenizer
      else:
          self.tokenizer = AutoTokenizer.from_pretrained(args.text_backbone)
          if self.tokenizer.pad_token is None:
              self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

      self.bart = BartForConditionalGeneration.from_pretrained(args.text_backbone)
      if len(self.tokenizer) != self.bart.config.vocab_size:
          self.bart.resize_token_embeddings(len(self.tokenizer))

      self.text_to_fsru = nn.Linear(self.bart.config.d_model, self.bart.config.d_model)
      self.fsru_to_text = nn.Linear(self.bart.config.d_model, self.bart.config.d_model)
      self.fsru_blend_alpha  = nn.Parameter(torch.tensor(0.0))
      self.early_blend_alpha = nn.Parameter(torch.tensor(0.0))
      self.stage0_logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

      self.source_type_embed = nn.Embedding(3, self.bart.config.d_model)
      self.visual_out_norm = nn.LayerNorm(self.bart.config.d_model)
      self.text_out_norm   = nn.LayerNorm(self.bart.config.d_model)

      self.retrieval_encoder_name = getattr(
          args, "retrieval_encoder_name",
          "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
      )
      self.retrieval_tokenizer = None
      self.retrieval_encoder   = None
      rag_ready = getattr(args, "use_rag", False) and getattr(args, "rag_embeddings_path", None)

      if getattr(args, "use_rag", False):
          self._init_retrieval_encoder()

      self.rag_bridge = None
      if rag_ready:
          self.rag_bridge = nn.Linear(self.bart.config.d_model, self.bart.config.d_model)

      # prefix_projector only feeds the optional prefix-memory path.
      # NOTE: intended to gate on args.use_prefix_fusion, but this checks
      # args.use_prefix (attribute doesn't exist on Config → always False).
      # Currently harmless since use_prefix_fusion=False anyway, but this is
      # NOT a working master switch so do not relying on this line to enable
      # prefix fusion via config; use model.enable_prefix() directly instead.
      self.prefix_length = getattr(args, "prefix_length", 16)
      self.prefix_projector = None
      if getattr(args, "use_prefix", False):
          self.prefix_projector = nn.Linear(
              self.bart.config.d_model, self.bart.config.d_model * self.prefix_length
          )

      self.fsru = FSRU(
          d_model    = self.bart.config.d_model,
          img_seq    = self.vision_encoder.num_tokens,   
          txt_seq    = 128,   
          num_filter = getattr(args, "fsru_num_filter", 2),   # configurable for ablation for future eeriemnts
          dropout    = args.dropout,
      )
      # Optional cross-attention fusion module. Not used in the current model.
      self.fusion_type = getattr(args, 'fusion_type', 'fsru')
      if self.fusion_type == 'cross_attn':
          self.cross_attn_fusion = CrossAttnFusion(
              self.bart.config.d_model, args.num_heads, args.dropout
          )

      if rag_ready:
          self.cosine_rag = CosineRAG(
              embedding_path = args.rag_embeddings_path,
              device         = next(self.parameters()).device if len(list(self.parameters())) > 0
                               else torch.device('cpu'),
              d_model        = args.d_model,
              top_k          = args.top_k,
              temperature    = getattr(args, 'rag_temperature', 0.5),
              min_similarity = getattr(args, 'rag_min_similarity', 0.35),
              texts_path     = getattr(args, 'rag_texts_path', None),
          )
      else:
          self.cosine_rag = None

      self.answer_type_head = nn.Linear(
          self.bart.config.d_model,
          3
      )
      self.use_prefix = False
      self.use_rag = False

      self._init_weights()

    @staticmethod
    def _bounded_alpha(raw: torch.Tensor, floor: float = 0.05, ceil: float = 0.95) -> torch.Tensor:
        return floor + (ceil - floor) * torch.sigmoid(raw)
    def _init_retrieval_encoder(self):
        """
        Builds the frozen SapBERT retrieval tokenizer/encoder used by
        _build_rag_query(). Factored out of __init__ so enable_rag() can
        call it too, for the case where args.use_rag was False at
        construction time but RAG is explicitly turned on later.
        """
        from transformers import AutoModel as _AutoModel, AutoTokenizer as _AutoTokenizer
        self.retrieval_tokenizer = _AutoTokenizer.from_pretrained(self.retrieval_encoder_name)
        self.retrieval_encoder   = _AutoModel.from_pretrained(self.retrieval_encoder_name)
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')
        self.retrieval_encoder = self.retrieval_encoder.to(device)
        for p in self.retrieval_encoder.parameters():
            p.requires_grad = False
        self.retrieval_encoder.eval()
    def _build_rag_bridge(self) -> nn.Linear:
        """
        Helper for optional RAG support. Creates the bridge layer if
        RAG is enabled in future experiments.
        """
        device = next(self.parameters()).device
        bridge = nn.Linear(self.bart.config.d_model, self.bart.config.d_model).to(device)
        nn.init.xavier_uniform_(bridge.weight, gain=0.1)
        nn.init.zeros_(bridge.bias)
        return bridge

    def enable_prefix(self):
        """
        Enables the optional prefix-memory module. If prefix projection was
        not created during initialization, it is built lazily so the feature
        can be enabled later without reinitializing the model.
        """
        if self.prefix_projector is None:
            device = next(self.parameters()).device
            self.prefix_projector = nn.Linear(
                self.bart.config.d_model, self.bart.config.d_model * self.prefix_length
            ).to(device)
            nn.init.xavier_uniform_(self.prefix_projector.weight, gain=0.1)
            nn.init.zeros_(self.prefix_projector.bias)
            print("🧩 prefix_projector lazily constructed in enable_prefix()")
        self.use_prefix = True
    def _init_weights(self):
        """Small weight init for projection layers — prevents early NaN via FFT"""
        init_modules = [self.text_to_fsru]
        if self.rag_bridge is not None:
            init_modules.append(self.rag_bridge)
        for module in init_modules:
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # fsru_to_text gets a tighter init (gain=0.01) because it directly
        # multiplies FSRU output which can have large rfft2 power values.
        # Smaller init gives more headroom before attention softmax overflow.
        nn.init.xavier_uniform_(self.fsru_to_text.weight, gain=0.01)
        if self.fsru_to_text.bias is not None:
            nn.init.zeros_(self.fsru_to_text.bias)
        # Late projection
        nn.init.xavier_uniform_(self.vision_encoder.proj.weight, gain=0.1)
        if self.vision_encoder.proj.bias is not None:
            nn.init.zeros_(self.vision_encoder.proj.bias)
        nn.init.xavier_uniform_(self.vision_encoder.early_proj.weight, gain=0.1)
        if self.vision_encoder.early_proj.bias is not None:
            nn.init.zeros_(self.vision_encoder.early_proj.bias)
        # FSRU output projection
        nn.init.xavier_uniform_(self.fsru.out_proj.weight, gain=0.1)
        if self.fsru.out_proj.bias is not None:
            nn.init.zeros_(self.fsru.out_proj.bias)
        nn.init.normal_(self.source_type_embed.weight, mean=0.0, std=0.02)

    def forward(self, images, question_ids, question_mask, answer_ids=None, question_texts=None):

        is_stage0 = (
            hasattr(self.args, 'training_stage') and
            self.args.training_stage == 'stage0'
        )
        # ── Vision: get BOTH late-semantic and early-frequency features ───
        img, img_early = self.vision_encoder(images, return_early=True)

        enc = self.bart.model.encoder(
            input_ids=question_ids,
            attention_mask=question_mask,
            return_dict=True
        )
        enc_hidden = enc.last_hidden_state   # [B, T, d_model]
        enc_out    = enc

        txt = self.text_to_fsru(enc_hidden)  # [B, T, d_model]
        if getattr(self.args, 'debug_magnitudes', False):
            print(f"[MAG CHECK] img_early: mean={img_early.mean().item():.4f}, "
                  f"std={img_early.std().item():.4f}, absmax={img_early.abs().max().item():.4f}")
            print(f"[MAG CHECK] txt: mean={txt.mean().item():.4f}, "
                  f"std={txt.std().item():.4f}, absmax={txt.abs().max().item():.4f}")

        # ── Stage 0 alignment loss ────────────────────────────────────────
        # Align BOTH late (img) and early (img_early) projections against txt.
        # Without the early term, early_proj gets zero gradient in Stage 0
        # and starts Stage 1 with random weights while proj is already warm.
        stage0_align_loss = None
        if is_stage0:
            img_raw       = img.float().mean(dim=1)
            img_early_raw = img_early.float().mean(dim=1)
            enc_raw       = enc_hidden.float().mean(dim=1)

            img_centered       = img_raw       - img_raw.mean(dim=0, keepdim=True)
            img_early_centered = img_early_raw - img_early_raw.mean(dim=0, keepdim=True)
            enc_centered        = enc_raw       - enc_raw.mean(dim=0, keepdim=True)

            logit_scale = self.stage0_logit_scale.exp().clamp(max=100.0)
            late_align  = self._info_nce_align_loss(img_centered,       enc_centered, logit_scale)
            early_align = self._info_nce_align_loss(img_early_centered, enc_centered, logit_scale)
            # without this, both branches are pulled toward the same text target
            # with no incentive to stay distinct, undermining the two-branch premise.
            redundancy_penalty = self._redundancy_loss(img_centered, img_early_centered)
            stage0_align_loss = 0.5 * late_align + 0.5 * early_align + 0.05 * redundancy_penalty



        if is_stage0:
            # Stage 0 uses only the alignment loss.
            # BART is frozen, so CE loss cannot improve generation.
            # Its gradients still flow through the frozen decoder into
            # trainable projection layers, causing unstable growth and
            # eventual fp16 NaN errors. Alignment loss alone is sufficient.
            # loss.py expects "decoder_logits", so return a zero-loss sentinel.
            if answer_ids is None:
                # Inference path: return minimal dict for generate()
                return {
                    "encoder_outputs":   None,
                    "encoder_mask":      None,
                    "rag_vec":           None,
                    "stage0_align_loss": stage0_align_loss,
                }

            # Training path: return align loss only, no CE
            # decoder_logits=None signals loss.py to skip CE computation
            return {
                "loss":               stage0_align_loss,
                "generation_loss":    None,
                "decoder_logits":     None,
                "fsru_feats":         None,
                "rag_vec":            None,
                "answer_type_logits": None,
                "text_feats":         txt,
                "decoder_hidden":     None,
                "stage0_align_loss":  stage0_align_loss,
                "_debug_img":         img,
                "_debug_img_early":   img_early,
                "_debug_enc_hidden":  enc_hidden,
            }


        # ── Stage 1+: full FSRU path ──────────────────────────────────────
        if self.fusion_type == 'fsru':
            freq_tokens = self.fsru(img_early, txt)
            freq_tokens = self.fsru_to_text(freq_tokens)
            freq_tokens = F.layer_norm(freq_tokens, [self.bart.config.d_model])
        elif self.fusion_type == 'cross_attn':
            freq_tokens = self.cross_attn_fusion(img_early, txt)
        else:  
            freq_tokens = torch.cat(
                [torch.zeros_like(img_early), torch.zeros_like(txt)], dim=1
            )

        kagg = None
        if self.use_rag:
            if self.cosine_rag is None:
                self.cosine_rag = CosineRAG(
                    embedding_path = self.args.rag_embeddings_path,
                    device         = next(self.parameters()).device,
                    d_model        = self.args.d_model,
                    top_k          = self.args.top_k,
                    temperature    = getattr(self.args, 'rag_temperature', 0.5),
                    min_similarity = getattr(self.args, 'rag_min_similarity', 0.35),
                    texts_path     = getattr(self.args, 'rag_texts_path', None),
                )
            if self.rag_bridge is None:
                self.rag_bridge = self._build_rag_bridge()
                print(" rag_bridge lazily constructed in forward()")

            rag_direct_query = self._build_rag_query(question_texts, img)

            kagg, _ = self.cosine_rag(direct_query=rag_direct_query)
            kagg = torch.tanh(self.rag_bridge(kagg))
            B_rag  = kagg.size(0)
            device = kagg.device
            rag_type_ids = torch.full(
                (B_rag, kagg.size(1)), 2, dtype=torch.long, device=device
            )
            kagg = kagg + self.source_type_embed(rag_type_ids)

        # ── Prefix memory ────────────────────────────────────────────────
        prefix_memory = None
        if self.use_prefix:
            pooled_summary = freq_tokens.mean(dim=1)
            B = pooled_summary.size(0)
            prefix_memory = self.prefix_projector(pooled_summary)
            prefix_memory = prefix_memory.view(B, self.prefix_length, -1)
            if self.use_rag and kagg is not None:
                kagg_prefix   = kagg.mean(dim=1, keepdim=True)
                prefix_memory = prefix_memory + kagg_prefix
            prefix_memory = F.dropout(
                prefix_memory, p=self.args.prefix_dropout, training=self.training
            )

        if answer_ids is None:
            N_img_early = img_early.size(1)
            vis_fsru    = freq_tokens[:, :N_img_early, :]   # FSRU image branch
            txt_fsru    = freq_tokens[:, N_img_early:, :]   # FSRU text branch

            alpha_early = self._bounded_alpha(self.early_blend_alpha)
            fused_vis   = img + alpha_early * vis_fsru
            alpha_txt   = self._bounded_alpha(self.fsru_blend_alpha)
            fused_txt   = enc_hidden + alpha_txt * txt_fsru
            fused_vis   = self.visual_out_norm(fused_vis)
            fused_txt   = self.text_out_norm(fused_txt)
            img_type_ids = torch.zeros(img.size(0), N_img_early, dtype=torch.long, device=img.device)
            txt_type_ids = torch.ones(img.size(0), fused_txt.size(1), dtype=torch.long, device=img.device)
            fused_vis = fused_vis + self.source_type_embed(img_type_ids)
            fused_txt = fused_txt + self.source_type_embed(txt_type_ids)

            fused_hidden = torch.cat([fused_vis, fused_txt], dim=1)
            fused_mask   = torch.cat(
                [torch.ones(img.size(0), N_img_early, device=img.device), question_mask],
                dim=1
            )
            from transformers.modeling_outputs import BaseModelOutput as _BMO
            fused_enc_out = _BMO(last_hidden_state=fused_hidden)
            return {
                "encoder_outputs":   fused_enc_out,
                "encoder_mask":      fused_mask,
                "rag_vec":           kagg if self.use_rag else None,
                "stage0_align_loss": stage0_align_loss,
            }

        # ── Training / teacher-forced path ───────────────────────────────
        N_img  = img_early.size(1)
        vis_fsru    = freq_tokens[:, :N_img, :]   # FSRU image branch
        txt_fsru    = freq_tokens[:, N_img:, :]   # FSRU text branch

        B      = vis_fsru.size(0)
        device = vis_fsru.device
        N_vis  = vis_fsru.size(1)

        visual_mask = torch.ones(B, N_vis, device=device)
        alpha_early = self._bounded_alpha(self.early_blend_alpha)   
        alpha_txt   = self._bounded_alpha(self.fsru_blend_alpha)    

        visual_blended = img + alpha_early * vis_fsru          
        txt_combined   = enc_hidden + alpha_txt * txt_fsru     
        visual_blended = self.visual_out_norm(visual_blended)
        txt_combined   = self.text_out_norm(txt_combined)

        img_type_ids = torch.zeros(B, N_vis, dtype=torch.long, device=device)
        txt_type_ids = torch.ones(B, txt_combined.size(1), dtype=torch.long, device=device)
        visual_blended = visual_blended + self.source_type_embed(img_type_ids)
        txt_combined   = txt_combined   + self.source_type_embed(txt_type_ids)

        encoder_hidden = torch.cat([visual_blended, txt_combined], dim=1)
        encoder_mask   = torch.cat([visual_mask, question_mask], dim=1)

        if self.use_rag and kagg is not None:
            rag_mask       = torch.ones(B, kagg.size(1), device=device)
            encoder_hidden = torch.cat([kagg, encoder_hidden], dim=1)
            encoder_mask   = torch.cat([rag_mask, encoder_mask], dim=1)

        if prefix_memory is not None:
            encoder_hidden = torch.cat([prefix_memory, encoder_hidden], dim=1)
            prefix_mask    = torch.ones(B, prefix_memory.size(1), device=device)
            encoder_mask   = torch.cat([prefix_mask, encoder_mask], dim=1)

        encoder_hidden = torch.nan_to_num(
            encoder_hidden, nan=0.0, posinf=10.0, neginf=-10.0
        )
        out = self.bart(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden),
            attention_mask=encoder_mask,
            labels=answer_ids,
            output_hidden_states=True
        )
        mask_f = encoder_mask.unsqueeze(-1).float()
        encoder_pooled = (encoder_hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        answer_type_logits = self.answer_type_head(encoder_pooled)

        total_loss = out.loss

        return {
            "loss":               total_loss,
            "generation_loss":    out.loss,
            "decoder_logits":     out.logits,
            "fsru_feats":         freq_tokens,
            "rag_vec":            kagg,
            "answer_type_logits": answer_type_logits,
            "text_feats":         txt,
            "stage0_align_loss":  stage0_align_loss,
            "img_early":          img_early,
        }
    
    @staticmethod
    def _info_nce_align_loss(feat_a: torch.Tensor, feat_b: torch.Tensor,
                              logit_scale: torch.Tensor) -> torch.Tensor:
        """
        Symmetric CLIP-style InfoNCE using in-batch negatives.
        feat_a, feat_b: [B, D] (already DC-centered upstream).
        The diagonal (i == i) is the only positive pair; every other
        (i, j) in the batch is a negative. This is what actually prevents
        projection collapse, the plain cosine-regression objective it
        replaces had no mechanism to push non-matching pairs apart.
        """
        a = F.normalize(feat_a, dim=-1)
        b = F.normalize(feat_b, dim=-1)
        logits = logit_scale * (a @ b.t())          # [B, B]
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_a2b = F.cross_entropy(logits, labels)
        loss_b2a = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_a2b + loss_b2a)

    @staticmethod
    def _redundancy_loss(feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        """Penalizes positive cosine similarity between late (img) and early
        (img_early) branches, enforcing branch distinctness rather than
        assuming Stage 0's shared alignment target will not collapse them."""
        a = F.normalize(feat_a, dim=-1)
        b = F.normalize(feat_b, dim=-1)
        cos_sim = (a * b).sum(dim=-1)
        return cos_sim.mean().clamp(min=0.0)
    def _build_rag_query(self, question_texts, img=None):
        """
        Query is built from raw question text through the frozen retrieval
        encoder, the same encoder used to build the KB in rag_builder.py.
        Does not depend on the trainable BART encoder hidden states.
        """
        device = next(self.retrieval_encoder.parameters()).device
        enc = self.retrieval_tokenizer(
            list(question_texts), padding=True, truncation=True,
            max_length=64, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out    = self.retrieval_encoder(**enc)
            hidden = out.last_hidden_state
            mask   = enc["attention_mask"].unsqueeze(-1).float()
            txt_q  = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        if getattr(self.args, "use_multimodal_rag_query", False):
            if img is None:
                raise ValueError("img is required when use_multimodal_rag_query=True")
            img_q = img.mean(dim=1)
            blend = getattr(self.args, "rag_query_image_weight", 0.2)
            query = (1.0 - blend) * F.normalize(txt_q, dim=-1) + blend * F.normalize(img_q, dim=-1)
        else:
            query = txt_q

        return F.normalize(query, dim=-1)
    def enable_rag(self):
        self.use_rag = True

        if self.retrieval_encoder is None:
            self._init_retrieval_encoder()
            print("🧠 Retrieval encoder (SapBERT) built lazily in enable_rag()")

        if self.cosine_rag is None:
            if hasattr(self.args, 'rag_embeddings_path') and self.args.rag_embeddings_path:
                device = next(self.parameters()).device
                self.cosine_rag = CosineRAG(
                    embedding_path = self.args.rag_embeddings_path,
                    device         = device,
                    d_model        = self.args.d_model,
                    top_k          = self.args.top_k,
                    temperature    = getattr(self.args, 'rag_temperature', 0.5),
                    min_similarity = getattr(self.args, 'rag_min_similarity', 0.35),
                    texts_path     = getattr(self.args, 'rag_texts_path', None),
                )
            else:
                raise RuntimeError(
                    "enable_rag() called but args.rag_embeddings_path is not set. "
                    "Build the RAG database first with rag_builder.py."
                )
        else:
            print(f"🧠 RAG ENABLED — CosineRAG already initialized")

        if self.rag_bridge is None:
            self.rag_bridge = self._build_rag_bridge()
            print("🧠 rag_bridge lazily constructed in enable_rag()")

    @torch.no_grad()

    def generate(self, images, question_ids, question_mask, max_length=128, question_texts=None):
        img, img_early = self.vision_encoder(images, return_early=True)

        enc = self.bart.model.encoder(
            input_ids=question_ids,
            attention_mask=question_mask,
            return_dict=True
        )
        enc_hidden = enc.last_hidden_state
        txt = self.text_to_fsru(enc_hidden)

        # ── FSRU on early features ────────────────────────────────────────
        if self.fusion_type == 'fsru':
            freq_tokens = self.fsru(img_early, txt)
            freq_tokens = self.fsru_to_text(freq_tokens)
            freq_tokens = F.layer_norm(freq_tokens, [self.bart.config.d_model])
        elif self.fusion_type == 'cross_attn':
            freq_tokens = self.cross_attn_fusion(img_early, txt)
        else: 
            freq_tokens = torch.cat(
                [torch.zeros_like(img_early), torch.zeros_like(txt)], dim=1
            )

        N_img        = img_early.size(1)
        vis_fsru     = freq_tokens[:, :N_img, :]
        txt_fsru     = freq_tokens[:, N_img:, :]
        B            = vis_fsru.size(0)
        device       = vis_fsru.device
        N_vis        = vis_fsru.size(1)
        visual_mask  = torch.ones(B, N_vis, device=device)

        alpha_early = self._bounded_alpha(self.early_blend_alpha)
        alpha_txt   = self._bounded_alpha(self.fsru_blend_alpha)

        visual_blended = self.visual_out_norm(img + alpha_early * vis_fsru)
        txt_blended    = self.text_out_norm(enc_hidden + alpha_txt * txt_fsru)

        img_type_ids = torch.zeros(B, N_vis, dtype=torch.long, device=device)
        txt_type_ids = torch.ones(B, txt_blended.size(1), dtype=torch.long, device=device)
        visual_blended = visual_blended + self.source_type_embed(img_type_ids)
        txt_blended    = txt_blended    + self.source_type_embed(txt_type_ids)

        encoder_hidden_states   = torch.cat([visual_blended, txt_blended], dim=1)
        encoder_attention_mask  = torch.cat([visual_mask, question_mask], dim=1)

        if self.use_rag:
            if self.cosine_rag is None:
                raise RuntimeError(
                    "use_rag=True but cosine_rag is None in generate(). "
                    "Call model.enable_rag() first."
                )

            # rag_bridge is guaranteed to exist here: enable_rag() (the only
            # supported way use_rag becomes True before generate() is called)
            # builds it alongside cosine_rag — no lazy fallback needed at
            # inference time.
            rag_direct_query = self._build_rag_query(question_texts, img)

            kagg, _ = self.cosine_rag(direct_query=rag_direct_query)
            kagg = torch.tanh(self.rag_bridge(kagg))
            rag_type_ids = torch.full(
                (B, kagg.size(1)), 2, dtype=torch.long, device=device
            )
            kagg = kagg + self.source_type_embed(rag_type_ids)
            encoder_hidden_states  = torch.cat([kagg, encoder_hidden_states], dim=1)
            encoder_attention_mask = torch.cat(
                [torch.ones(B, kagg.size(1), device=device), encoder_attention_mask],
                dim=1
            )

        if self.use_prefix:
            pooled_summary = freq_tokens.mean(dim=1)          
            prefix_memory  = self.prefix_projector(pooled_summary)
            prefix_memory  = prefix_memory.view(B, self.prefix_length, -1)
            if self.use_rag and kagg is not None:
                prefix_memory = prefix_memory + kagg.mean(dim=1, keepdim=True)
            # training=False means dropout is identity at inference
            prefix_memory = F.dropout(
                prefix_memory, p=self.args.prefix_dropout, training=False
            )
            prefix_mask            = torch.ones(B, prefix_memory.size(1), device=device)
            encoder_hidden_states  = torch.cat([prefix_memory, encoder_hidden_states], dim=1)
            encoder_attention_mask = torch.cat([prefix_mask, encoder_attention_mask], dim=1)

        generated_ids = self.bart.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
            attention_mask=encoder_attention_mask,
            max_length=max_length,
            num_beams=4,
            do_sample=False,
            early_stopping=True,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
        )
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded
