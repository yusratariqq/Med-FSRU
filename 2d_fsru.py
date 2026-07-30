# fsru.py


import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

def _bounded_gate(x: torch.Tensor, floor: float = 0.1, ceil: float = 0.9) -> torch.Tensor:
    return floor + (ceil - floor) * torch.sigmoid(x)

class _ComplexFilterBank1D(nn.Module):
    def __init__(self, seq_half: int, d_model: int, num_filter: int = 2):
        super().__init__()
        self.num_filter = num_filter
        init = torch.zeros(num_filter, seq_half, d_model, 2)
        init[..., 0] = 1.0
        init[..., 1] = torch.randn(num_filter, seq_half, d_model) * 0.01
        self.filter_bank = nn.Parameter(init)
        self.mix_head = nn.Linear(d_model, num_filter)
        nn.init.zeros_(self.mix_head.weight)
        nn.init.zeros_(self.mix_head.bias)   

    def forward(self, X_freq: torch.Tensor) -> torch.Tensor:
        fb = torch.view_as_complex(self.filter_bank)
        cond = X_freq.abs().mean(dim=1)         
        weights = F.softmax(self.mix_head(cond), dim=-1)   
        out = torch.zeros_like(X_freq)
        for k in range(self.num_filter):
            w_k = weights[:, k].view(-1, 1, 1)   
            out = out + w_k * (X_freq * fb[k])
        return out


class _FrequencyGate1D(nn.Module):
    """
    Cross-modal gate driven by a 1D rfft spectrum.
    Produces a real gate [B, 1, D] to multiply against a target spectrum.

    NOTE: retained for interface/checkpoint compatibility but not currently
    instantiated by FSRU (FSRU uses _FrequencyGate2D / _Image2TextGate).
    Left untouched since it carries no gate-collapse risk while unused.
    """
    def __init__(self, source_seq_half: int, d_model: int):
        super().__init__()
        self.select_para = nn.Parameter(
            torch.randn(source_seq_half, d_model, 2) * 0.01
        )
        self.avg_pool = nn.AvgPool1d(kernel_size=source_seq_half)
        self.conv     = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, source_freq: torch.Tensor) -> torch.Tensor:
        """source_freq: complex [B, src_seq_half, D] → real gate [B, 1, D]"""
        sp    = torch.view_as_complex(self.select_para)   
        gated = (source_freq * sp).real                   
        gated = gated.permute(0, 2, 1)                    
        gated = self.avg_pool(gated)                      
        gated = self.conv(gated)                          
        gated = gated.permute(0, 2, 1)                    
        return gated

class _ComplexFilterBank2D(nn.Module):
    """
    Learnable 2D complex filter bank elementwise spectral
    filtering over the [H_f, W_f] spatial-frequency plane, phase-preserving.

    """
    def __init__(self, freq_h: int, freq_w: int, d_model: int, num_filter: int = 2):
        super().__init__()
        self.num_filter = num_filter
        self.freq_h     = freq_h
        self.freq_w     = freq_w
        init = torch.zeros(num_filter, freq_h, freq_w, d_model, 2)
        init[..., 0] = 1.0
        init[..., 1] = torch.randn(num_filter, freq_h, freq_w, d_model) * 0.01
        self.filter_bank = nn.Parameter(init)
        self.mix_head = nn.Linear(d_model, num_filter)
        nn.init.zeros_(self.mix_head.weight)
        nn.init.zeros_(self.mix_head.bias)   

    def forward(self, X_freq: torch.Tensor) -> torch.Tensor:
        """X_freq: complex [B, H_f, W_f, D] → complex [B, H_f, W_f, D]"""
        fb = torch.view_as_complex(self.filter_bank)     
        cond = X_freq.abs().mean(dim=(1, 2))               
        weights = F.softmax(self.mix_head(cond), dim=-1)    
        out = torch.zeros_like(X_freq)
        for k in range(self.num_filter):
            w_k = weights[:, k].view(-1, 1, 1, 1)
            out = out + w_k * (X_freq * fb[k])
        return out


class _AttentionPool1D(nn.Module):
    """
    Learnable-query attention pooling over a sequence dimension.
    Replaces AvgPool1d, which collapses all positional/content variation
    into one unweighted mean before the gate ever sees it letting the
    pool itself weight positions differently instead of throwing that
    signal away before the gate's linear layers get a chance to use it.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.scale = d_model ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:   
        q = self.query.expand(x.size(0), -1, -1)            
        attn = torch.softmax((q @ x.transpose(1, 2)) * self.scale, dim=-1)  
        return (attn @ x).squeeze(1)                          


class _FrequencyGate2D(nn.Module):
    def __init__(self, src_seq_half: int, d_model: int, freq_h: int, freq_w: int):
        super().__init__()
        self.freq_h, self.freq_w = freq_h, freq_w
        self.select_para = nn.Parameter(
            torch.randn(src_seq_half, d_model, 2) * 0.1
        )
        self.pool = _AttentionPool1D(d_model)                 
        self.pre_gate_norm = nn.LayerNorm(d_model)
        self.channel_fc = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.channel_fc.weight, gain=0.1)  
        nn.init.zeros_(self.channel_fc.bias)                        
        self.band_proj = nn.Linear(d_model, freq_h * freq_w)
        nn.init.xavier_uniform_(self.band_proj.weight, gain=0.1)    
        nn.init.zeros_(self.band_proj.bias)                           

    def forward(self, source_1d_freq: torch.Tensor) -> torch.Tensor:
        sp     = torch.view_as_complex(self.select_para)
        gated  = (source_1d_freq * sp).real                
        pooled = self.pool(gated)                            
        pooled = self.pre_gate_norm(pooled)                    # stabilize gate input scale

        channel_gate = _bounded_gate(self.channel_fc(pooled))
        channel_gate = channel_gate.view(-1, 1, 1, channel_gate.size(-1))
        band_gate = _bounded_gate(self.band_proj(pooled))
        band_gate = band_gate.view(-1, self.freq_h, self.freq_w, 1)
        return channel_gate * band_gate


class _Image2TextGate(nn.Module):
    def __init__(self, freq_h: int, freq_w: int, d_model: int, txt_half: int):
        super().__init__()
        self.freq_h, self.freq_w = freq_h, freq_w
        self.txt_half = txt_half
        self.select_para = nn.Parameter(
            torch.randn(freq_h, freq_w, d_model, 2) * 0.1
        )
        self.pool = _AttentionPool1D(d_model)                       
        self.pre_gate_norm = nn.LayerNorm(d_model)
        self.channel_fc = nn.Linear(d_model, d_model)                 
        nn.init.xavier_uniform_(self.channel_fc.weight, gain=0.1)    
        nn.init.zeros_(self.channel_fc.bias)                            
        self.pos_proj = nn.Linear(d_model, txt_half)
        nn.init.xavier_uniform_(self.pos_proj.weight, gain=0.1)       
        nn.init.zeros_(self.pos_proj.bias)                            

    def forward(self, img_2d_freq: torch.Tensor) -> torch.Tensor:
        """img_2d_freq: complex [B, H_f, W_f, D] → real gate [B, txt_half, D]"""
        sp    = torch.view_as_complex(self.select_para)    
        gated = (img_2d_freq * sp).real                   
        B, H, W, D = gated.shape
        gated  = gated.reshape(B, H * W, D)                 
        pooled = self.pool(gated)                               
        pooled = self.pre_gate_norm(pooled)                      # stabilize gate input scale

        #  bounded gate instead of raw sigmoid to avoid gate collapse.
        channel_gate = _bounded_gate(self.channel_fc(pooled))  

        pos_gate = _bounded_gate(self.pos_proj(pooled))         
        pos_gate = pos_gate.unsqueeze(-1)                          

        return channel_gate.unsqueeze(1) * pos_gate    


class _AddNorm(nn.Module):
    """Add & Norm with FeedForward — unchanged from original."""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        residual = x
        x = self.drop(x)
        x = self.ff(x) + residual
        return self.norm2(x)


class FSRU(nn.Module):
    """
    Frequency Spectral Representation Unit

    Image branch: 2D FFT over the 14×14 ViT patch grid.
    Text branch:  1D FFT over the token sequence (FNet-style token mixing).

    The cross-modal E&S gate retains its original direction:
      text 1D spectrum → gate applied to image 2D spectrum
      image 2D spectrum → gate applied to text 1D spectrum

    Semantic meaning of image 2D gate:
      Low spatial frequency components (top-left of 14×14 DFT) encode global
      patterns (large organ structures, diffuse density changes).
      High spatial frequency components encode fine-grain features (lesion
      boundaries, texture). The text question's learned gate selects which
      band to emphasize for the answer.

    Interface preserved: forward(image_feats, text_feats) → [B, Ni+Nt, d_model]
    Only the INTERNAL computation changes.
    """

    # The ViT-B/16 patch grid for a 224×224 image
    GRID_H = 14
    GRID_W = 14

    def __init__(
        self,
        d_model:    int,
        img_seq:    int = 196,   # must equal GRID_H * GRID_W
        txt_seq:    int = 128,
        num_filter: int = 2,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.d_model   = d_model
        self.img_seq   = img_seq
        self.txt_seq   = txt_seq

        assert img_seq == self.GRID_H * self.GRID_W, (
            f"img_seq must equal {self.GRID_H}*{self.GRID_W}={self.GRID_H*self.GRID_W} "
            f"for 2D FFT. Got img_seq={img_seq}."
        )

        # 2D rfft2 over [14, 14] produces [14, 8] complex coefficients
        # (rfft2 halves the last FFT dimension: W//2 + 1 = 7+1 = 8)
        self.freq_h = self.GRID_H          # 14
        self.freq_w = self.GRID_W // 2 + 1 # 8

        # 1D rfft over txt_seq
        self.txt_half = txt_seq // 2 + 1   # 65

        print(
            f"[FSRU 2D init] grid={self.GRID_H}×{self.GRID_W} "
            f"→ freq_plane={self.freq_h}×{self.freq_w} complex coefficients | "
            f"txt_half={self.txt_half}"
        )

        # ── IMAGE BRANCH: 2D filter bank ─────────────────────────────────
        self.img_filter_2d = _ComplexFilterBank2D(
            self.freq_h, self.freq_w, d_model, num_filter
        )
        # absolute 2D position injected before the translation-equivariant
        # FFT branch, since rfft2/irfft2 filtering is circular-convolution-
        # equivalent and would otherwise treat the grid as toroidal.
        self.img_pos_embed = nn.Parameter(torch.randn(1, self.GRID_H, self.GRID_W, d_model) * 0.02)

        # ── TEXT BRANCH: 1D filter bank (original logic) ─────────────────
        self.txt_filter_1d = _ComplexFilterBank1D(self.txt_half, d_model, num_filter)
        # ── CROSS-MODAL GATES ─────────────────────────────────────────────
        # Text (1D) spectrum drives the 2D image gate, now band-selective
        self.text2img_gate = _FrequencyGate2D(self.txt_half, d_model, self.freq_h, self.freq_w)
        # Image (2D) spectrum drives the 1D text gate , now position-selective
        self.img2text_gate = _Image2TextGate(self.freq_h, self.freq_w, d_model, self.txt_half)

        # ── ADD & NORM ────────────────────────────────────────────────────
        self.img_add_norm = _AddNorm(d_model, dropout)
        self.txt_add_norm = _AddNorm(d_model, dropout)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

        # Diagnostic attribute read by train.py
        self.gate_activation_mean = 0.5

    def forward(
        self,
        image_feats: torch.Tensor,  
        text_feats:  torch.Tensor,   
    ) -> torch.Tensor:
        """
        Returns z_freq: [B, 196+T, d_model]
        Shape contract with model.py is fully preserved.
        """
        B, Ni, D = image_feats.shape
        B, Nt, _ = text_feats.shape

        assert Ni == self.img_seq, (
            f"[FSRU] Expected {self.img_seq} image tokens, got {Ni}"
        )

        x_img = image_feats   
        x_txt = text_feats

        # ── 1a. Reshape image tokens to 2D patch grid ─────────────────────
        img_2d = image_feats.view(B, self.GRID_H, self.GRID_W, D)
        img_2d = img_2d + self.img_pos_embed   


        # ── 1b. 2D rfft over spatial patch grid ───────────────────────────
        
        with torch.amp.autocast('cuda', enabled=False):
            img_freq = fft.rfft2(
                img_2d.float(), dim=(1, 2), norm='ortho'
            )   # [B, freq_h=14, freq_w=8, D]

        # ── 1c. 1D rfft over text sequence ────────────────────────────────
        with torch.amp.autocast('cuda', enabled=False):
            txt_freq = fft.rfft(
                text_feats.float(), dim=1, norm='ortho'
            )   

        # ── 2. USC: Uni-modal Spectrum Compression ────────────────────────
        img_compressed = self.img_filter_2d(img_freq)   
        txt_compressed = self.txt_filter_1d(txt_freq)  

        # ── 3. E&S: Cross-modal Emphasize-and-Suppress ────────────────────
        img_gate = self.text2img_gate(txt_compressed)   
        txt_gate = self.img2text_gate(img_compressed)   

        img_selected = img_compressed * img_gate   
        txt_selected = txt_compressed * txt_gate   

        with torch.no_grad():
          self.gate_activation_mean = float(
              (img_gate.detach().abs().mean() + txt_gate.detach().abs().mean()) / 2
          )
          self.gate_activation_std = float(
              (img_gate.detach().std() + txt_gate.detach().std()) / 2
          )
          self._last_img_gate = img_gate.detach()   #  per-sample gate, used for question-type breakdown

        # ── 4a. 2D irfft → back to patch grid ────────────────────────────
        with torch.amp.autocast('cuda', enabled=False):
            img_spatial = fft.irfft2(
                img_selected,
                s=(self.GRID_H, self.GRID_W),   
                dim=(1, 2),
                norm='ortho'
            ).real   

        # Flatten back to sequence: [B, 196, D]
        img_spatial = img_spatial.reshape(B, Ni, D)

        # ── 4b. 1D irfft → back to text sequence ─────────────────────────
        with torch.amp.autocast('cuda', enabled=False):
            txt_spatial = fft.irfft(
                txt_selected, n=Nt, dim=1, norm='ortho'
            ).real   

        # NaN guard
        img_spatial = torch.nan_to_num(img_spatial, nan=0.0, posinf=1.0, neginf=-1.0)
        txt_spatial = torch.nan_to_num(txt_spatial, nan=0.0, posinf=1.0, neginf=-1.0)

        # ── 5. Add & Norm with residual ───────────────────────────────────
        img_out = self.img_add_norm(img_spatial + x_img)   
        txt_out = self.txt_add_norm(txt_spatial + x_txt)   

        # ── 6. Concatenate and project ────────────────────────────────────
        fused  = torch.cat([img_out, txt_out], dim=1)   
        fused  = self.dropout(fused)
        z_freq = self.out_proj(fused)                    
        z_freq = F.layer_norm(z_freq, [self.d_model])

        return z_freq

class CrossAttnFusion(nn.Module):
    """
    Optional cross-attention fusion module kept for future experiments.
    """
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.img2txt_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.txt2img_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.img_norm = nn.LayerNorm(d_model)
        self.txt_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, image_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        img_ctx, _ = self.txt2img_attn(image_feats, text_feats, text_feats)
        txt_ctx, _ = self.img2txt_attn(text_feats, image_feats, image_feats)
        img_out = self.img_norm(image_feats + img_ctx)
        txt_out = self.txt_norm(text_feats + txt_ctx)
        fused = torch.cat([img_out, txt_out], dim=1)
        return F.layer_norm(self.out_proj(fused), [fused.size(-1)])
