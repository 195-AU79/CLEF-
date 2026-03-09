#!/usr/bin/env python3
"""
CLEF Pipeline — Particle Transformer (ParT) for HH → bbbb
Dual-head architecture:
  1. Classification: signal (HH) vs background (QCD 4b, tt̄+jets)
  2. Regression: κ_λ extraction via parameterized network

Based on: Qu & Gouskos, "Particle Transformer for Jet Tagging" (2022)
          arXiv:2202.03772

Usage:
    python part_model.py --mode train --data-dir ./data/processed/
    python part_model.py --mode infer --checkpoint best_model.pt --input test.h5
"""

import argparse
import numpy as np
import sys
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("ERROR: install PyTorch: pip install torch")
    sys.exit(1)


# ============================================================
# Model hyperparameters
# ============================================================
class ParTConfig:
    """Configuration for Particle Transformer."""
    # Input features
    jet_feat_dim = 5          # pT, eta, phi, mass, btag
    event_feat_dim = 12       # HT, MET, n_jets, ... (from root_to_part.py)
    n_jets_max = 6
    
    # Transformer architecture
    embed_dim = 128           # Embedding dimension
    num_heads = 8             # Attention heads
    num_layers = 6            # Transformer blocks
    ffn_dim = 512             # Feed-forward dimension
    dropout = 0.1
    
    # Interaction features (pairwise)
    pair_feat_dim = 4         # deltaR, delta_pT, delta_eta, delta_phi
    use_pair_features = True
    
    # Task heads
    n_classes = 2             # signal vs background
    n_kl_bins = 20            # Bins for κ_λ (parameterized output)
    kl_range = (-2.0, 6.0)   # κ_λ range for binned output
    
    # Training
    batch_size = 256
    lr = 1e-4
    weight_decay = 1e-5
    epochs = 50
    warmup_epochs = 5
    scheduler = 'cosine'


# ============================================================
# Pairwise interaction features
# ============================================================
def compute_pair_features(jets, mask):
    """
    Compute pairwise jet interaction features.
    
    Args:
        jets: (B, N, 5) — jet features [pT, eta, phi, mass, btag]
        mask: (B, N) — boolean mask
    
    Returns:
        pair_feats: (B, N, N, pair_feat_dim) — pairwise features
    """
    B, N, _ = jets.shape
    
    pt = jets[:, :, 0:1]    # (B, N, 1)
    eta = jets[:, :, 1:2]
    phi = jets[:, :, 2:3]
    
    # Pairwise differences
    deta = eta.unsqueeze(2) - eta.unsqueeze(1)       # (B, N, N, 1)
    dphi = phi.unsqueeze(2) - phi.unsqueeze(1)
    # Wrap dphi
    dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
    
    dR = torch.sqrt(deta**2 + dphi**2 + 1e-8)
    dpt = (pt.unsqueeze(2) - pt.unsqueeze(1)) / (pt.unsqueeze(2) + pt.unsqueeze(1) + 1e-8)
    
    pair_feats = torch.cat([dR, dpt, deta, dphi], dim=-1)  # (B, N, N, 4)
    
    # Mask invalid pairs
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # (B, N, N)
    pair_feats = pair_feats * pair_mask.unsqueeze(-1).float()
    
    return pair_feats


# ============================================================
# Particle Transformer Block
# ============================================================
class ParTBlock(nn.Module):
    """Single Particle Transformer block with pair-aware attention."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.embed_dim
        h = config.num_heads
        
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=config.dropout, batch_first=True)
        
        # Pair bias projection (interaction-aware attention)
        if config.use_pair_features:
            self.pair_proj = nn.Linear(config.pair_feat_dim, h)
        
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, d),
            nn.Dropout(config.dropout),
        )
    
    def forward(self, x, pair_feats=None, mask=None):
        """
        Args:
            x: (B, N, D) — jet embeddings
            pair_feats: (B, N, N, pair_dim) — pairwise features
            mask: (B, N) — attention mask (True = valid)
        """
        # Self-attention with pair bias
        residual = x
        x = self.norm1(x)
        
        # Compute attention bias from pair features
        attn_mask = None
        if pair_feats is not None and self.config.use_pair_features:
            # (B, N, N, pair_dim) → (B, N, N, n_heads) → (B*n_heads, N, N)
            pair_bias = self.pair_proj(pair_feats)  # (B, N, N, h)
            B, N, _, h = pair_bias.shape
            pair_bias = pair_bias.permute(0, 3, 1, 2)  # (B, h, N, N)
            pair_bias = pair_bias.reshape(B * h, N, N)
            attn_mask = pair_bias
        
        # Key padding mask
        key_padding_mask = ~mask if mask is not None else None
        
        x, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = residual + x
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x


# ============================================================
# Full Particle Transformer
# ============================================================
class ParticleTransformer(nn.Module):
    """
    Particle Transformer for HH → bbbb
    
    Architecture:
      Input → Embed → N × ParTBlock → Pool → Classification + κ_λ Regression
    """
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ParTConfig()
        cfg = self.config
        
        # Input embedding
        self.jet_embed = nn.Sequential(
            nn.Linear(cfg.jet_feat_dim, cfg.embed_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.embed_dim),
        )
        
        # Event feature embedding
        self.event_embed = nn.Sequential(
            nn.Linear(cfg.event_feat_dim, cfg.embed_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.embed_dim),
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            ParTBlock(cfg) for _ in range(cfg.num_layers)
        ])
        
        self.norm_final = nn.LayerNorm(cfg.embed_dim)
        
        # --- Classification head: signal vs background ---
        self.cls_head = nn.Sequential(
            nn.Linear(cfg.embed_dim * 2, 256),  # *2 for event features
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, cfg.n_classes),
        )
        
        # --- κ_λ regression head ---
        # Parameterized neural network: outputs binned κ_λ posterior
        self.kl_head = nn.Sequential(
            nn.Linear(cfg.embed_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, cfg.n_kl_bins),
        )
        
        # κ_λ bin centers for posterior mean extraction
        kl_edges = torch.linspace(cfg.kl_range[0], cfg.kl_range[1], cfg.n_kl_bins + 1)
        self.register_buffer('kl_bin_centers', (kl_edges[:-1] + kl_edges[1:]) / 2)
    
    def forward(self, jets, event_feats, mask):
        """
        Args:
            jets: (B, N_jets, jet_feat_dim)
            event_feats: (B, event_feat_dim)
            mask: (B, N_jets) boolean
        
        Returns:
            cls_logits: (B, 2) — signal/background
            kl_posterior: (B, n_kl_bins) — softmax over κ_λ bins
            kl_mean: (B,) — posterior mean κ_λ
        """
        B = jets.shape[0]
        
        # Embed jets
        x = self.jet_embed(jets)  # (B, N, D)
        
        # Compute pair features
        pair_feats = compute_pair_features(jets, mask) if self.config.use_pair_features else None
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, pair_feats, mask)
        
        x = self.norm_final(x)
        
        # Global pooling (masked mean)
        mask_float = mask.unsqueeze(-1).float()
        x_pooled = (x * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-8)
        
        # Concatenate event-level features
        evt = self.event_embed(event_feats)
        combined = torch.cat([x_pooled, evt], dim=-1)  # (B, 2*D)
        
        # Classification
        cls_logits = self.cls_head(combined)
        
        # κ_λ regression (binned posterior)
        kl_logits = self.kl_head(combined)
        kl_posterior = F.softmax(kl_logits, dim=-1)
        kl_mean = (kl_posterior * self.kl_bin_centers.unsqueeze(0)).sum(dim=-1)
        
        return cls_logits, kl_posterior, kl_mean


# ============================================================
# Dataset
# ============================================================
class HHbbbbDataset(Dataset):
    """Dataset for HH → bbbb from HDF5 files."""
    
    def __init__(self, h5_files, kl_target=None):
        import h5py
        self.jets_list = []
        self.event_feats_list = []
        self.mask_list = []
        self.labels_list = []
        self.kl_values = []
        
        for fpath in h5_files:
            with h5py.File(fpath, 'r') as f:
                self.jets_list.append(f['jets'][:])
                self.event_feats_list.append(f['event_features'][:])
                self.mask_list.append(f['jet_mask'][:])
                self.labels_list.append(f['labels'][:])
                kl = f.attrs.get('kappa_lambda', 1.0)
                self.kl_values.extend([kl] * len(f['jets']))
        
        self.jets = np.concatenate(self.jets_list)
        self.event_feats = np.concatenate(self.event_feats_list)
        self.masks = np.concatenate(self.mask_list)
        self.labels = np.concatenate(self.labels_list)
        self.kl_values = np.array(self.kl_values, dtype=np.float32)
    
    def __len__(self):
        return len(self.jets)
    
    def __getitem__(self, idx):
        return {
            'jets': torch.FloatTensor(self.jets[idx]),
            'event_feats': torch.FloatTensor(self.event_feats[idx]),
            'mask': torch.BoolTensor(self.masks[idx]),
            'label': torch.LongTensor([self.labels[idx]])[0],
            'kl': torch.FloatTensor([self.kl_values[idx]])[0],
        }


# ============================================================
# Training
# ============================================================
def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    n_batches = 0
    
    cls_criterion = nn.CrossEntropyLoss()
    kl_criterion = nn.MSELoss()
    
    for batch in loader:
        jets = batch['jets'].to(device)
        event_feats = batch['event_feats'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['label'].to(device)
        kl_true = batch['kl'].to(device)
        
        cls_logits, kl_posterior, kl_mean = model(jets, event_feats, mask)
        
        # Combined loss
        loss_cls = cls_criterion(cls_logits, labels)
        loss_kl = kl_criterion(kl_mean, kl_true)
        
        # For signal events only, weight κ_λ loss
        signal_mask = (labels == 1).float()
        loss_kl_weighted = (loss_kl * signal_mask).sum() / (signal_mask.sum() + 1e-8)
        
        loss = loss_cls + 0.5 * loss_kl_weighted
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_cls_preds = []
    all_labels = []
    all_kl_pred = []
    all_kl_true = []
    all_kl_posterior = []
    
    for batch in loader:
        jets = batch['jets'].to(device)
        event_feats = batch['event_feats'].to(device)
        mask = batch['mask'].to(device)
        
        cls_logits, kl_posterior, kl_mean = model(jets, event_feats, mask)
        
        all_cls_preds.append(F.softmax(cls_logits, dim=-1)[:, 1].cpu().numpy())
        all_labels.append(batch['label'].numpy())
        all_kl_pred.append(kl_mean.cpu().numpy())
        all_kl_true.append(batch['kl'].numpy())
        all_kl_posterior.append(kl_posterior.cpu().numpy())
    
    return {
        'cls_preds': np.concatenate(all_cls_preds),
        'labels': np.concatenate(all_labels),
        'kl_pred': np.concatenate(all_kl_pred),
        'kl_true': np.concatenate(all_kl_true),
        'kl_posterior': np.concatenate(all_kl_posterior),
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="CLEF Pipeline — ParT Training")
    parser.add_argument('--mode', choices=['train', 'infer'], default='train')
    parser.add_argument('--data-dir', type=str, default='./data/processed/')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output', type=str, default='./results/')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"CLEF Pipeline — Particle Transformer")
    print(f"{'='*50}")
    print(f"Mode:   {args.mode}")
    print(f"Device: {device}")
    print()
    
    config = ParTConfig()
    config.epochs = args.epochs
    model = ParticleTransformer(config).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    if args.mode == 'train':
        # Collect HDF5 files
        data_dir = Path(args.data_dir)
        h5_files = sorted(data_dir.glob('*.h5'))
        
        if not h5_files:
            print(f"No .h5 files found in {data_dir}")
            print("Run root_to_part.py first to convert Delphes output.")
            sys.exit(1)
        
        print(f"Found {len(h5_files)} data files")
        
        # Split train/val (80/20)
        n_train = int(0.8 * len(h5_files))
        train_ds = HHbbbbDataset(h5_files[:n_train])
        val_ds = HHbbbbDataset(h5_files[n_train:])
        
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        
        best_val_loss = float('inf')
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(config.epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
            val_results = evaluate(model, val_loader, device)
            scheduler.step()
            
            # Metrics
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(val_results['labels'], val_results['cls_preds'])
            except:
                auc = 0.0
            
            sig_mask = val_results['labels'] == 1
            kl_rmse = np.sqrt(np.mean((val_results['kl_pred'][sig_mask] - val_results['kl_true'][sig_mask])**2)) if sig_mask.sum() > 0 else 0.0
            
            print(f"Epoch {epoch+1:3d}/{config.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val AUC: {auc:.4f} | "
                  f"κ_λ RMSE: {kl_rmse:.4f}")
            
            # Save best
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config.__dict__,
                    'epoch': epoch,
                    'auc': auc,
                    'kl_rmse': kl_rmse,
                }, output_dir / 'best_model.pt')
        
        print(f"\nTraining complete. Best model saved to {output_dir / 'best_model.pt'}")
    
    elif args.mode == 'infer':
        if args.checkpoint is None:
            print("ERROR: --checkpoint required for inference mode")
            sys.exit(1)
        
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint (epoch {checkpoint['epoch']}, AUC={checkpoint['auc']:.4f})")


if __name__ == '__main__':
    main()
