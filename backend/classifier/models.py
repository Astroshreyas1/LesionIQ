import torch
import torch.nn as nn
import timm
from backend.classifier.config import NUM_CLASSES

class LesionIQHybrid(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, meta_dim=13, mode='full', pretrained=True):
        """
        mode options:
          'effnet_only'  — Experiment A
          'swin_only'    — Experiment B  
          'image_only'   — Experiment C (both backbones, no metadata)
          'full'         — Experiment D (your full model)
        """
        super().__init__()
        self.mode = mode

        if mode in ('effnet_only', 'image_only', 'full'):
            self.effnet = timm.create_model(
                'efficientnet_b4', pretrained=pretrained, num_classes=0)
            eff_dim = self.effnet.num_features  # 1792
        
        if mode in ('swin_only', 'image_only', 'full'):
            self.swin = timm.create_model(
                'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k', pretrained=pretrained, num_classes=0)
            swin_dim = self.swin.num_features   # 1024

        if mode == 'full':
            self.meta_mlp = nn.Sequential(
                nn.Linear(meta_dim, 64), nn.BatchNorm1d(64),
                nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, 32), nn.ReLU()
            )

        # Compute fusion dim based on mode
        fusion_dim = {
            'effnet_only': 1792,
            'swin_only':   1024,
            'image_only':  1792 + 1024,
            'full':        1792 + 1024 + 32
        }[mode]

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.BatchNorm1d(512),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Auto-freeze backbones for image_only and full modes
        if mode in ('image_only', 'full'):
            self.freeze_backbones()

    def freeze_backbones(self):
        """Freeze all backbone layers except the last stage.
        This prevents the 88M+ pretrained parameters from overfitting
        while still allowing the final feature extraction layers to adapt."""
        frozen_count = 0
        
        if hasattr(self, 'effnet'):
            # Freeze all EfficientNet blocks except the last one (blocks.6)
            for name, param in self.effnet.named_parameters():
                if not name.startswith('blocks.6') and not name.startswith('conv_head') and not name.startswith('bn2'):
                    param.requires_grad = False
                    frozen_count += 1
        
        if hasattr(self, 'swin'):
            # Freeze all Swin stages except the last one (layers.3)
            for name, param in self.swin.named_parameters():
                if not name.startswith('layers.3') and not name.startswith('norm'):
                    param.requires_grad = False
                    frozen_count += 1
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f'[FREEZE] Frozen {frozen_count} param groups | '
              f'Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M total')

    def forward(self, img, meta=None):
        features = []

        if self.mode in ('effnet_only', 'image_only', 'full'):
            features.append(self.effnet(img))

        if self.mode in ('swin_only', 'image_only', 'full'):
            features.append(self.swin(img))

        if self.mode == 'full':
            if meta is None:
                raise ValueError(
                    "full mode requires metadata tensor (13-d). "
                    "Got meta=None. Use mode='image_only' if metadata "
                    "is unavailable, or pass a zero tensor as fallback."
                )
            features.append(self.meta_mlp(meta))
        elif meta is not None:
            import warnings
            warnings.warn(
                f"Metadata tensor passed to '{self.mode}' mode — "
                f"metadata is only used in 'full' mode. Ignoring.",
                stacklevel=2,
            )

        return self.classifier(torch.cat(features, dim=1))
