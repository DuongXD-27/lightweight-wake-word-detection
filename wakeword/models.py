"""Small keyword-spotting backbones. All models output one logit per window (wake vs not-wake).

* ``ds_cnn``   - Depthwise-separable CNN (Zhang et al., "Hello Edge", 2017)       input (B,1,F,T)
* ``tc_resnet`` - Temporal-convolution ResNet (Choi et al., Interspeech 2019)    input (B,1,F,T)
* ``crnn``     - Conv + GRU                                                      input (B,1,F,T)
* ``emb_head`` - small classifier on frozen openWakeWord embeddings             input (B,T,96)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatNorm(nn.Module):
    """Per-frequency-bin BatchNorm on (B,1,F,T) features (acts as learned global CMVN)."""

    def __init__(self, n_freq: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_freq)

    def forward(self, x):
        return self.bn(x.squeeze(1)).unsqueeze(1)


# ------------------------------------------------------------------------------------- DS-CNN
class DSBlock(nn.Module):
    def __init__(self, c: int, stride=(1, 1)):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, stride=stride, padding=1, groups=c, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        x = F.relu(self.bn1(self.dw(x)))
        return F.relu(self.bn2(self.pw(x)))


class DSCNN(nn.Module):
    def __init__(self, in_shape, channels: int = 64, n_blocks: int = 4, dropout: float = 0.2):
        super().__init__()
        n_freq = in_shape[1]
        self.norm = FeatNorm(n_freq)
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=(4, 10), stride=(2, 2), padding=(1, 4), bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[DSBlock(channels, (2, 2) if i == 1 else (1, 1)) for i in range(n_blocks)])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        x = self.blocks(self.stem(self.norm(x)))
        x = x.mean(dim=(2, 3))
        return self.fc(self.drop(x)).squeeze(-1)


# ---------------------------------------------------------------------------------- TC-ResNet
class TCBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int, k: int = 9):
        super().__init__()
        self.c1 = nn.Conv1d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(c_out)
        self.c2 = nn.Conv1d(c_out, c_out, k, padding=k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(c_out)
        self.short = None
        if stride != 1 or c_in != c_out:
            self.short = nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False),
                                       nn.BatchNorm1d(c_out), nn.ReLU(inplace=True))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        s = x if self.short is None else self.short(x)
        return F.relu(y + s)


class TCResNet(nn.Module):
    """TC-ResNet8 (layers=3) / TC-ResNet14 (layers=6) with width multiplier."""

    def __init__(self, in_shape, width: float = 1.0, layers: int = 3, kernel: int = 9, dropout: float = 0.2):
        super().__init__()
        n_freq = in_shape[1]
        base = [16, 24, 32, 48]
        ch = [int(round(c * width)) for c in base]
        self.norm = FeatNorm(n_freq)
        self.stem = nn.Conv1d(n_freq, ch[0], 3, padding=1, bias=False)
        blocks, c_in = [], ch[0]
        for i in range(layers):
            stage = min(1 + i // max(1, layers // 3), 3)
            c_out = ch[stage]
            stride = 2 if (layers == 3 or i % 2 == 0) else 1
            blocks.append(TCBlock(c_in, c_out, stride, kernel))
            c_in = c_out
        self.blocks = nn.Sequential(*blocks)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c_in, 1)

    def forward(self, x):
        x = self.norm(x).squeeze(1)  # (B, F, T): frequency bins become channels
        x = self.blocks(self.stem(x))
        return self.fc(self.drop(x.mean(dim=-1))).squeeze(-1)


# --------------------------------------------------------------------------------------- CRNN
class CRNN(nn.Module):
    def __init__(self, in_shape, channels: int = 32, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        n_freq = in_shape[1]
        self.norm = FeatNorm(n_freq)
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, (5, 3), stride=(2, 1), padding=(2, 1), bias=False), nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, (5, 3), stride=(2, 2), padding=(2, 1), bias=False), nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        f_out = (n_freq + 1) // 2
        f_out = (f_out + 1) // 2
        self.gru = nn.GRU(channels * f_out, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.conv(self.norm(x))  # (B, C, F', T')
        B, C, Fq, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * Fq)
        out, _ = self.gru(x)
        return self.fc(self.drop(out[:, -1])).squeeze(-1)


# ------------------------------------------------------------------------- embedding head (C)
class EmbeddingHead(nn.Module):
    """openWakeWord-style head: flattened (T,96) embeddings -> MLP. ``kind='gru'`` for a GRU head."""

    def __init__(self, in_shape, hidden: int = 128, kind: str = "mlp", dropout: float = 0.3):
        super().__init__()
        T, D = in_shape
        self.kind = kind
        if kind == "mlp":
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(T * D, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
                                     nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        else:
            self.gru = nn.GRU(D, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        if self.kind == "mlp":
            return self.net(x).squeeze(-1)
        out, _ = self.gru(x)
        return self.fc(out[:, -1]).squeeze(-1)


MODELS = {"ds_cnn": DSCNN, "tc_resnet": TCResNet, "crnn": CRNN, "emb_head": EmbeddingHead}


def build_model(cfg, in_shape) -> nn.Module:
    mcfg = dict(cfg.model)
    name = mcfg.pop("name")
    if name not in MODELS:
        raise ValueError(f"unknown model {name}; choose from {list(MODELS)}")
    import inspect

    cls = MODELS[name]
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self", "in_shape"}
    ignored = sorted(set(mcfg) - accepted)
    if ignored:
        print(f"[model] ignoring keys not used by {name}: {ignored}")
    return cls(in_shape, **{k: v for k, v in mcfg.items() if k in accepted})


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_macs(model: nn.Module, in_shape) -> int:
    """Multiply-accumulate count for one window (conv / linear / GRU), via forward hooks."""
    macs = [0]

    def conv_hook(m, inp, out):
        k = m.weight.numel() // m.out_channels  # (C_in/groups) * prod(kernel)
        macs[0] += int(out.numel() * k)

    def lin_hook(m, inp, out):
        macs[0] += int(m.in_features * m.out_features * (out.numel() // m.out_features))

    def gru_hook(m, inp, out):
        x = inp[0]
        T = x.shape[1] if m.batch_first else x.shape[0]
        H, I = m.hidden_size, m.input_size
        macs[0] += int(T * 3 * (I * H + H * H))

    hooks = []
    for mod in model.modules():
        if isinstance(mod, (nn.Conv1d, nn.Conv2d)):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(lin_hook))
        elif isinstance(mod, nn.GRU):
            hooks.append(mod.register_forward_hook(gru_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *in_shape))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return macs[0]
