"""
hrm_model.py'deki sınıfları (SwiGLU, HRMBlock, ReasoningModule, TinyHrm) değiştirmeden saran katman.

Eklenenler:
- GridEncoder     : belirteç + 2B konum gömmesi. hrm_model.py'deki dikkat katmanlarının konum bilgisi
                    yok; konum girdiye eklenir ve her L adımında "girdi enjeksiyonu" ile yeniden verilir.
- HRMReasoner     : TinyHrm'in L/H modüllerini kullanır. İki eğitim biçimi:
                      "bptt"     -> TinyHrm.forward (sıfırdan başlar, tüm döngü boyunca geri yayılım)
                      "one_step" -> makaledeki tek adımlı gradyan: son L ve son H adımı dışında
                                    gradyan tutulmaz (bellek döngü sayısından bağımsız)
                    ve derin denetim için taşınan durum (carry) + durma başı (q_head).
- LoopedNet       : tek ReasoningModule'ün tekrar tekrar uygulanması (hiyerarşisiz karşılaştırma).
- TransformerNet  : HRMBlock yığını, döngüsüz (klasik karşılaştırma).
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hrm_model import HRMBlock, ReasoningModule, TinyHrm  # noqa: E402  (kullanıcının modeli)


class GridEncoder(nn.Module):
    def __init__(self, vocab, rows, cols, hidden, use_position=True):
        super().__init__()
        self.tok = nn.Embedding(vocab, hidden)
        self.use_position = use_position
        self.rows, self.cols = rows, cols
        self.row = nn.Embedding(rows, hidden)
        self.col = nn.Embedding(cols, hidden)
        idx = torch.arange(rows * cols)
        self.register_buffer("r_idx", idx // cols, persistent=False)
        self.register_buffer("c_idx", idx % cols, persistent=False)

    def forward(self, tokens):
        x = self.tok(tokens)
        if self.use_position:
            x = x + self.row(self.r_idx) + self.col(self.c_idx)
        return x


class _Base(nn.Module):
    """Ortak: kodlayıcı, hücre başı ve durma başı. Alt sınıf `reason(x, carry)` yazar."""

    recurrent = False

    def __init__(self, task, hidden, heads, use_position=True):
        super().__init__()
        self.task = task
        self.encoder = GridEncoder(task.vocab, task.rows, task.cols, hidden, use_position)
        self.head = nn.Linear(hidden, task.classes)
        self.q_head = nn.Linear(hidden, 1)

    def step(self, tokens, carry=None):
        """Bir segment: (yeni carry, hücre logitleri [B, L, C], durma logiti [B])."""
        x = self.encoder(tokens)
        carry, out = self.reason(x, carry)
        return carry, self.head(out), self.q_head(out.mean(dim=1)).squeeze(-1)


class HRMReasoner(_Base):
    recurrent = True

    def __init__(self, task, hidden=128, heads=4, H_layers=1, L_layers=1, H_cycles=2, L_cycles=3,
                 grad="one_step", use_position=True):
        super().__init__(task, hidden, heads, use_position)
        self.core = TinyHrm(hidden_size=hidden, num_heads=heads, H_layers=H_layers, L_layers=L_layers,
                            H_cycles=H_cycles, L_cycles=L_cycles)
        self.grad = grad
        self.z0_H = nn.Parameter(torch.zeros(hidden))
        self.z0_L = nn.Parameter(torch.zeros(hidden))

    def reason(self, x, carry):
        if self.grad == "bptt" and carry is None:
            z_H, z_L = self.core(x)                            # kullanıcının forward'ı, olduğu gibi
            return (z_H, z_L), z_H
        if carry is None:
            z_H = self.z0_H.expand_as(x).contiguous()
            z_L = self.z0_L.expand_as(x).contiguous()
        else:
            z_H, z_L = carry
        N, T = self.core.H_cycles, self.core.L_cycles
        L, H = self.core.L_level, self.core.H_level
        if self.grad == "one_step":
            with torch.no_grad():
                for i in range(N * T - 1):
                    z_L = L(z_L, z_H + x)
                    if (i + 1) % T == 0:
                        z_H = H(z_H, z_L)
            z_L = L(z_L, z_H + x)                              # yalnızca son L ve son H adımı
            z_H = H(z_H, z_L)
        else:                                                  # bptt, taşınan durumdan devam
            for _ in range(N):
                for _ in range(T):
                    z_L = L(z_L, z_H + x)
                z_H = H(z_H, z_L)
        return (z_H, z_L), z_H


class LoopedNet(_Base):
    """Hiyerarşi yok: tek ReasoningModule, z = f(z + x), `loops` kez (son adım dışında gradyansız)."""
    recurrent = True

    def __init__(self, task, hidden=128, heads=4, layers=2, loops=8, use_position=True):
        super().__init__(task, hidden, heads, use_position)
        self.net = ReasoningModule(hidden, heads, layers)
        self.loops = loops
        self.z0 = nn.Parameter(torch.zeros(hidden))

    def reason(self, x, carry):
        z = self.z0.expand_as(x).contiguous() if carry is None else carry
        with torch.no_grad():
            for _ in range(self.loops - 1):
                z = self.net(z, x)
        z = self.net(z, x)
        return z, z


class TransformerNet(_Base):
    """Döngüsüz HRMBlock yığını (aynı blok, paylaşılmayan ağırlıklar)."""

    def __init__(self, task, hidden=128, heads=4, layers=2, use_position=True):
        super().__init__(task, hidden, heads, use_position)
        self.blocks = nn.ModuleList([HRMBlock(hidden, heads) for _ in range(layers)])

    def reason(self, x, carry):
        h = x
        for b in self.blocks:
            h = b(h)
        return None, h


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
