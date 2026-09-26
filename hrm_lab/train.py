"""
Eğitim ve değerlendirme.

Adil karşılaştırma: her yöntem aynı sayıda optimizer güncellemesi (`steps`) alır. Derin denetimde
bir mini-toplu `segments` kez güncelleme verir (her segmentten sonra durum koparılır, gradyan
segmentler arasında akmaz); döngüsüz yöntemlerde bir mini-toplu bir güncellemedir.

Durma (ACT, TRM'deki basit biçim): q_head, o segmentteki tahminin tamamen doğru olup olmadığını
ikili çapraz entropiyle öğrenir. Çıkarımda örnek, q > 0 olunca (ya da en fazla segmentte) durur.
"""
import json
import math
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from . import tasks as T


@dataclass
class TrainConfig:
    steps: int = 3000
    batch: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup: int = 200
    segments: int = 1              # eğitimde derin denetim segmenti (1 = yok)
    eval_segments: int = 1         # değerlendirmede en fazla segment
    act: bool = False              # durma başı kaybı + erken durma
    augment: bool = True
    seed: int = 0
    gpu_temp_limit: int = 75       # bu sıcaklıkta eğitim durur, 5 derece soğuyunca sürer (0 = kapalı)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    extra: dict = field(default_factory=dict)


def _readout(logits, task, a_pos=None):
    if task.readout == "a_cell":
        return logits[torch.arange(logits.shape[0]), a_pos]
    return logits


def _loss_and_exact(logits, y, task, a_pos=None):
    out = _readout(logits, task, a_pos)
    if task.readout == "a_cell":
        return F.cross_entropy(out, y), (out.argmax(-1) == y)
    loss = F.cross_entropy(out.reshape(-1, task.classes), y.reshape(-1))
    return loss, (out.argmax(-1) == y).all(dim=1)


def gpu_temperature():
    """nvidia-smi ile GPU sıcaklığı (°C); okunamazsa None."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return int(out.split()[0])
    except Exception:
        return None


def thermal_guard(limit, log=print):
    """GPU limit derecedeyse soğuyana (limit-5) kadar bekler; beklenen süreyi döndürür."""
    temp = gpu_temperature()
    if not limit or temp is None or temp < limit:
        return 0.0
    t0 = time.time()
    log(f"  GPU {temp}°C: soğuması bekleniyor (sınır {limit}°C)")
    while temp is not None and temp > limit - 5:
        time.sleep(5)
        temp = gpu_temperature()
    return time.time() - t0


def train(model, task, data, cfg: TrainConfig, log_every=250, log=print):
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    dev = cfg.device
    model.to(dev)
    xtr, ytr = data["train"][:2]
    atr = data["train"][2] if len(data["train"]) > 2 else None
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / cfg.steps))))
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history, step, t0, cooled = [], 0, time.time(), 0.0
    model.train()
    while step < cfg.steps:
        if dev == "cuda" and step % 25 == 0:
            cooled += thermal_guard(cfg.gpu_temp_limit, log)
        idx = np.array(rng.sample(range(len(xtr)), min(cfg.batch, len(xtr))))
        xb, yb = xtr[idx], ytr[idx]
        if cfg.augment:
            xb, yb = T.augment_batch(task, xb, yb, rng)
        x = torch.as_tensor(xb, device=dev).long()
        y = torch.as_tensor(yb, device=dev).long()
        a = torch.as_tensor(atr[idx], device=dev).long() if atr is not None else None
        carry = None
        for _ in range(cfg.segments):
            carry, logits, q = model.step(x, carry)
            loss, exact = _loss_and_exact(logits, y, task, a)
            if cfg.act:
                loss = loss + 0.5 * F.binary_cross_entropy_with_logits(q, exact.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if carry is not None:
                carry = tuple(c.detach() for c in carry) if isinstance(carry, tuple) else carry.detach()
            if step % log_every == 0 or step == cfg.steps:
                history.append({"step": step, "loss": round(loss.item(), 4),
                                "train_exact": round(exact.float().mean().item(), 3),
                                "seconds": round(time.time() - t0 - cooled, 1)})
                log(f"  adım {step:5d}  kayıp {loss.item():.4f}  eğitim tam doğru {exact.float().mean().item():.3f}")
            if step >= cfg.steps:
                break
    peak = torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else float("nan")
    return {"history": history, "train_seconds": round(time.time() - t0 - cooled, 1), "peak_mem_mb": round(peak, 1),
            "cooling_wait_seconds": round(cooled, 1)}


@torch.no_grad()
def evaluate(model, task, split, cfg: TrainConfig, batch=250):
    """Segment başına tam doğruluk eğrisi + (ACT varsa) erken durmalı doğruluk ve ortalama segment."""
    model.eval()
    dev = cfg.device
    xs, ys = split[:2]
    a_all = split[2] if len(split) > 2 else None
    segs = cfg.eval_segments
    exact_by_seg = np.zeros(segs)
    token_acc = 0.0
    act_correct, act_steps = 0, 0
    preds = []
    for i in range(0, len(xs), batch):
        x = torch.as_tensor(xs[i:i + batch], device=dev).long()
        y = torch.as_tensor(ys[i:i + batch], device=dev).long()
        a = torch.as_tensor(a_all[i:i + batch], device=dev).long() if a_all is not None else None
        carry = None
        halted = torch.zeros(len(x), dtype=torch.bool, device=dev)
        act_exact = torch.zeros(len(x), dtype=torch.bool, device=dev)
        used = torch.full((len(x),), segs, device=dev)
        for s in range(segs):
            carry, logits, q = model.step(x, carry)
            _, exact = _loss_and_exact(logits, y, task, a)
            exact_by_seg[s] += exact.sum().item()
            stop = (~halted) & ((q > 0) | torch.tensor(s == segs - 1, device=dev))
            act_exact |= stop & exact
            used = torch.where(stop, torch.full_like(used, s + 1), used)
            halted |= stop
            if not model.recurrent:
                exact_by_seg[s + 1:] += exact.sum().item()
                break
        out = _readout(logits, task, a).argmax(-1)
        token_acc += (out == y).float().mean(dim=-1).sum().item() if out.dim() > 1 else (out == y).float().sum().item()
        act_correct += act_exact.sum().item()
        act_steps += used.sum().item()
        preds.append(out.cpu().numpy())
    n = len(xs)
    res = {"exact_by_segment": [round(v / n, 4) for v in exact_by_seg],
           "exact": round(exact_by_seg[-1] / n, 4), "token_acc": round(token_acc / n, 4)}
    if cfg.act:
        res.update({"exact_act": round(act_correct / n, 4), "mean_segments_act": round(act_steps / n, 2)})
    return res, np.concatenate(preds)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else str(o))
