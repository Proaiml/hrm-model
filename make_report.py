"""
results/ altındaki deney çıktılarından tablo ve grafikler.

    py -3.11 make_report.py

Yazar: docs/figures/*.png ve Markdown tablolarını ekrana basar (README'de kullanılanlar).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import torch                     # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
FIG = ROOT / "docs" / "figures"
LABEL = {
    "transformer2": "Transformer, 2 blok (aynı parametre)",
    "transformer8": "Transformer, 8 blok (aynı derinlik, 4x parametre)",
    "tinyhrm_bptt": "TinyHrm, kendi forward'ı (BPTT)",
    "hrm_onestep": "HRM + tek adımlı gradyan",
    "hrm_ds": "HRM + tek adım + derin denetim",
    "hrm_ds_act": "HRM + derin denetim + durma (ACT)",
    "looped_ds": "Hiyerarşisiz döngülü ağ + derin denetim",
    "tinyhrm_nopos": "TinyHrm, konumsuz (depodaki hali)",
}
ORDER = list(LABEL)
TASK_TITLE = {"maze": "Labirent 19x19, en kısa yol", "sudoku": "Sudoku 9x9", "direction": "Yön (A→B), 8x8"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def load():
    rows = defaultdict(lambda: defaultdict(list))
    for f in sorted((ROOT / "results").glob("*/*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        rows[r["task"]][r["method"]].append(r)
    return rows


def ms(values):
    v = np.array(values, dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def table(task, methods):
    lines = ["| Yöntem | Parametre | Test tam doğruluk | Hücre doğruluğu | 1 segment → en çok segment | Eğitim (s) | Tepe GPU bellek (MB) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for m in [m for m in ORDER if m in methods]:
        rs = methods[m]
        e_m, e_s = ms([r["test"]["exact"] for r in rs])
        t_m, _ = ms([r["test"]["token_acc"] for r in rs])
        first = np.mean([r["test"]["exact_by_segment"][0] for r in rs])
        segs = len(rs[0]["test"]["exact_by_segment"])
        seg_txt = f"%{100 * first:.1f} → %{100 * e_m:.1f} ({segs})" if segs > 1 else "—"
        act = ""
        if "exact_act" in rs[0]["test"]:
            a_m, _ = ms([r["test"]["exact_act"] for r in rs])
            s_m, _ = ms([r["test"]["mean_segments_act"] for r in rs])
            act = f"; ACT ile %{100 * a_m:.1f}, ort. {s_m:.1f} segment"
        pm = np.mean([r["train"]["peak_mem_mb"] for r in rs])
        tt = np.mean([r["train"]["train_seconds"] for r in rs])
        pm_txt = f"{pm:,.0f}" if not np.isnan(pm) else "—"
        lines.append(f"| {LABEL[m]} | {rs[0]['params']:,} | **%{100 * e_m:.1f}** ± {100 * e_s:.1f} | %{100 * t_m:.1f} | "
                     f"{seg_txt}{act} | {tt:.0f} | {pm_txt} |")
    n = {len(v) for v in methods.values()}
    return "\n".join(lines) + f"\n\n{max(n)} tohum ortalaması ± standart sapma; test kümesi {methods[next(iter(methods))][0]['n_test']} örnek."


def bar_figure(rows):
    tasks = [t for t in ("maze", "sudoku") if t in rows]
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.5 * len(tasks), 4.8), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        ms_ = [m for m in ORDER if m in rows[task] and m != "tinyhrm_nopos"]
        vals = [ms([r["test"]["exact"] for r in rows[task][m]]) for m in ms_]
        colors = ["#9ca3af", "#9ca3af", "#f59e0b", "#60a5fa", "#2563eb", "#1d4ed8", "#10b981"][:len(ms_)]
        y = np.arange(len(ms_))[::-1]
        ax.barh(y, [100 * v[0] for v in vals], xerr=[100 * v[1] for v in vals], color=colors, height=0.6)
        for yi, v in zip(y, vals):
            ax.text(100 * v[0] + 1.5, yi, f"%{100 * v[0]:.1f}", va="center", fontsize=9)
        ax.set_yticks(y, [LABEL[m] for m in ms_])
        ax.set_xlim(0, 110)
        ax.set_xlabel("Test tam doğruluk (%; tüm ızgara doğru)")
        ax.set_title(TASK_TITLE[task])
    fig.tight_layout()
    fig.savefig(FIG / "accuracy_by_method.png", dpi=130)
    plt.close(fig)


def segment_figure(rows):
    tasks = [t for t in ("maze", "sudoku") if t in rows]
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for m, c in (("hrm_ds", "#2563eb"), ("hrm_ds_act", "#1d4ed8"), ("looped_ds", "#10b981")):
            if m not in rows[task]:
                continue
            curves = np.array([r["test"]["exact_by_segment"] for r in rows[task][m]]) * 100
            x = np.arange(1, curves.shape[1] + 1)
            ax.plot(x, curves.mean(0), "-o", ms=3, color=c, label=LABEL[m])
            ax.fill_between(x, curves.min(0), curves.max(0), color=c, alpha=0.12)
        for m, c in (("transformer8", "#6b7280"), ("tinyhrm_bptt", "#f59e0b")):
            if m in rows[task]:
                v = 100 * np.mean([r["test"]["exact"] for r in rows[task][m]])
                ax.axhline(v, ls="--", color=c, lw=1.2, label=LABEL[m])
        ax.axvline(4, color="#d1d5db", lw=1)
        ax.text(4.2, 3, "eğitimdeki segment sayısı", fontsize=8, color="#6b7280")
        ax.set_xlabel("Çıkarımda düşünme segmenti")
        ax.set_ylabel("Test tam doğruluk (%)")
        ax.set_title(TASK_TITLE[task])
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.25)
    axes[0][-1].legend(fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "accuracy_vs_segments.png", dpi=130)
    plt.close(fig)


def position_figure(rows):
    if "direction" not in rows:
        return
    ms_ = [m for m in ("tinyhrm_nopos", "tinyhrm_bptt") if m in rows["direction"]]
    vals = [100 * np.mean([r["test"]["exact"] for r in rows["direction"][m]]) for m in ms_]
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.barh([1, 0][:len(ms_)], vals, color=["#ef4444", "#2563eb"][:len(ms_)], height=0.55)
    ax.axvline(25, ls=":", color="#6b7280")
    ax.text(26, 1.35, "rastgele tahmin %25", fontsize=8, color="#6b7280")
    for yi, v in zip([1, 0], vals):
        ax.text(v + 1.5, yi, f"%{v:.1f}", va="center")
    ax.set_yticks([1, 0][:len(ms_)], ["Konum bilgisi yok (depodaki hali)", "2B konum gömmesi eklendi"][:len(ms_)])
    ax.set_xlim(0, 112)
    ax.set_xlabel("Görülmemiş ızgaralarda doğru ilk adım (%)")
    ax.set_title("Aynı TinyHrm, aynı eğitim: yalnızca konum bilgisi farklı")
    fig.tight_layout()
    fig.savefig(FIG / "position_ablation.png", dpi=130)
    plt.close(fig)


def memory_figure(rows):
    if "maze" not in rows:
        return
    ms_ = [m for m in ("transformer2", "hrm_onestep", "hrm_ds", "tinyhrm_bptt", "transformer8") if m in rows["maze"]]
    vals = [np.mean([r["train"]["peak_mem_mb"] for r in rows["maze"][m]]) / 1024 for m in ms_]
    fig, ax = plt.subplots(figsize=(7, 3))
    y = np.arange(len(ms_))[::-1]
    ax.barh(y, vals, color=["#9ca3af", "#60a5fa", "#2563eb", "#f59e0b", "#9ca3af"][:len(ms_)], height=0.55)
    for yi, v in zip(y, vals):
        ax.text(v + 0.05, yi, f"{v:.2f} GB", va="center", fontsize=9)
    ax.set_yticks(y, [LABEL[m] for m in ms_])
    ax.set_xlabel("Eğitimde tepe GPU belleği (GB, toplu 64, labirent 361 hücre)")
    ax.set_xlim(0, max(vals) * 1.25)
    fig.tight_layout()
    fig.savefig(FIG / "memory.png", dpi=130)
    plt.close(fig)


def example_figure(rows):
    """En iyi labirent modelinin görülmemiş bir labirentteki çözümü."""
    from hrm_lab import models as M
    from hrm_lab import tasks as T
    if "maze" not in rows or "hrm_ds" not in rows["maze"]:
        return
    ck = ROOT / "checkpoints" / "maze_hrm_ds_s0.pt"
    if not ck.exists():
        return
    spec = T.maze_spec(9)
    model = M.HRMReasoner(spec, grad="one_step")
    model.load_state_dict(torch.load(ck, map_location="cpu")["state_dict"])
    model.eval()
    z = np.load(ROOT / "results" / "data_maze.npz")
    x, y = z["test0"][:3], z["test1"][:3]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    with torch.no_grad():
        tokens = torch.as_tensor(x).long()
        carry = None
        for _ in range(16):
            carry, logits, _ = model.step(tokens, carry)
        pred = logits.argmax(-1).numpy()
    for ax, xi, yi, pi in zip(axes, x, y, pred):
        g = xi.reshape(19, 19)
        img = np.ones((19, 19, 3))
        img[g == 1] = (0.15, 0.18, 0.22)
        img[pi.reshape(19, 19) == 1] = (0.15, 0.39, 0.92)
        img[(yi.reshape(19, 19) == 1) & (pi.reshape(19, 19) == 0)] = (0.94, 0.27, 0.27)
        img[g == 2] = (0.06, 0.73, 0.51)
        img[g == 3] = (0.96, 0.62, 0.04)
        ok = T.check_maze_answer(xi, pi, 19)
        ax.imshow(img, interpolation="nearest")
        ax.set_title("en kısa yol doğru" if ok else "hatalı", color="#15803d" if ok else "#b91c1c")
        ax.axis("off")
    fig.suptitle("Görülmemiş labirentler: HRM + derin denetim (mavi = modelin yolu, yeşil başlangıç, turuncu hedef, kırmızı = kaçan hücre)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "maze_examples.png", dpi=130)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    FIG.mkdir(parents=True, exist_ok=True)
    rows = load()
    for task in ("direction", "maze", "sudoku"):
        if task in rows:
            print(f"\n### {TASK_TITLE[task]}\n")
            print(table(task, rows[task]))
    bar_figure(rows)
    segment_figure(rows)
    position_figure(rows)
    memory_figure(rows)
    example_figure(rows)
    print("\nşekiller:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
