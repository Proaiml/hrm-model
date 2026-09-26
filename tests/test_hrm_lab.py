import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hrm_lab import models as M  # noqa: E402
from hrm_lab import tasks as T   # noqa: E402
from hrm_lab import train as TR  # noqa: E402
from hrm_model import TinyHrm    # noqa: E402


def small(spec, **kw):
    return M.HRMReasoner(spec, hidden=32, heads=2, **kw)


# ---------------------------------------------------------------- kullanıcının modeli
def test_bptt_mode_is_exactly_the_users_forward():
    torch.manual_seed(0)
    m = small(T.DIRECTION, grad="bptt")
    tokens = torch.randint(0, 4, (2, 64))
    x = m.encoder(tokens)
    z_H, z_L = TinyHrm.forward(m.core, x)
    (w_H, w_L), out = m.reason(x, None)
    assert torch.equal(z_H, w_H) and torch.equal(z_L, w_L) and torch.equal(out, z_H)


def test_one_step_gradient_only_through_the_last_L_and_H_step():
    m = small(T.DIRECTION, grad="one_step")
    calls = {"L": 0, "H": 0}

    def hook(name):
        def f(module, inputs, output):
            calls[name] += int(torch.is_grad_enabled())
        return f

    m.core.L_level.register_forward_hook(hook("L"))
    m.core.H_level.register_forward_hook(hook("H"))
    _, logits, _ = m.step(torch.randint(0, 4, (2, 64)))
    logits.sum().backward()
    assert calls == {"L": 1, "H": 1}                                   # N*T = 6 L adımından yalnızca biri
    assert m.core.L_level.layers[0].mlp.layer1.weight.grad is not None


def test_bptt_keeps_every_step_in_the_graph():
    m = small(T.DIRECTION, grad="bptt")
    calls = {"L": 0}
    m.core.L_level.register_forward_hook(lambda *_: calls.__setitem__("L", calls["L"] + int(torch.is_grad_enabled())))
    m.step(torch.randint(0, 4, (2, 64)))
    assert calls["L"] == m.core.H_cycles * m.core.L_cycles


def test_without_position_attention_cannot_tell_where_things_are():
    """Konumsuz: A hücresinin çıktısı diğer hücrelerin yerleşiminden bağımsızdır (permütasyon eşdeğerliği)."""
    torch.manual_seed(0)
    tokens = torch.zeros(1, 64, dtype=torch.long)
    tokens[0, 0], tokens[0, 7] = 2, 3                                   # B sağda
    moved = tokens.clone()
    moved[0, 7], moved[0, 56] = 0, 3                                    # B aşağıda
    for pos, same in ((False, True), (True, False)):
        m = small(T.DIRECTION, grad="bptt", use_position=pos).eval()
        with torch.no_grad():
            a = m.step(tokens)[1][0, 0]
            b = m.step(moved)[1][0, 0]
        assert torch.allclose(a, b, atol=1e-5) == same


def test_deep_supervision_detaches_state_between_segments():
    spec = T.maze_spec(3)
    data = {"train": T.make_maze(16, 3, 0, min_path=3)}
    m = M.HRMReasoner(spec, hidden=32, heads=2, grad="one_step")
    cfg = TR.TrainConfig(steps=4, batch=8, segments=2, device="cpu", warmup=1)
    info = TR.train(m, spec, data, cfg, log_every=2, log=lambda *a: None)
    assert info["history"][-1]["step"] == 4
    res, _ = TR.evaluate(m, spec, data["train"], TR.TrainConfig(eval_segments=3, device="cpu", act=True))
    assert len(res["exact_by_segment"]) == 3 and 1 <= res["mean_segments_act"] <= 3


@pytest.mark.parametrize("cls,kw", [(M.TransformerNet, dict(layers=2)), (M.LoopedNet, dict(layers=1, loops=2))])
def test_baselines_share_the_interface(cls, kw):
    m = cls(T.SUDOKU, hidden=32, heads=2, **kw)
    carry, logits, q = m.step(torch.randint(0, 10, (3, 81)))
    assert logits.shape == (3, 81, 9) and q.shape == (3,)


# ---------------------------------------------------------------- görevler
def test_maze_labels_are_valid_shortest_paths_and_survive_augmentation():
    x, y = T.make_maze(40, 5, 0)
    assert all(T.check_maze_answer(x[i], y[i], 11) for i in range(40))
    xa, ya = T.augment_batch(T.maze_spec(5), x, y, random.Random(0))
    assert all(T.check_maze_answer(xa[i], ya[i], 11) for i in range(40))


def test_maze_checker_rejects_a_broken_or_longer_path():
    x, y = T.make_maze(1, 5, 1)
    broken = y[0].copy()
    broken[np.flatnonzero(broken)[len(np.flatnonzero(broken)) // 2]] = 0
    assert not T.check_maze_answer(x[0], broken, 11)
    extra = y[0].copy()
    extra[np.flatnonzero((x[0] == 0) & (y[0] == 0))[0]] = 1
    assert not T.check_maze_answer(x[0], extra, 11)


def test_sudoku_puzzles_have_one_solution_and_augmentation_keeps_it():
    x, y = T.make_sudoku(5, 0)
    assert all(T.count_solutions(p) == 1 for p in x)
    assert all(T.check_sudoku_answer(x[i], y[i]) for i in range(5))
    xa, ya = T.augment_batch(T.SUDOKU, x, y, random.Random(3))
    assert all(T.check_sudoku_answer(xa[i], ya[i]) and T.count_solutions(xa[i]) == 1 for i in range(5))


def test_direction_answer_matches_geometry():
    x, y, a = T.make_direction(200, 0)
    for g, label, pos in zip(x, y, a):
        b = int(np.flatnonzero(g == 3)[0])
        dr, dc = np.sign(b // 8 - pos // 8), np.sign(b % 8 - pos % 8)
        assert T.DIRECTIONS[int(label)] == (dr, dc)
