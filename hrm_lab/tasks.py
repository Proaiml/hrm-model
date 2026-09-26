"""
Izgara akıl yürütme görevleri: veri üretimi, doğrulama ve veri çoğaltma.

Her örnek (girdi, hedef) çiftidir; ikisi de düzleştirilmiş ızgara (uzunluk = satır * sütun).

- direction : 8x8 ızgarada A'dan B'ye ilk adımın yönü (hrm_model.py'deki örneğin genellenmiş hali).
              A ve B aynı satır ya da sütundadır, cevap tektir. Tek çıktı: A hücresindeki yön.
- maze      : mükemmel labirentte (tek yol) başlangıçtan hedefe en kısa yol. Çıktı: her hücre yol mu?
              (engelli bir ızgarada rota planlama; drone/robot için doğrudan karşılığı var)
- sudoku    : 9x9, tek çözümlü bulmaca. Çıktı: her hücrenin rakamı.
"""
import random
from collections import deque
from dataclasses import dataclass

import numpy as np

# ------------------------------------------------------------------ ortak


@dataclass
class TaskSpec:
    name: str
    rows: int
    cols: int
    vocab: int          # girdi belirteç sayısı
    classes: int        # hücre başına çıktı sınıfı
    readout: str        # "token0" (tek hücre) ya da "all" (her hücre)


DIRECTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}      # 0=YUKARI 1=AŞAĞI 2=SOL 3=SAĞ


# ------------------------------------------------------------------ yön (A -> B)
DIRECTION = TaskSpec("direction", 8, 8, vocab=4, classes=4, readout="a_cell")


def make_direction(n, seed):
    """0 boş, 1 engel, 2 A, 3 B. Cevap: A'dan B'ye ilk adımın yönü (aynı satır/sütun, arada engel yok)."""
    rng = random.Random(seed)
    xs, ys, a_pos = [], [], []
    while len(xs) < n:
        g = np.zeros((8, 8), dtype=np.int64)
        for _ in range(rng.randint(0, 8)):
            g[rng.randrange(8), rng.randrange(8)] = 1
        ar, ac = rng.randrange(8), rng.randrange(8)
        if rng.random() < 0.5:
            br, bc = ar, rng.randrange(8)
        else:
            br, bc = rng.randrange(8), ac
        if (ar, ac) == (br, bc):
            continue
        # A ile B arasındaki hücreler boş olsun
        r0, r1, c0, c1 = min(ar, br), max(ar, br), min(ac, bc), max(ac, bc)
        g[r0:r1 + 1, c0:c1 + 1] = 0
        g[ar, ac], g[br, bc] = 2, 3
        label = 0 if br < ar else 1 if br > ar else 2 if bc < ac else 3
        xs.append(g.ravel())
        ys.append(label)
        a_pos.append(ar * 8 + ac)
    return np.stack(xs), np.array(ys), np.array(a_pos)


# ------------------------------------------------------------------ labirent


def maze_spec(cells):
    side = 2 * cells + 1
    return TaskSpec(f"maze{side}", side, side, vocab=4, classes=2, readout="all")


def generate_maze(cells, rng):
    """Rastgele derinlik öncelikli arama ile mükemmel labirent. 1 = duvar, 0 = yol."""
    side = 2 * cells + 1
    g = np.ones((side, side), dtype=np.int64)
    stack = [(0, 0)]
    seen = {(0, 0)}
    g[1, 1] = 0
    while stack:
        r, c = stack[-1]
        nbrs = [(r + dr, c + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= r + dr < cells and 0 <= c + dc < cells and (r + dr, c + dc) not in seen]
        if not nbrs:
            stack.pop()
            continue
        nr, nc = rng.choice(nbrs)
        g[2 * r + 1 + (nr - r), 2 * c + 1 + (nc - c)] = 0
        g[2 * nr + 1, 2 * nc + 1] = 0
        seen.add((nr, nc))
        stack.append((nr, nc))
    return g


def shortest_path(grid, start, goal):
    """BFS; engel = 1. Yol hücreleri listesi (başlangıç ve hedef dahil) ya da None."""
    rows, cols = grid.shape
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for dr, dc in DIRECTIONS.values():
            nxt = (cur[0] + dr, cur[1] + dc)
            if 0 <= nxt[0] < rows and 0 <= nxt[1] < cols and grid[nxt] != 1 and nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)
    if goal not in prev:
        return None
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def make_maze(n, cells, seed, min_path=None):
    """Girdi: 0 yol, 1 duvar, 2 başlangıç, 3 hedef. Hedef: 1 = en kısa yol üstünde."""
    rng = random.Random(seed)
    min_path = min_path if min_path is not None else 2 * cells
    xs, ys = [], []
    while len(xs) < n:
        g = generate_maze(cells, rng)
        free = list(zip(*np.nonzero(g == 0)))
        s, t = rng.sample(free, 2)
        path = shortest_path(g, s, t)
        if path is None or len(path) < min_path:
            continue
        x = g.copy()
        x[s], x[t] = 2, 3
        y = np.zeros_like(g)
        for p in path:
            y[p] = 1
        xs.append(x.ravel())
        ys.append(y.ravel())
    return np.stack(xs), np.stack(ys)


def check_maze_answer(x_flat, pred_flat, side):
    """Tahmin edilen hücreler başlangıçtan hedefe kesintisiz, duvarsız ve en kısa bir yol mu?"""
    x = x_flat.reshape(side, side)
    pred = pred_flat.reshape(side, side).astype(bool)
    s = tuple(np.argwhere(x == 2)[0])
    t = tuple(np.argwhere(x == 3)[0])
    if not pred[s] or not pred[t] or (pred & (x == 1)).any():
        return False
    sub = np.where(pred, 0, 1)
    path = shortest_path(sub, s, t)
    best = shortest_path(x, s, t)
    return path is not None and len(path) == len(best) and int(pred.sum()) == len(best)


# ------------------------------------------------------------------ sudoku
SUDOKU = TaskSpec("sudoku", 9, 9, vocab=10, classes=9, readout="all")


def _candidates(board, i):
    r, c = divmod(i, 9)
    br, bc = 3 * (r // 3), 3 * (c // 3)
    used = set(board[r * 9:(r + 1) * 9]) | set(board[c::9])
    used |= {board[(br + a) * 9 + bc + b] for a in range(3) for b in range(3)}
    return [d for d in range(1, 10) if d not in used]


def count_solutions(board, limit=2):
    """Geri izlemeli çözücü; en az aday olan hücreden dallanır. limit'e ulaşınca durur."""
    board = list(board)
    empties = [i for i, v in enumerate(board) if v == 0]
    if not empties:
        return 1
    best, cands = None, None
    for i in empties:
        cs = _candidates(board, i)
        if best is None or len(cs) < len(cands):
            best, cands = i, cs
            if len(cs) <= 1:
                break
    total = 0
    for d in cands:
        board[best] = d
        total += count_solutions(board, limit - total)
        if total >= limit:
            break
    return total


def solve(board):
    board = list(board)
    empties = [i for i, v in enumerate(board) if v == 0]
    if not empties:
        return board
    best, cands = None, None
    for i in empties:
        cs = _candidates(board, i)
        if best is None or len(cs) < len(cands):
            best, cands = i, cs
    for d in cands:
        board[best] = d
        out = solve(board)
        if out:
            return out
    return None


def _random_solution(rng):
    """Adayları rastgele sırayla deneyen geri izleme: her seferinde farklı bir tam çözüm."""
    board = [0] * 81

    def fill(i):
        if i == 81:
            return True
        cands = _candidates(board, i)
        rng.shuffle(cands)
        for d in cands:
            board[i] = d
            if fill(i + 1):
                return True
        board[i] = 0
        return False

    fill(0)
    return np.array(board).reshape(9, 9)


def make_sudoku(n, seed, clues=(24, 30)):
    """Tek çözümlü bulmacalar: rastgele çözümden, tekliği bozmadan ipucu silinir."""
    rng = random.Random(seed)
    xs, ys = [], []
    while len(xs) < n:
        sol = _random_solution(rng).ravel().tolist()
        puzzle = list(sol)
        target = rng.randint(*clues)
        order = list(range(81))
        rng.shuffle(order)
        for i in order:
            if sum(v > 0 for v in puzzle) <= target:
                break
            keep = puzzle[i]
            puzzle[i] = 0
            if count_solutions(puzzle) != 1:
                puzzle[i] = keep
        xs.append(np.array(puzzle))
        ys.append(np.array(sol) - 1)             # sınıf 0..8 = rakam 1..9
    return np.stack(xs), np.stack(ys)


def check_sudoku_answer(x_flat, pred_classes):
    board = (np.asarray(pred_classes) + 1).reshape(9, 9)
    x = np.asarray(x_flat).reshape(9, 9)
    if ((x > 0) & (board != x)).any():
        return False
    ok = all(len(set(board[r])) == 9 for r in range(9)) and all(len(set(board[:, c])) == 9 for c in range(9))
    return ok and all(len(set(board[r:r + 3, c:c + 3].ravel())) == 9 for r in (0, 3, 6) for c in (0, 3, 6))


# ------------------------------------------------------------------ veri çoğaltma (eğitimde, her adımda)


def _augment_sudoku_pair(x, y, rng):
    """Geçerliliği koruyan dönüşümler: bant/yığın ve içlerindeki satır/sütun karışımı, devrik."""
    bands = rng.sample(range(3), 3)
    rows = [b * 3 + r for b in bands for r in rng.sample(range(3), 3)]
    stacks = rng.sample(range(3), 3)
    cols = [s * 3 + c for s in stacks for c in rng.sample(range(3), 3)]
    x, y = x[rows][:, cols], y[rows][:, cols]
    if rng.random() < 0.5:
        x, y = x.T, y.T
    return x.copy(), y.copy()


def augment_batch(task, xb, yb, rng):
    """numpy toplu veri çoğaltma. Sudoku: satır/sütun/bant + rakam permütasyonu. Labirent: 8 simetri."""
    if task.name == "sudoku":
        xs, ys = [], []
        for x, y in zip(xb, yb):
            perm = np.array([0] + rng.sample(range(1, 10), 9))
            x2, y2 = _augment_sudoku_pair(perm[x.reshape(9, 9)], perm[y.reshape(9, 9) + 1] - 1, rng)
            xs.append(x2.ravel())
            ys.append(y2.ravel())
        return np.stack(xs), np.stack(ys)
    if task.name.startswith("maze"):
        side = task.rows
        xs, ys = [], []
        for x, y in zip(xb, yb):
            k, flip = rng.randrange(4), rng.random() < 0.5
            x2, y2 = np.rot90(x.reshape(side, side), k), np.rot90(y.reshape(side, side), k)
            if flip:
                x2, y2 = x2[:, ::-1], y2[:, ::-1]
            xs.append(x2.ravel())
            ys.append(y2.ravel())
        return np.stack(xs), np.stack(ys)
    return xb, yb
