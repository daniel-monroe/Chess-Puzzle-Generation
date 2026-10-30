"""Self-check for the BT4 port and the diffusion adapter (run by setup_b300.sh).

1. Loads the lc0 BT4 net and compares against reference outputs recorded from the lc0 engine
   (v0.33, cuda fp32 backend, policy temperature 1, history-fill fen_only): best policy move
   and Q = W - L of the WDL head.
2. Checks the diffusion adapter: its lc0 input planes equal the reference encoder, and with
   the board known its move distribution restricted to legal moves equals BT4's policy.

    python src/BT4/check_bt4.py --weights path/to/BT4-1024x15x32h-swa-6147500.pb.gz
"""
import argparse
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])

import chess
import numpy as np
import torch

from BT4.bt4 import BT4, load_bt4_weights
from BT4.lc0_encoding import fen_to_planes, policy_logits_for_legal_moves

# (fen, lc0 policy argmax, lc0 Q)
LC0_REFERENCE = [
    ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "f1b5", 0.0287),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "f3g5", 0.0487),
    ("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", "d2d4", 0.4514),
    ("rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3", "d7d5", 0.1717),
    ("8/8/8/8/8/2k5/1p6/3K4 b - - 0 1", "b2b1q", 0.9998),
    ("r3k2r/ppp2ppp/2n1bn2/2bpp3/4P3/2NP1N2/PPP1BPPP/R1BQK2R b KQkq - 3 8", "d5d4", -0.9983),
    ("r2q1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R2Q1RK1 b - - 15 12", "g7g6", -0.0046),
    ("5rk1/1b3p1p/pp3p2/3n1N2/1P6/P1qB1PP1/3Q3P/4R1K1 w - - 0 1", "d2h6", 0.8982),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(args.device)

    # ---- 1. plain BT4 vs lc0 ----
    m = BT4()
    load_bt4_weights(m.body, args.weights, m.policy, m.value)
    m.to(dev).eval()
    fens = [f for f, _, _ in LC0_REFERENCE]
    planes = torch.tensor(np.stack([fen_to_planes(f) for f in fens]), device=dev)
    scores, promo, wdl = m(planes)
    ok = True
    for i, (fen, best, q) in enumerate(LC0_REFERENCE):
        lg = policy_logits_for_legal_moves(scores[i].cpu(), promo[i].cpu(), chess.Board(fen))
        ours = max(lg, key=lg.get)
        w = wdl[i].softmax(0)
        oq = (w[0] - w[2]).item()
        good = ours == best and abs(oq - q) < 0.01
        ok &= good
        print(f"  {'ok ' if good else 'BAD'} best {ours} (lc0 {best})  Q {oq:+.4f} (lc0 {q:+.4f})")
    if not ok:
        sys.exit("BT4 port does not match lc0 reference outputs")

    # ---- 2. diffusion adapter ----
    from Config import Config
    from BT4.diffusion_model import BT4Diffusion
    from tokenization.tokenization import tokenize_fen, tokenize_move, FENTokens as T
    cfg = Config(predict_moves=True, model_type="bt4", bt4_weights=args.weights)
    dm = BT4Diffusion(cfg, load_weights=False)
    dm.body, dm.policy = m.body, m.policy          # same weights, avoid a second load
    dm.to(dev).eval()
    # At init the square -> extra-token attention gate is zero, so the network is exactly BT4.
    flows = {}
    hook = dm.body.register_forward_pre_hook(lambda mod, a: flows.setdefault("x", a[0]))
    toks = torch.tensor([tokenize_fen(f) + [cfg.mask_token] * 5 for f in fens], device=dev)
    zeros_t, zeros_r = torch.zeros(len(fens), 66, device=dev), torch.zeros(len(fens), device=dev)
    dm(toks, zeros_t, zeros_r)
    hook.remove()
    ref = planes.permute(0, 2, 3, 1).reshape(len(fens), 64, 112)
    diff = (flows["x"] - ref).abs().max().item()
    print(f"  adapter planes vs lc0 encoder: max diff {diff}")
    if diff > 0:
        sys.exit("adapter input planes differ from lc0 encoding")

    worst = 0.0
    for fen in fens[:6]:
        b = chess.Board(fen)
        i = fens.index(fen)
        lg = policy_logits_for_legal_moves(scores[i].cpu(), promo[i].cpu(), b)
        refp = torch.tensor(list(lg.values())).softmax(0)
        full = torch.tensor([tokenize_fen(fen) + tokenize_move(mv) for mv in lg], device=dev)
        t = full.clone()
        t[:, 76:] = cfg.mask_token
        logp = torch.zeros(len(full), device=dev)
        n = len(full)
        for pos in range(76, 81):
            l = dm(t, torch.zeros(n, 66, device=dev), torch.zeros(n, device=dev)).log_softmax(-1)
            logp += l[torch.arange(n), pos, full[:, pos]]
            t[:, pos] = full[:, pos]
        joint = logp.exp().cpu()
        tv = 0.5 * (joint / joint.sum() - refp).abs().sum().item()
        worst = max(worst, tv)
    print(f"  adapter move distribution vs BT4 policy (legal moves): max total variation {worst:.2e}")
    if worst > 1e-3:
        sys.exit("adapter move head does not reproduce BT4 policy")

    # ---- 3. the non-square tokens are attended to: changing one changes the square outputs ----
    dm.extra_attn_gate.fill_(0.1)              # as after some training: squares now use the extra tokens
    t1 = toks[:1].clone()
    t2 = t1.clone()
    t2[0, 71:73] = torch.tensor([int(T.counter_4), int(T.counter_2)], device=dev)   # halfmove clock
    t1[0, 76:81] = cfg.mask_token
    t2[0, 76:81] = cfg.mask_token
    z = torch.zeros(1, 66, device=dev), torch.zeros(1, device=dev)
    d_sq = (dm(t1, *z)[0, :64] - dm(t2, *z)[0, :64]).abs().max().item()
    t3 = t1.clone()
    t3[0, 76] = int(T.fileb)                                     # reveal a move token
    d_mv = (dm(t1, *z)[0, :64] - dm(t3, *z)[0, :64]).abs().max().item()
    print(f"  square logits change when a clock token changes: {d_sq:.2e}, when a move token is revealed: {d_mv:.2e}")
    if not (d_sq > 0 and d_mv > 0):
        sys.exit("non-square tokens do not influence the square tokens")
    print("BT4 check passed")


if __name__ == "__main__":
    main()
