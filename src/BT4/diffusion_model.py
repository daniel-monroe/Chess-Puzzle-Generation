"""BT4 adapted to the masked-diffusion puzzle generator (same inputs/outputs as MaskedDiffusion).

Interface (identical to MaskedDiffusion):
    forward(tokens (B, L), theme_tokens (B, 66), ratings (B,)) -> logits (B, L, n_tokens)
with L = 76 FEN tokens (+ 5 best-move tokens if config.predict_moves), any of which may be
config.mask_token.

How the puzzle tokens are mapped onto BT4 (kept as close to lc0 BT4 as possible):

  Frame.  lc0 sees the board from the side to move. If the side-to-move token is known and
  is black, the board is flipped vertically and colours are swapped ("us" = black), exactly
  as lc0 does. If the side-to-move token is still masked, the board is shown as-is (white
  frame). Outputs are mapped back to absolute FEN tokens.

  Sequence.  All L tokens are positions in BT4's attention: the 64 square tokens followed by
  the 12 FEN metadata tokens and the 5 best-move tokens (81 total), so every token can attend
  to every other token in all 15 layers.
    * Square tokens: the BT4 embedding of lc0's classical 112 input planes (12 piece planes,
      history slots filled like lc0's `--history-fill=fen_only`, castling, side to move,
      rule50, constant planes) built from whatever tokens are known, plus zero-initialised
      projections for things lc0 cannot represent (square masked, partial en-passant /
      best-move file+rank matches, theme and rating conditioning).
    * Metadata / move tokens: a new learned embedding of (position, token) incl. the mask
      state, plus the theme and rating conditioning.
  Attention: square queries attend to the 64 squares exactly as in BT4, with smolgen (the
  Chessformer paper's GAB) generating the 64x64 bias from the board, and to the 17 extra tokens
  through a second softmax added with a learned per-layer, per-head gate (zero at init, so
  the network is exactly BT4 at step 0). Extra-token queries attend to all 81 tokens. Every
  pair involving an extra token has a learned per-layer, per-head bias. All attention uses
  BT4's pretrained q/k/v projections.

  Outputs: every position gets logits over the full vocabulary (no hard restriction), exactly
  like MaskedDiffusion's classifier, so the ELBO is directly comparable.
    * squares: linear head on the square token; metadata: linear head on its own token.
    * best-move tokens: linear head on the move token (zero-init) plus the log-marginals of
      BT4's attention policy, restricted to moves consistent with everything known (piece
      on the from-square, blockers, castling rights, en passant, revealed move tokens);
      castling follows lc0 (king -> rook in the policy, e1g1 in UCI).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenization.tokenization import FENTokens as T
from .bt4 import BT4Body, BT4Config, AttentionPolicyHead, load_bt4_weights, lc0_style_init_
from .policy_map import POLICY_MAP

N_SQ_FEAT = 7          # mask, ep file, ep rank, move from file/rank, move to file/rank
META_POS = list(range(64, 76))
PROMO_TOKENS = [int(T.none), int(T.promote_q), int(T.promote_r), int(T.promote_b), int(T.promote_n)]


def _sq(file, rank):
    return rank * 8 + file


def _between(f, t):
    """Squares strictly between f and t on a rank/file/diagonal (empty otherwise)."""
    ff, fr, tf, tr = f % 8, f // 8, t % 8, t // 8
    df, dr = tf - ff, tr - fr
    if not (df == 0 or dr == 0 or abs(df) == abs(dr)):
        return []
    n = max(abs(df), abs(dr))
    sf, sr = (df > 0) - (df < 0), (dr > 0) - (dr < 0)
    return [_sq(ff + sf * i, fr + sr * i) for i in range(1, n)]


def _build_candidates():
    """Static move candidates in the 'us moves up the board' direction.

    Returns list of dicts with from/to squares, promo index (0 none, 1 q, 2 r, 3 b, 4 n),
    lc0 raw policy index for the base logit, and geometry flags.
    """
    cands = []
    for raw in POLICY_MAP:
        if raw < 4096:
            f, t = divmod(raw, 64)
            cands.append((f, t, 0))
            if 48 <= f < 56 and t >= 56 and abs(f % 8 - t % 8) <= 1:
                cands.append((f, t, 4))        # knight promotion = plain from->to logit in lc0
        else:
            j = raw - 4096
            ff, rem = divmod(j, 24)
            tf, pc = divmod(rem, 3)
            cands.append((48 + ff, 56 + tf, pc + 1))
    out = []
    for f, t, p in cands:
        ff, fr, tf, tr = f % 8, f // 8, t % 8, t // 8
        df, dr = tf - ff, tr - fr
        knight = (abs(df), abs(dr)) in ((1, 2), (2, 1))
        orth = (df == 0) != (dr == 0)
        diag = abs(df) == abs(dr) and df != 0
        king_step = max(abs(df), abs(dr)) == 1
        castle = f == 4 and t in (6, 2)
        push1 = df == 0 and dr == 1 and fr >= 1
        push2 = df == 0 and dr == 2 and fr == 1
        pcap = abs(df) == 1 and dr == 1 and fr >= 1
        last = tr == 7
        pawn_ok = (push1 or push2 or pcap) and ((p != 0) if last else (p == 0))
        geo = {
            0: True,                                    # unknown piece: anything
            1: pawn_ok,
            2: knight and p == 0,
            3: diag and p == 0,
            4: orth and p == 0,
            5: (diag or orth) and p == 0,
            6: (king_step or castle) and p == 0,
        }
        out.append(dict(f=f, t=t, p=p, geo=geo, push=push1 or push2, pcap=pcap, castle=castle,
                        between=_between(f, t) if not knight else []))
    return out


class BT4Diffusion(nn.Module):
    def __init__(self, config, load_weights: bool = True):
        super().__init__()
        self.config = config
        self.predict_moves = config.predict_moves
        self.seq_length = config.fen_length + (config.move_length if config.predict_moves else 0)
        self.V = config.n_tokens
        self.mask_id = config.mask_token
        self.history = getattr(config, "bt4_history", "repeat")
        cfg = BT4Config()
        self.bt4_cfg = cfg
        E = cfg.embedding_size

        self.body = BT4Body(cfg)
        self.policy = AttentionPolicyHead(cfg) if self.predict_moves else None

        # ---- square-token adapters (zero-initialised) ----
        self.n_extra = K = self.seq_length - 64
        self.cond_dim = (config.n_themes + config.rating_dim) if config.use_context else 0
        self.sq_feat_proj = nn.Linear(N_SQ_FEAT, E, bias=False)
        self.preproc_mask_proj = nn.Linear(64, 64 * cfg.embedding_dense_sz, bias=False)
        self.cond_proj = nn.Linear(self.cond_dim, E, bias=False) if self.cond_dim else None

        # ---- metadata / move tokens as attention positions ----
        self.extra_embed = nn.Embedding(K * (self.V + 1), E)       # (position, token incl. mask)
        self.extra_cond = nn.Linear(self.cond_dim, E, bias=False) if self.cond_dim else None
        self.extra_ln = nn.LayerNorm(E, eps=1e-3)
        n_tok = 64 + K
        self.extra_attn_bias = nn.Parameter(torch.zeros(cfg.encoder_layers, cfg.encoder_heads, n_tok, n_tok))
        self.extra_attn_gate = nn.Parameter(torch.zeros(cfg.encoder_layers, cfg.encoder_heads))

        # ---- output heads: full vocabulary at every position ----
        self.board_head = nn.Linear(E, self.V)
        self.meta_head = nn.Linear(E, self.V)
        self.move_head = nn.Linear(E, self.V) if self.predict_moves else None

        self._init_weights(getattr(config, "bt4_init", "lc0"), getattr(config, "bt4_weights", None), load_weights)
        self._register_tables()

    # ------------------------------------------------------------------ init
    def _init_weights(self, init, weights_path, load_weights):
        heads = [self.policy] if self.policy is not None else []
        lc0_style_init_(self.body, *heads)
        if init == "lc0" and load_weights:
            if not weights_path:
                raise ValueError("config.bt4_weights must point to the lc0 BT4 .pb.gz for bt4_init='lc0'")
            load_bt4_weights(self.body, weights_path, self.policy, None)
        for m in (self.sq_feat_proj, self.preproc_mask_proj, self.cond_proj):
            if m is not None:
                nn.init.zeros_(m.weight)
        # "Square is masked" must be visible from step 0: in lc0 planes a masked square looks
        # empty, so the mask-flag inputs get random weights at the scale of the piece-plane
        # inputs. BT4 never saw masked squares, so fully known boards still give exactly BT4.
        with torch.no_grad():
            self.sq_feat_proj.weight[:, 0].normal_(0, self.body.embedding.weight[:, :12].std().item())
            self.preproc_mask_proj.weight.normal_(0, self.body.emb_preproc.weight.std().item())
        if self.extra_cond is not None:
            nn.init.normal_(self.extra_cond.weight, std=0.02)
        nn.init.normal_(self.extra_embed.weight, std=1.0)
        with torch.no_grad():
            self.extra_attn_bias.zero_()
            self.extra_attn_gate.zero_()        # squares start ignoring the extra tokens -> exactly BT4
        for head in (self.board_head, self.meta_head):
            nn.init.normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)
        if self.move_head is not None:                 # starts as exactly the BT4-policy marginals
            nn.init.zeros_(self.move_head.weight)
            nn.init.zeros_(self.move_head.bias)

    def pretrained_parameter_names(self):
        """Names of parameters that come from the lc0 checkpoint (for per-group learning rates)."""
        names = {f"body.{n}" for n, _ in self.body.named_parameters()}
        if self.policy is not None:
            names |= {f"policy.{n}" for n, _ in self.policy.named_parameters()}
        return names

    # ---------------------------------------------------------------- tables
    def _register_tables(self):
        V = self.V
        sq = torch.arange(64)
        self.register_buffer("fen_idx_white", sq ^ 56, persistent=False)   # canonical square -> FEN index
        self.register_buffer("fen_idx_black", sq.clone(), persistent=False)
        self.register_buffer("sq_file", sq % 8, persistent=False)
        self.register_buffer("sq_rank", sq // 8, persistent=False)
        # colour swap of the 13 board classes
        swap = torch.tensor([0, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6])
        self.register_buffer("class_swap", swap, persistent=False)
        # vocabulary permutation for metadata tokens under colour flip (incl. mask id at index V)
        P = torch.arange(V + 1)
        pairs = [(T.side_white, T.side_black), (T.castle_white_king, T.castle_black_king),
                 (T.castle_white_queen, T.castle_black_queen)]
        pairs += [(T.rank1 + i, T.rank8 - i) for i in range(4)]
        pairs += [(T.white_pawn + i, T.black_pawn + i) for i in range(6)]       # piece colours
        for a, b in pairs:
            P[int(a)], P[int(b)] = int(b), int(a)
        self.register_buffer("vocab_flip", P, persistent=False)
        pos = torch.arange(self.seq_length - 64)
        pos[1], pos[3] = pos[3].clone(), pos[1].clone()   # 65 <-> 67 (K <-> k)
        pos[2], pos[4] = pos[4].clone(), pos[2].clone()   # 66 <-> 68 (Q <-> q)
        self.register_buffer("meta_pos_flip", pos, persistent=False)
        n_tok = self.seq_length
        bmask = torch.ones(n_tok, n_tok)
        bmask[:64, :64] = 0                     # the square block comes from smolgen
        self.register_buffer("extra_bias_mask", bmask, persistent=False)

        if not self.predict_moves:
            return
        cands = _build_candidates()
        E = len(cands)
        f = torch.tensor([c["f"] for c in cands])
        t = torch.tensor([c["t"] for c in cands])
        p = torch.tensor([c["p"] for c in cands])
        geo = torch.tensor([[c["geo"][k] for c in cands] for k in range(7)])
        between = torch.zeros(E, 64)
        for i, c in enumerate(cands):
            between[i, c["between"]] = 1
        push = torch.tensor([c["push"] for c in cands])
        pcap = torch.tensor([c["pcap"] for c in cands])
        castle = torch.tensor([c["castle"] for c in cands])
        # castling uses lc0's king->rook logit: e1g1 <- e1h1, e1c1 <- e1a1
        alt = torch.full((E,), -1)
        alt[castle & (t == 6)] = 4 * 64 + 7
        alt[castle & (t == 2)] = 4 * 64 + 0
        # second direction ("them" moves down the board): vertical mirror of every candidate
        def mirror(idx):
            return torch.where(idx >= 0, ((idx // 64) ^ 56) * 64 + ((idx % 64) ^ 56), idx)
        self.n_cand = 2 * E
        self.register_buffer("c_from", torch.cat([f, f ^ 56]), persistent=False)
        self.register_buffer("c_to", torch.cat([t, t ^ 56]), persistent=False)
        self.register_buffer("c_promo", torch.cat([p, p]), persistent=False)
        self.register_buffer("c_dir", torch.cat([torch.zeros(E, dtype=torch.long), torch.ones(E, dtype=torch.long)]), persistent=False)
        self.register_buffer("c_geo", torch.cat([geo, geo], dim=1), persistent=False)          # (7, 2E)
        flip_cols = (torch.arange(64) ^ 56)
        self.register_buffer("c_between", torch.cat([between, between[:, flip_cols]]), persistent=False)
        self.register_buffer("c_push", torch.cat([push, push]), persistent=False)
        self.register_buffer("c_pcap", torch.cat([pcap, pcap]), persistent=False)
        self.register_buffer("c_castle", torch.cat([castle, castle]), persistent=False)
        self.register_buffer("c_kingside", torch.cat([t == 6, t == 6]), persistent=False)
        self.register_buffer("c_raw", torch.cat([f * 64 + t, mirror(f * 64 + t)]), persistent=False)
        self.register_buffer("c_alt", torch.cat([alt, mirror(alt)]), persistent=False)
        self.register_buffer("promo_tokens", torch.tensor(PROMO_TOKENS), persistent=False)

    # --------------------------------------------------------------- forward
    def forward(self, tokens, theme_tokens=None, ratings=None, checkpoint_activations=False):
        B, L = tokens.shape
        m = self.mask_id
        tokens = tokens.long()
        flip = tokens[:, 64] == T.side_black                                  # (B,) black to move & known
        flip_c = flip.unsqueeze(1)

        # ---------- board in the side-to-move frame ----------
        fen_idx = torch.where(flip_c, self.fen_idx_black, self.fen_idx_white)   # (B,64)
        board = tokens.gather(1, fen_idx)
        sq_masked = board == m
        cls = torch.where(sq_masked, torch.zeros_like(board), board.clamp(0, 12))
        cls = torch.where(flip_c, self.class_swap[cls], cls)                 # 0 empty, 1-6 us, 7-12 them
        canon = torch.where(sq_masked, torch.full_like(cls, 13), cls)         # 13 = unknown
        cur = F.one_hot(cls, 13)[..., 1:].float()                            # (B,64,12)

        # ---------- metadata ----------
        ep_f_tok, ep_r_tok = tokens[:, 69], tokens[:, 70]
        ep_f_known = (ep_f_tok >= T.filea) & (ep_f_tok <= T.fileh)
        ep_r_known = (ep_r_tok >= T.rank1) & (ep_r_tok <= T.rank8)
        ep_file = (ep_f_tok - T.filea).clamp(0, 7)
        ep_rank = torch.where(flip, 7 - (ep_r_tok - T.rank1).clamp(0, 7), (ep_r_tok - T.rank1).clamp(0, 7))
        ep_sq_known = ep_f_known & ep_r_known
        ep_sq = ep_rank * 8 + ep_file                                        # canonical square
        ep_unknown = (ep_f_tok == m) | (ep_r_tok == m)

        wK, wQ = tokens[:, 65] == T.castle_white_king, tokens[:, 66] == T.castle_white_queen
        bK, bQ = tokens[:, 67] == T.castle_black_king, tokens[:, 68] == T.castle_black_queen
        us_oo, us_ooo = torch.where(flip, bK, wK), torch.where(flip, bQ, wQ)
        them_oo, them_ooo = torch.where(flip, wK, bK), torch.where(flip, wQ, bQ)
        # castling right known to be absent (canonical us/them, king/queen side)
        no_castle = tokens[:, 65:69] == T.no_castle
        no_c = torch.where(flip_c, no_castle[:, [2, 3, 0, 1]], no_castle)     # (B,4): us K, us Q, them K, them Q

        def digit(tok):
            return torch.where((tok >= T.counter_0) & (tok <= T.counter_9), tok - T.counter_0, torch.zeros_like(tok))
        half_known = (tokens[:, 71] != m) & (tokens[:, 72] != m)
        rule50 = torch.where(half_known, digit(tokens[:, 71]) * 10 + digit(tokens[:, 72]), torch.zeros_like(ep_file)).float()

        # ---------- lc0 classical 112 planes ----------
        if self.history == "repeat":
            hist = cur
            # undo the opponent's double pawn push in the history slots (lc0 fen_only fill)
            pawn_sq = (ep_sq - 8).clamp(0, 63)
            orig_sq = (ep_sq + 8).clamp(0, 63)
            has_pawn = cur[torch.arange(B, device=tokens.device), pawn_sq, 6] > 0
            do = (ep_sq_known & has_pawn & (ep_rank == 5)).float().unsqueeze(1)
            delta = (F.one_hot(orig_sq, 64) - F.one_hot(pawn_sq, 64)).float() * do
            hist = torch.cat([cur[..., :6], cur[..., 6:7] + delta.unsqueeze(-1), cur[..., 7:]], dim=-1)
            slot_rest = torch.cat([hist, torch.zeros_like(hist[..., :1])], dim=-1)    # 13 planes
            history_planes = slot_rest.repeat(1, 1, 7)
        else:
            history_planes = cur.new_zeros(B, 64, 91)
        ones = cur.new_ones(B, 64, 1)
        bc = lambda v: v.float().view(B, 1, 1).expand(B, 64, 1)
        flow = torch.cat([cur, cur.new_zeros(B, 64, 1), history_planes,
                          bc(us_ooo), bc(us_oo), bc(them_ooo), bc(them_oo),
                          bc(flip), bc(rule50), cur.new_zeros(B, 64, 1), ones], dim=-1)   # (B,64,112)

        # ---------- adapter features ----------
        f_file, f_rank = self.sq_file.unsqueeze(0), self.sq_rank.unsqueeze(0)
        feats = [sq_masked.float(),
                 (ep_f_known.unsqueeze(1) & (f_file == ep_file.unsqueeze(1))).float(),
                 (ep_r_known.unsqueeze(1) & (f_rank == ep_rank.unsqueeze(1))).float()]
        if self.predict_moves:
            for i, kind in enumerate(["file", "rank", "file", "rank"]):
                tok = tokens[:, 76 + i]
                if kind == "file":
                    known = (tok >= T.filea) & (tok <= T.fileh)
                    val, grid = (tok - T.filea).clamp(0, 7), f_file
                else:
                    known = (tok >= T.rank1) & (tok <= T.rank8)
                    r = (tok - T.rank1).clamp(0, 7)
                    val, grid = torch.where(flip, 7 - r, r), f_rank
                feats.append((known.unsqueeze(1) & (grid == val.unsqueeze(1))).float())
        else:
            feats += [torch.zeros_like(feats[0])] * 4
        sq_feat = torch.stack(feats, dim=-1)                                 # (B,64,7)

        # ---------- conditioning ----------
        cond = None
        if self.config.use_context:
            cond = torch.cat([theme_tokens.float(), ratings.float().view(B, -1)], dim=1)

        # ---------- metadata / move tokens (canonical frame) as sequence positions ----------
        K = self.n_extra
        meta = tokens[:, 64:]
        meta = torch.where(flip_c, self.vocab_flip[meta][:, self.meta_pos_flip], meta)
        pos_off = torch.arange(K, device=tokens.device) * (self.V + 1)
        extra = self.extra_embed(meta.clamp(0, self.V) + pos_off)                    # (B,K,E)
        if cond is not None:
            extra = extra + self.extra_cond(cond).unsqueeze(1)
        extra = self.extra_ln(extra)

        emb_extra = self.sq_feat_proj(sq_feat)
        if cond is not None:
            emb_extra = emb_extra + self.cond_proj(cond).unsqueeze(1)
        preproc_extra = self.preproc_mask_proj(sq_masked.float())
        bias = self.extra_attn_bias * self.extra_bias_mask
        h = self.body(flow, preproc_extra, emb_extra, checkpoint_activations,
                      extra_tokens=extra, extra_bias=bias, extra_gate=self.extra_attn_gate)                           # (B,64+K,E)
        x, xe = h[:, :64], h[:, 64:]

        # ---------- board squares: full vocabulary, canonical -> absolute ----------
        bl = self.board_head(x).float()                                               # (B,64,V)
        bl = torch.where(flip.view(B, 1, 1), bl[..., self.vocab_flip[:self.V]], bl)
        bl = bl.gather(1, fen_idx.unsqueeze(-1).expand(B, 64, self.V))                # fen_idx is an involution

        # ---------- metadata / move tokens: full vocabulary ----------
        n_meta = 12
        el = [self.meta_head(xe[:, :n_meta])]
        if self.predict_moves:
            el.append(self.move_head(xe[:, n_meta:]))
        el = torch.cat(el, dim=1).float()                                             # (B,K,V) canonical
        el = torch.where(flip.view(B, 1, 1), el[:, self.meta_pos_flip][..., self.vocab_flip[:self.V]], el)
        if self.predict_moves:
            mv = self._move_logits(x, tokens, canon, flip, ep_sq, ep_sq_known, ep_unknown, no_c)
            el = torch.cat([el[:, :n_meta], el[:, n_meta:] + mv], dim=1)
        return torch.cat([bl, el], dim=1)

    # ------------------------------------------------------------ move head
    def move_distribution(self, x, tokens, canon, flip, ep_sq, ep_sq_known, ep_unknown, no_c):
        """Log-probabilities over move candidates (B, n_cand) and their absolute tokens (B, n_cand, 5)."""
        B = x.size(0)
        m = self.mask_id
        scores, promo_us, keys = self.policy.raw_scores(x)
        scores = scores.float().flatten(1)                                   # (B, 4096)
        promo_them = self.policy.ppo(keys[:, 0:8, :]).float()
        promo_them = promo_them[..., :3] + promo_them[..., 3:4]
        promo_us = promo_us.float()

        cf, ct, cp, cd = self.c_from, self.c_to, self.c_promo, self.c_dir
        base = scores[:, self.c_raw]                                         # (B, N)
        to_file = (ct % 8).unsqueeze(0).expand(B, -1)
        qrb = (cp >= 1) & (cp <= 3)
        pidx = (cp - 1).clamp(0, 2).unsqueeze(0).expand(B, -1)
        po_us = promo_us.gather(1, to_file.unsqueeze(-1).expand(-1, -1, 3)).gather(2, pidx.unsqueeze(-1)).squeeze(-1)
        po_th = promo_them.gather(1, to_file.unsqueeze(-1).expand(-1, -1, 3)).gather(2, pidx.unsqueeze(-1)).squeeze(-1)
        score = base + torch.where(qrb, torch.where(cd == 0, po_us, po_th), torch.zeros_like(base))

        # --- pseudo-legality under partial knowledge ---
        fp = canon[:, cf]                                                    # piece on from-square (0..13)
        tp = canon[:, ct]
        dir_off = (cd * 6).unsqueeze(0)
        own_f = (fp >= 1 + dir_off) & (fp <= 6 + dir_off)
        unknown_f = fp == 13
        ptype = torch.where(own_f, fp - dir_off, torch.zeros_like(fp))      # 1..6, 0 unknown
        geo = self.c_geo[ptype, torch.arange(self.n_cand, device=x.device).unsqueeze(0)]
        ok = (own_f | unknown_f) & geo
        own_t = (tp >= 1 + dir_off) & (tp <= 6 + dir_off)
        opp_t = (tp >= 7 - dir_off) & (tp <= 12 - dir_off)
        ok &= ~own_t
        occ = ((canon >= 1) & (canon <= 12)).float()
        blocked = (occ @ self.c_between.t()) > 0
        ok &= ~blocked
        is_pawn = ptype == 1
        empty_t = tp == 0
        ep_ok = (ep_sq_known.unsqueeze(1) & (ct.unsqueeze(0) == ep_sq.unsqueeze(1))) | ep_unknown.unsqueeze(1)
        pawn_ok = torch.where(self.c_push.unsqueeze(0), ~(own_t | opp_t),
                              opp_t | (tp == 13) | (empty_t & ep_ok))
        ok &= ~is_pawn | pawn_ok
        is_king = ptype == 6
        right_gone = torch.where(self.c_kingside.unsqueeze(0),
                                 torch.where(cd.unsqueeze(0) == 0, no_c[:, 0:1], no_c[:, 2:3]),
                                 torch.where(cd.unsqueeze(0) == 0, no_c[:, 1:2], no_c[:, 3:4]))
        castle_c = self.c_castle.unsqueeze(0) & is_king
        ok &= ~(castle_c & (right_gone | ~(empty_t | (tp == 13))))
        # side to move known -> only "us" moves; unknown -> either colour may be moving
        side_known = tokens[:, 64] != m
        ok &= ~(side_known.unsqueeze(1) & (cd.unsqueeze(0) == 1))
        # castling uses lc0's king->rook logit when the mover is a known king
        alt = self.c_alt.clamp(min=0)
        score = torch.where(castle_c & (self.c_alt >= 0).unsqueeze(0), scores[:, alt], score)

        # --- absolute tokens of every candidate ---
        flip56 = (flip.long() * 56).unsqueeze(1)
        af, at = cf.unsqueeze(0) ^ flip56, ct.unsqueeze(0) ^ flip56
        ctok = torch.stack([T.filea + af % 8, T.rank1 + af // 8, T.filea + at % 8, T.rank1 + at // 8,
                            self.promo_tokens[cp].unsqueeze(0).expand(B, -1)], dim=-1)   # (B,N,5)
        mv = tokens[:, 76:81]
        consistent = ((mv.unsqueeze(1) == ctok) | (mv.unsqueeze(1) == m)).all(-1)
        valid = ok & consistent
        valid = torch.where(valid.any(1, keepdim=True), valid, consistent)
        valid = torch.where(valid.any(1, keepdim=True), valid, torch.ones_like(valid))
        logp = torch.log_softmax(score.masked_fill(~valid, float("-inf")), dim=1)
        return logp, ctok

    def _move_logits(self, x, tokens, *args):
        B = x.size(0)
        logp, ctok = self.move_distribution(x, tokens, *args)
        probs = logp.exp()
        marg = probs.new_zeros(B, 5, self.V)
        marg.scatter_add_(2, ctok.transpose(1, 2), probs.unsqueeze(1).expand(B, 5, -1).contiguous())
        return torch.log(marg.clamp_min(1e-9))     # BT4-policy prior; the learned move_head adds to it

    # elbo_loss / sample are shared with MaskedDiffusion
    from MaskedDiffusion.model import elbo_loss_impl as elbo_loss  # noqa: E402
    from MaskedDiffusion.model import sample_impl as _sample_impl  # noqa: E402

    @torch.no_grad()
    def sample(self, *args, **kwargs):
        return self._sample_impl(*args, **kwargs)
