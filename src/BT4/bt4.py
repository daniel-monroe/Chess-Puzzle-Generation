"""PyTorch port of Leela Chess Zero's BT4 transformer (BT4-1024x15x32h), loadable from lc0 .pb.gz.

Mirrors lczero-training tf/tfprocess.py (construct_net / encoder_layer / mha / ffn /
smolgen_weights / policy head / value head):
  - "new" embedding: the 12 current-position piece planes of all 64 squares are flattened
    (768) -> Dense(64*512) "preprocess" -> per-square 512 features, concatenated with the
    112 input planes (624) -> Dense(1024) mish -> LayerNorm -> ma-gating -> DeepNorm FFN.
  - 15 post-LN encoder layers with DeepNorm residual scaling alpha = (2N)^-0.25.
  - Attention logits get an additive, position-dependent bias from smolgen, which the
    Chessformer paper calls GAB (geometric attention bias): the board is compressed
    to a per-head vector and expanded by a dense layer shared across all layers into a
    64x64 bias for each head.
  - Vanilla attention policy head: from/to logits = q.k / sqrt(d) and promotion offsets.
  - Winner (WDL) value head.
LayerNorm eps is the Keras default (1e-3). Default activation mish, smolgen swish.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

LN_EPS = 1e-3


@dataclass
class BT4Config:
    input_planes: int = 112
    embedding_size: int = 1024
    embedding_dense_sz: int = 512
    encoder_layers: int = 15
    encoder_heads: int = 32
    encoder_dff: int = 1536
    policy_embedding_size: int = 1024
    policy_d_model: int = 1024
    value_embedding_size: int = 128
    smolgen_hidden_channels: int = 32
    smolgen_hidden_sz: int = 256
    smolgen_gen_sz: int = 256

    @property
    def head_depth(self) -> int:
        return self.embedding_size // self.encoder_heads

    @property
    def deepnorm_alpha(self) -> float:
        return (2.0 * self.encoder_layers) ** -0.25

    @property
    def deepnorm_beta(self) -> float:
        return (8.0 * self.encoder_layers) ** -0.25


def _ln(dim):
    return nn.LayerNorm(dim, eps=LN_EPS)


class FFN(nn.Module):
    def __init__(self, d, dff):
        super().__init__()
        self.dense1 = nn.Linear(d, dff)
        self.dense2 = nn.Linear(dff, d)

    def forward(self, x):
        return self.dense2(F.mish(self.dense1(x)))


class Smolgen(nn.Module):
    """Smolgen / GAB: board-dependent attention bias (B, H, 64, 64)."""

    def __init__(self, cfg: BT4Config):
        super().__init__()
        self.cfg = cfg
        self.compress = nn.Linear(cfg.embedding_size, cfg.smolgen_hidden_channels, bias=False)
        self.dense1 = nn.Linear(64 * cfg.smolgen_hidden_channels, cfg.smolgen_hidden_sz)
        self.ln1 = _ln(cfg.smolgen_hidden_sz)
        self.dense2 = nn.Linear(cfg.smolgen_hidden_sz, cfg.encoder_heads * cfg.smolgen_gen_sz)
        self.ln2 = _ln(cfg.encoder_heads * cfg.smolgen_gen_sz)

    def forward(self, x, global_gen: nn.Linear):
        B = x.size(0)
        h = self.compress(x[:, :64]).reshape(B, 64 * self.cfg.smolgen_hidden_channels)   # squares only
        h = self.ln1(F.silu(self.dense1(h)))
        g = self.ln2(F.silu(self.dense2(h)))
        g = g.reshape(B, self.cfg.encoder_heads, self.cfg.smolgen_gen_sz)
        return global_gen(g).reshape(B, self.cfg.encoder_heads, 64, 64)


class EncoderLayer(nn.Module):
    def __init__(self, cfg: BT4Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.embedding_size
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.dense = nn.Linear(d, d)
        self.smolgen = Smolgen(cfg)
        self.ln1 = _ln(d)
        self.ffn = FFN(d, cfg.encoder_dff)
        self.ln2 = _ln(d)
        self.alpha = cfg.deepnorm_alpha

    def forward(self, x, global_gen, extra=None):
        """x: (B, T, D) with the 64 square tokens first.

        For T > 64 (extra non-square tokens), extra = (bias (H, T, T), gate (H,)):
          * square queries attend to the 64 squares exactly as in BT4 (smolgen / GAB bias), and
            to the extra tokens in a second softmax whose output is added with a per-head gate
            (zero at init -> exactly BT4 at init, opened by training);
          * extra-token queries attend to all T tokens with the learned bias rows.
        All attention uses this layer's (pretrained) q/k/v projections."""
        B, T, D = x.shape
        H, hd = self.cfg.encoder_heads, self.cfg.head_depth
        q = self.wq(x).view(B, T, H, hd).transpose(1, 2)
        k = self.wk(x).view(B, T, H, hd).transpose(1, 2)
        v = self.wv(x).view(B, T, H, hd).transpose(1, 2)
        smol = self.smolgen(x, global_gen).to(q.dtype)
        # softmax(q.k / sqrt(hd) + smolgen) v  -- SDPA's default scale is 1/sqrt(hd)
        if T == 64:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=smol)
        else:
            bias, gate = extra
            bias = bias.to(q.dtype)
            qs, ks, vs = q[:, :, :64], k[:, :, :64], v[:, :, :64]
            out_sq = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=smol)
            out_sx = F.scaled_dot_product_attention(qs, k[:, :, 64:], v[:, :, 64:],
                                                    attn_mask=bias[:, :64, 64:].unsqueeze(0).expand(B, -1, -1, -1))
            out_sq = out_sq + gate.view(1, H, 1, 1).to(q.dtype) * out_sx
            out_x = F.scaled_dot_product_attention(q[:, :, 64:], k, v,
                                                   attn_mask=bias[:, 64:, :].unsqueeze(0).expand(B, -1, -1, -1))
            out = torch.cat([out_sq, out_x], dim=2)
        out = self.dense(out.transpose(1, 2).reshape(B, T, D))
        x = self.ln1(x + out * self.alpha)
        return self.ln2(x + self.ffn(x) * self.alpha)


class BT4Body(nn.Module):
    """Input planes (B, 64, 112) per square -> encoder output (B, 64, 1024)."""

    def __init__(self, cfg: BT4Config = BT4Config()):
        super().__init__()
        self.cfg = cfg
        e = cfg.embedding_size
        self.emb_preproc = nn.Linear(64 * 12, 64 * cfg.embedding_dense_sz)
        self.embedding = nn.Linear(cfg.input_planes + cfg.embedding_dense_sz, e)
        self.emb_ln = _ln(e)
        self.mult_gate = nn.Parameter(torch.ones(64, e))
        self.add_gate = nn.Parameter(torch.zeros(64, e))
        self.emb_ffn = FFN(e, cfg.encoder_dff)
        self.emb_ffn_ln = _ln(e)
        self.global_smolgen = nn.Linear(cfg.smolgen_gen_sz, 64 * 64, bias=False)
        self.encoders = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.encoder_layers))
        self.alpha = cfg.deepnorm_alpha

    def embed(self, flow, preproc_extra=None, emb_extra=None):
        """flow: (B,64,112). Optional additive terms (used by the diffusion adapter):
        preproc_extra (B, 64*512) added to the preprocess output, emb_extra (B,64,1024)
        added to the square embedding before its activation."""
        B = flow.size(0)
        pos = self.emb_preproc(flow[..., :12].reshape(B, 64 * 12))
        if preproc_extra is not None:
            pos = pos + preproc_extra
        flow = torch.cat([flow, pos.view(B, 64, self.cfg.embedding_dense_sz)], dim=-1)
        h = self.embedding(flow)
        if emb_extra is not None:
            h = h + emb_extra
        h = self.emb_ln(F.mish(h))
        # Keras NonNeg constraint on the multiplicative gate.
        h = h * F.relu(self.mult_gate) + self.add_gate
        return self.emb_ffn_ln(h + self.emb_ffn(h) * self.alpha)

    def forward(self, flow, preproc_extra=None, emb_extra=None, checkpoint_activations=False,
                extra_tokens=None, extra_bias=None, extra_gate=None):
        """extra_tokens (B, K, d): embedded non-square tokens appended after the 64 squares, attending
        to and attended by every token in every layer; extra_bias (layers, H, T, T) and
        extra_gate (layers, H) parametrise that attention (see EncoderLayer.forward)."""
        x = self.embed(flow, preproc_extra, emb_extra)
        if extra_tokens is not None:
            x = torch.cat([x, extra_tokens.to(x.dtype)], dim=1)
        for i, layer in enumerate(self.encoders):
            eb = (extra_bias[i], extra_gate[i]) if extra_tokens is not None else None
            if checkpoint_activations and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, self.global_smolgen, eb, use_reentrant=False)
            else:
                x = layer(x, self.global_smolgen, eb)
        return x


class AttentionPolicyHead(nn.Module):
    """lc0 vanilla attention policy head.

    raw_scores() returns (scores (B,64,64) = q.k/sqrt(d), promo (B,8,3) offsets) where the
    q/r/b promotion logit for a7->a8-type move (from file i, to file j) is
    scores[48+i, 56+j] + promo[j, c]; knight promotion uses the plain from->to logit.
    """

    def __init__(self, cfg: BT4Config = BT4Config()):
        super().__init__()
        self.cfg = cfg
        self.pol_emb = nn.Linear(cfg.embedding_size, cfg.policy_embedding_size)
        self.wq = nn.Linear(cfg.policy_embedding_size, cfg.policy_d_model)
        self.wk = nn.Linear(cfg.policy_embedding_size, cfg.policy_d_model)
        self.ppo = nn.Linear(cfg.policy_d_model, 4, bias=False)

    def raw_scores(self, x):
        t = F.mish(self.pol_emb(x))
        q, k = self.wq(t), self.wk(t)
        dk = math.sqrt(self.cfg.policy_d_model)
        scores = torch.matmul(q, k.transpose(1, 2)) / dk
        # promotion offsets come from the keys of the last rank (raw scale, see tfprocess)
        po = self.ppo(k[:, 56:64, :])                       # (B, 8 to-file, 4) [q, r, b, n]
        promo = po[..., :3] + po[..., 3:4]                  # (B, 8, 3) q/r/b offsets incl. knight
        return scores, promo, k

    def forward(self, x):
        return self.raw_scores(x)[:2]


class WDLValueHead(nn.Module):
    def __init__(self, cfg: BT4Config = BT4Config()):
        super().__init__()
        self.emb = nn.Linear(cfg.embedding_size, cfg.value_embedding_size)
        self.dense1 = nn.Linear(64 * cfg.value_embedding_size, 128)
        self.dense2 = nn.Linear(128, 3)

    def forward(self, x):
        h = F.mish(self.emb(x)).flatten(1)
        return self.dense2(F.mish(self.dense1(h)))


class BT4(nn.Module):
    """Plain BT4 (body + vanilla policy + winner WDL head), for checking against lc0."""

    def __init__(self, cfg: BT4Config = BT4Config()):
        super().__init__()
        self.cfg = cfg
        self.body = BT4Body(cfg)
        self.policy = AttentionPolicyHead(cfg)
        self.value = WDLValueHead(cfg)

    def forward(self, planes):
        """planes: (B,112,8,8) lc0 input planes. Returns (scores, promo, wdl_logits)."""
        B = planes.size(0)
        flow = planes.float().permute(0, 2, 3, 1).reshape(B, 64, self.cfg.input_planes)
        x = self.body(flow)
        scores, promo = self.policy(x)
        return scores, promo, self.value(x)


# =============================== proto loading ===============================

def _fill(param: torch.Tensor, arr, name, transpose_from=None):
    t = torch.from_numpy(arr.copy())
    if transpose_from is not None:          # stored as JAX/TF-logical.T flattened
        t = t.view(*transpose_from).t()
    t = t.reshape(param.shape)
    with torch.no_grad():
        param.copy_(t)


def load_bt4_weights(body: BT4Body, pb_path, policy: AttentionPolicyHead | None = None,
                     value: WDLValueHead | None = None, verbose=True):
    """Load lc0 BT4 weights into the PyTorch modules. Returns number of tensors loaded."""
    from .lc0_proto import F as PF, layer_to_numpy as L, load_net, network_format

    net = load_net(pb_path)
    fmt = network_format(net)
    w = net.msg(PF.NET_WEIGHTS)
    cfg = body.cfg
    n = 0

    def put(param, layer_msg, name, transpose_from=None):
        nonlocal n
        if layer_msg is None:
            raise KeyError(f"missing weight {name} in {pb_path}")
        arr = L(layer_msg)
        if arr.size != param.numel():
            raise ValueError(f"{name}: proto has {arr.size} values, param has {param.numel()}")
        _fill(param, arr, name, transpose_from)
        n += 1

    def lin(mod: nn.Linear, wmsg, bmsg, name):
        put(mod.weight, wmsg, name + ".w")      # lc0 stores dense kernels as (out, in)
        if mod.bias is not None:
            put(mod.bias, bmsg, name + ".b")

    def ln(mod: nn.LayerNorm, g, b, name):
        put(mod.weight, g, name + ".gamma")
        put(mod.bias, b, name + ".beta")

    lin(body.emb_preproc, w.msg(PF.IP_EMB_PREPROC_W), w.msg(PF.IP_EMB_PREPROC_B), "emb_preproc")
    lin(body.embedding, w.msg(PF.IP_EMB_W), w.msg(PF.IP_EMB_B), "embedding")
    ln(body.emb_ln, w.msg(PF.IP_EMB_LN_G), w.msg(PF.IP_EMB_LN_B), "emb_ln")
    e = cfg.embedding_size
    put(body.mult_gate, w.msg(PF.IP_MULT_GATE), "mult_gate", transpose_from=(e, 64))
    put(body.add_gate, w.msg(PF.IP_ADD_GATE), "add_gate", transpose_from=(e, 64))
    ffn = w.msg(PF.IP_EMB_FFN)
    lin(body.emb_ffn.dense1, ffn.msg(PF.FFN_D1W), ffn.msg(PF.FFN_D1B), "emb_ffn.dense1")
    lin(body.emb_ffn.dense2, ffn.msg(PF.FFN_D2W), ffn.msg(PF.FFN_D2B), "emb_ffn.dense2")
    ln(body.emb_ffn_ln, w.msg(PF.IP_EMB_FFN_LN_G), w.msg(PF.IP_EMB_FFN_LN_B), "emb_ffn_ln")
    put(body.global_smolgen.weight, w.msg(PF.SMOLGEN_W), "smolgen_w")

    encs = w.msgs(PF.ENCODER)
    if len(encs) != len(body.encoders):
        raise ValueError(f"proto has {len(encs)} encoder layers, model has {len(body.encoders)}")
    if w.uint(PF.HEADCOUNT) != cfg.encoder_heads:
        raise ValueError(f"proto headcount {w.uint(PF.HEADCOUNT)} != {cfg.encoder_heads}")
    for i, (el, layer) in enumerate(zip(encs, body.encoders)):
        p = f"encoder{i}"
        mha = el.msg(PF.EL_MHA)
        lin(layer.wq, mha.msg(PF.MHA_QW), mha.msg(PF.MHA_QB), p + ".wq")
        lin(layer.wk, mha.msg(PF.MHA_KW), mha.msg(PF.MHA_KB), p + ".wk")
        lin(layer.wv, mha.msg(PF.MHA_VW), mha.msg(PF.MHA_VB), p + ".wv")
        lin(layer.dense, mha.msg(PF.MHA_DW), mha.msg(PF.MHA_DB), p + ".dense")
        sg = mha.msg(PF.MHA_SMOLGEN)
        put(layer.smolgen.compress.weight, sg.msg(PF.SG_COMPRESS), p + ".smolgen.compress")
        lin(layer.smolgen.dense1, sg.msg(PF.SG_D1W), sg.msg(PF.SG_D1B), p + ".smolgen.dense1")
        ln(layer.smolgen.ln1, sg.msg(PF.SG_LN1G), sg.msg(PF.SG_LN1B), p + ".smolgen.ln1")
        lin(layer.smolgen.dense2, sg.msg(PF.SG_D2W), sg.msg(PF.SG_D2B), p + ".smolgen.dense2")
        ln(layer.smolgen.ln2, sg.msg(PF.SG_LN2G), sg.msg(PF.SG_LN2B), p + ".smolgen.ln2")
        ln(layer.ln1, el.msg(PF.EL_LN1_G), el.msg(PF.EL_LN1_B), p + ".ln1")
        f = el.msg(PF.EL_FFN)
        lin(layer.ffn.dense1, f.msg(PF.FFN_D1W), f.msg(PF.FFN_D1B), p + ".ffn.dense1")
        lin(layer.ffn.dense2, f.msg(PF.FFN_D2W), f.msg(PF.FFN_D2B), p + ".ffn.dense2")
        ln(layer.ln2, el.msg(PF.EL_LN2_G), el.msg(PF.EL_LN2_B), p + ".ln2")

    if policy is not None:
        phs = w.msg(PF.POLICY_HEADS)
        lin(policy.pol_emb, phs.msg(PF.PHS_IP_POL_W), phs.msg(PF.PHS_IP_POL_B), "policy.embedding")
        van = phs.msg(PF.PHS_VANILLA)
        lin(policy.wq, van.msg(PF.PH_IP2_W), van.msg(PF.PH_IP2_B), "policy.wq")
        lin(policy.wk, van.msg(PF.PH_IP3_W), van.msg(PF.PH_IP3_B), "policy.wk")
        put(policy.ppo.weight, van.msg(PF.PH_IP4_W), "policy.ppo")

    if value is not None:
        win = w.msg(PF.VALUE_HEADS).msg(PF.VHS_WINNER)
        lin(value.emb, win.msg(PF.VH_IP_W), win.msg(PF.VH_IP_B), "value.emb")
        lin(value.dense1, win.msg(PF.VH_IP1_W), win.msg(PF.VH_IP1_B), "value.dense1")
        lin(value.dense2, win.msg(PF.VH_IP2_W), win.msg(PF.VH_IP2_B), "value.dense2")

    if verbose:
        print(f"[bt4] loaded {n} tensors from {pb_path} (format {fmt})")
    return fmt


def lc0_style_init_(body: BT4Body, *heads: nn.Module):
    """Random init used for the from-scratch run (lczero-training conventions):
    DeepNorm-scaled xavier for v / out-proj / FFN, glorot-normal elsewhere, zero biases."""
    beta = body.cfg.deepnorm_beta
    deep = set()
    for layer in body.encoders:
        deep.update({layer.wv, layer.dense, layer.ffn.dense1, layer.ffn.dense2})
    deep.update({body.emb_ffn.dense1, body.emb_ffn.dense2})
    for mod in [body, *heads]:
        for m in mod.modules():
            if isinstance(m, nn.Linear):
                fan_in, fan_out = m.in_features, m.out_features
                scale = beta if m in deep else 1.0
                std = math.sqrt(2.0 * scale / (fan_in + fan_out))
                nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    with torch.no_grad():
        body.mult_gate.fill_(1.0)
        body.add_gate.zero_()
