"""Minimal, dependency-free reader for lc0 network files (*.pb.gz).

Walks the protobuf wire format directly, so no protoc / generated net_pb2 is needed.
Only the fields used by the BT4 loader are interpreted (field numbers from lczero-common
proto/net.proto). Weight layers are LINEAR16-quantized: v = min + (max - min) * q / 65535.
"""
import gzip
import struct

import numpy as np


def _varint(buf, i):
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def parse_message(buf):
    """Return {field_number: [values...]}; length-delimited values are left as raw bytes."""
    buf = memoryview(buf)          # zero-copy slicing of nested messages
    fields = {}
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            val, i = _varint(buf, i)
        elif wire == 1:
            val = buf[i:i + 8]
            i += 8
        elif wire == 2:
            ln, i = _varint(buf, i)
            val = buf[i:i + ln]
            i += ln
        elif wire == 5:
            val = buf[i:i + 4]
            i += 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        fields.setdefault(field, []).append(val)
    return fields


class Msg:
    """Lazy view of a protobuf message addressed by field number."""

    def __init__(self, buf):
        self.f = parse_message(buf)

    def has(self, num):
        return num in self.f

    def msg(self, num):
        return Msg(self.f[num][-1]) if num in self.f else None

    def msgs(self, num):
        return [Msg(b) for b in self.f.get(num, [])]

    def uint(self, num, default=0):
        return self.f[num][-1] if num in self.f else default

    def float32(self, num, default=0.0):
        return struct.unpack("<f", self.f[num][-1])[0] if num in self.f else default

    def raw(self, num):
        return self.f[num][-1] if num in self.f else None


def layer_to_numpy(layer: Msg) -> np.ndarray:
    """Decode a Weights.Layer (LINEAR16) to float32."""
    params = layer.raw(3)
    if params is None:
        return np.zeros(0, dtype=np.float32)
    q = np.frombuffer(params, dtype=np.uint16).astype(np.float32)
    q /= 65535.0
    lo, hi = layer.float32(1), layer.float32(2)
    return q * (hi - lo) + lo


# --- field numbers (lczero-common proto/net.proto) ---
class F:
    # Net
    NET_FORMAT, NET_WEIGHTS = 4, 10
    # Format / NetworkFormat
    FORMAT_ENCODING, FORMAT_NETWORK = 1, 2
    NF_INPUT, NF_OUTPUT, NF_NETWORK, NF_POLICY, NF_VALUE, NF_MLH = 1, 2, 3, 4, 5, 6
    NF_DEFAULT_ACT, NF_SMOLGEN_ACT, NF_FFN_ACT, NF_INPUT_EMB = 7, 8, 9, 10
    # Weights
    IP_EMB_PREPROC_W, IP_EMB_PREPROC_B = 37, 38
    IP_EMB_W, IP_EMB_B = 25, 26
    IP_EMB_LN_G, IP_EMB_LN_B = 39, 40
    IP_MULT_GATE, IP_ADD_GATE = 33, 34
    IP_EMB_FFN, IP_EMB_FFN_LN_G, IP_EMB_FFN_LN_B = 41, 42, 43
    ENCODER, HEADCOUNT = 27, 28
    SMOLGEN_W = 35
    VALUE_HEADS, POLICY_HEADS = 44, 45
    # EncoderLayer
    EL_MHA, EL_LN1_G, EL_LN1_B, EL_FFN, EL_LN2_G, EL_LN2_B = 1, 2, 3, 4, 5, 6
    # MHA
    MHA_QW, MHA_QB, MHA_KW, MHA_KB, MHA_VW, MHA_VB, MHA_DW, MHA_DB, MHA_SMOLGEN = 1, 2, 3, 4, 5, 6, 7, 8, 9
    # Smolgen
    SG_COMPRESS, SG_D1W, SG_D1B, SG_LN1G, SG_LN1B, SG_D2W, SG_D2B, SG_LN2G, SG_LN2B = 1, 2, 3, 4, 5, 6, 7, 8, 9
    # FFN
    FFN_D1W, FFN_D1B, FFN_D2W, FFN_D2B = 1, 2, 3, 4
    # PolicyHeads / PolicyHead
    PHS_IP_POL_W, PHS_IP_POL_B, PHS_VANILLA = 1, 2, 3
    PH_IP2_W, PH_IP2_B, PH_IP3_W, PH_IP3_B, PH_IP4_W = 3, 4, 5, 6, 7
    # ValueHeads / ValueHead
    VHS_WINNER = 1
    VH_IP_W, VH_IP_B, VH_IP1_W, VH_IP1_B, VH_IP2_W, VH_IP2_B = 1, 2, 3, 4, 5, 6


def load_net(path):
    with gzip.open(path, "rb") as fh:
        return Msg(fh.read())


def network_format(net: Msg) -> dict:
    nf = net.msg(F.NET_FORMAT).msg(F.FORMAT_NETWORK)
    names = {"input": F.NF_INPUT, "output": F.NF_OUTPUT, "network": F.NF_NETWORK,
             "policy": F.NF_POLICY, "value": F.NF_VALUE, "moves_left": F.NF_MLH,
             "default_activation": F.NF_DEFAULT_ACT, "smolgen_activation": F.NF_SMOLGEN_ACT,
             "ffn_activation": F.NF_FFN_ACT, "input_embedding": F.NF_INPUT_EMB}
    return {k: nf.uint(v) for k, v in names.items()}
