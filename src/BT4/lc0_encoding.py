"""lc0 INPUT_CLASSICAL_112_PLANE encoding of a single FEN (no game history), as BT4 expects.

Planes (side-to-move perspective; for black to move the board is flipped vertically and
"ours" = black):
  13*i + 0..5   our P N B R Q K, history slot i (i = 0 current, 1..7 history)
  13*i + 6..11  their P N B R Q K
  13*i + 12     repetition (always 0 here)
  104..107      castling: we can O-O-O, we can O-O, they can O-O-O, they can O-O
  108           all ones if black to move
  109           rule50 half-move counter (raw count)
  110           zeros
  111           ones
History fill mirrors lc0's default `--history-fill=fen_only`: slots 1..7 repeat the current
position, and if an en passant square is set, the opponent's double pawn push is undone in
those slots. `history="none"` leaves slots 1..7 empty instead.
"""
import chess
import numpy as np

PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def fen_to_planes(fen: str, history: str = "repeat") -> np.ndarray:
    board = chess.Board(fen)
    black = board.turn == chess.BLACK
    planes = np.zeros((112, 64), dtype=np.float32)

    def sq(s):  # absolute square -> side-to-move frame
        return s ^ 56 if black else s

    cur = np.zeros((12, 64), dtype=np.float32)
    us, them = board.turn, not board.turn
    for k, pt in enumerate(PIECE_ORDER):
        for s in board.pieces(pt, us):
            cur[k, sq(s)] = 1
        for s in board.pieces(pt, them):
            cur[6 + k, sq(s)] = 1
    planes[0:12] = cur

    if history == "repeat":
        hist = cur.copy()
        ep = board.ep_square
        if ep is not None:
            f = chess.square_file(ep)
            # their pawn that just double-pushed sits on our-frame rank 5; put it back on rank 7
            hist[6, 32 + f] = 0
            hist[6, 48 + f] = 1
        for i in range(1, 8):
            planes[13 * i:13 * i + 12] = hist

    wk, wq = board.has_kingside_castling_rights(chess.WHITE), board.has_queenside_castling_rights(chess.WHITE)
    bk, bq = board.has_kingside_castling_rights(chess.BLACK), board.has_queenside_castling_rights(chess.BLACK)
    us_oo, us_ooo, them_oo, them_ooo = (bk, bq, wk, wq) if black else (wk, wq, bk, bq)
    planes[104] = float(us_ooo)
    planes[105] = float(us_oo)
    planes[106] = float(them_ooo)
    planes[107] = float(them_oo)
    planes[108] = float(black)
    planes[109] = float(board.halfmove_clock)
    planes[111] = 1.0
    return planes.reshape(112, 8, 8)


def move_to_frame_squares(board: chess.Board, move: chess.Move):
    """(from, to, promo) in the side-to-move frame, with castling as king->rook (lc0 style)."""
    f, t = move.from_square, move.to_square
    if board.is_castling(move):
        t = chess.square(7 if chess.square_file(t) == 6 else 0, chess.square_rank(f))
    if board.turn == chess.BLACK:
        f, t = f ^ 56, t ^ 56
    return f, t, move.promotion


def policy_logits_for_legal_moves(scores, promo, board: chess.Board):
    """scores (64,64), promo (8,3) for one position -> {uci: logit} over legal moves."""
    out = {}
    for mv in board.legal_moves:
        f, t, p = move_to_frame_squares(board, mv)
        logit = scores[f, t]
        if p is not None and p != chess.KNIGHT:
            logit = logit + promo[t - 56, {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2}[p]]
        out[mv.uci()] = float(logit)
    return out
