"""Build src/dataset/with_best_move/{trainset,testset}.pt from the Lichess puzzle database.

Follows the thesis: the puzzle position is the CSV FEN after the opponent's first move,
the target move is the second move of the PV, and 50,000 random puzzles form the test set.
Each sample is (fen tokens [76], best-move tokens [5], themes multi-hot [66], scaled rating).

    python src/tokenization/prepare_dataset.py [--csv path/to/lichess_db_puzzle.csv(.zst)]
"""
import argparse
import io
import os
import sys
import urllib.request
from multiprocessing import Pool
from pathlib import Path

# run from anywhere: put src/ first (and drop this dir, whose tokenization.py would shadow the package)
sys.path[0] = str(Path(__file__).resolve().parents[1])

import chess
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from tokenization.tokenization import tokenize

URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"


def puzzle_fen(args):
    fen, moves = args
    try:
        board = chess.Board(fen)
        board.push_uci(moves.split()[0])
        b = board.fen()
        half, full = b.split()[4:6]
        if int(half) > 99 or int(full) > 999:      # tokenizer has 2 / 3 counter digits
            return None
        return b
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="src/dataset/lichess_db_puzzle.csv.zst")
    p.add_argument("--out", default="src/dataset/with_best_move")
    p.add_argument("--n_test", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args()

    csv = Path(args.csv)
    if not csv.exists():
        csv.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL} -> {csv}")
        urllib.request.urlretrieve(URL, csv)

    print("reading", csv)
    if csv.suffix == ".zst":
        import zstandard
        with open(csv, "rb") as fh:
            data = zstandard.ZstdDecompressor().stream_reader(fh).read()
        df = pd.read_csv(io.BytesIO(data), usecols=["FEN", "Moves", "Rating", "Themes"])
    else:
        df = pd.read_csv(csv, usecols=["FEN", "Moves", "Rating", "Themes"])
    df = df.dropna(subset=["FEN", "Moves", "Rating"]).reset_index(drop=True)
    df["Themes"] = df["Themes"].fillna("")
    print(f"{len(df):,} puzzles; computing puzzle positions with {args.workers} workers")

    with Pool(args.workers) as pool:
        df["Puzzle_FEN"] = pool.map(puzzle_fen, zip(df["FEN"], df["Moves"]), chunksize=20_000)
    df = df[df["Puzzle_FEN"].notna() & (df["Moves"].str.split().str.len() >= 2)].reset_index(drop=True)
    print(f"{len(df):,} usable puzzles")

    fens, moves, themes, ratings = tokenize(df)
    rng = np.random.default_rng(args.seed)
    perm = torch.from_numpy(rng.permutation(len(df)))
    test_idx, train_idx = perm[:args.n_test], perm[args.n_test:]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, idx in [("trainset", train_idx), ("testset", test_idx)]:
        ds = TensorDataset(fens[idx], moves[idx], themes[idx], ratings[idx])
        torch.save(ds, out / f"{name}.pt")
        print(f"saved {out / name}.pt ({len(ds):,} samples)")
    df.iloc[test_idx.numpy()][["Puzzle_FEN", "Moves", "Rating", "Themes"]].to_csv(out / "testset.csv", index=False)


if __name__ == "__main__":
    main()
