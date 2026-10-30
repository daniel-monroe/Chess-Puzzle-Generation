# Chess Puzzle Generation

This repository contains the official code for our paper `Conditional Generation of Creative Chess Puzzles with Diffusion Models`.

**Authors:** [Aatu Selkee](https://openreview.net/profile?id=~Aatu_Selkee1), [Severi Rissanen](https://openreview.net/profile?id=~Severi_Rissanen1), [Xidong Feng](https://openreview.net/profile?id=~Xidong_Feng1), [Tom Zahavy](https://openreview.net/profile?id=~Tom_Zahavy2), and [Eric Malmi](https://openreview.net/profile?id=~Eric_Malmi1).

## Overview

Procedurally generating high-quality, aesthetic, and controllable chess puzzles is a complex task. Traditional methods rely on filtering positions from human play. In this project, we train a generative model for chess puzzle generation. Furthermore, we employ Reinforcement Learning via Denoising Diffusion Policy Optimization (DDPO) for further model training.

### Key Contributions
- **Masked Diffusion for Chess:** A novel application of discrete diffusion models for generating chess positions.
- **RL-based Alignment (DDPO):** Utilizing Stockfish evaluations and theme heuristics as reward signals to steer the diffusion process towards better puzzles.
- **Controllable Generation:** Generating puzzles conditional on themes, ratings, partial board states and best moves.

---

## Master's Thesis

This paper extends Aatu Selkee's master's thesis. The thesis was written with guidance from Adjunct Professor Eric Malmi and Professor Nuutti Hyvönen as well as all other co-authors. The thesis and presentation are presented in [`Thesis/`](Thesis/) and [`Thesis Presentation/`](Thesis%20Presentation/) for interested readers.

---

## Environment Setup

The environment requirements are specified in `environment.yml`.

To set up the environment, run:

```bash
conda env create -f environment.yml
conda activate environment
```

**Note:** You must have the [Stockfish engine](https://stockfishchess.org/download/) installed and accessible, or compiled inside the project directory structure (`./Stockfish/src/stockfish` by default).

---

## Project Structure

- `src/MaskedDiffusion/model.py`: Masked Diffusion architecture, training loss, and sampling methods.
- `src/MaskingSchedule/MaskingSchedule.py`: Linear, cosine, polynomial, and geometric masking schedules.
- `src/Config.py`: Configuration dataclass for tokenization, model architecture, and supervised training.
- `src/tokenization/`: FEN and move tokenization, theme and rating preprocessing, and dataset preparation.
- `src/scripts/training/`: Supervised and DDPO training scripts, replay buffer initialization, and legacy ESPO training scripts.
- `src/rl/espo.py`: Legacy ESPO losses, ELBO estimation, and grouped sampling utilities.
- `src/metrics/theme_conditioning.py`: Shared theme and move conditioning generation, theme matching rewards, and extra-theme counting.
- `src/metrics/`: Stockfish-based puzzle evaluation, theme detection, reward components, diversity measures, and replay buffer utilities.
- `src/scripts/evaluation/`: Checkpoint and Lichess evaluation, recomputation of metrics for existing puzzle CSVs, and distances within and between puzzle datasets.
- `src/scripts/generation/`: Position generation, partial-board conditioning, and branching diffusion trajectories.
- Slurm job scripts (`.sh`) are stored alongside their Python entry points in `src/scripts/training/`, `src/scripts/evaluation/`, and `src/scripts/generation/`.
- `src/analysis.ipynb`, `src/plots.ipynb`, and `src/move_visualization.ipynb`: Notebooks for analysis, plotting, and move visualization.

---

## Usage

Run the following commands from the repository root on an HPC cluster with Slurm. Before submitting jobs, adjust the settings in the corresponding `.sh` scripts to match your setup.

**Dataset preparation:** Before training, prepare the Lichess puzzle data using the preprocessing and tokenization utilities in `src/tokenization/`, and save the training and test splits as PyTorch datasets. Each sample should contain FEN tokens, best-move tokens, encoded themes, and a scaled rating, in that order. These dataset files are not included in the repository.

### 1. Supervised Training

To train the base Masked Diffusion model on a dataset of chess puzzles:

```bash
sbatch src/scripts/training/supervised.sh
```

### 2. Reinforcement Learning (DDPO)

To train a pre-trained supervised model using DDPO:

```bash
sbatch src/scripts/training/train_rl_ddpo.sh
```

### 3. Evaluation

- **Checkpoint Evaluation:** Generate and evaluate positions from every `.pt` checkpoint in a directory, saving one CSV per checkpoint.

  ```bash
  sbatch src/scripts/evaluation/evaluate_checkpoints.sh
  ```

- **Lichess Evaluation:** Evaluate puzzle legality, solution uniqueness, and counter-intuitiveness for a Lichess dataset.

  ```bash
  sbatch src/scripts/evaluation/evaluate_lichess.sh
  ```

- **Diversity and Distance:** Compute board and solution-line distances within generated puzzles and from generated puzzles to a Lichess reference dataset.

  ```bash
  sbatch src/scripts/evaluation/compute_distances.sh
  ```

- **Recompute Evaluations:** Recalculate puzzle metrics for existing CSVs.

  ```bash
  sbatch src/scripts/evaluation/recompute_evaluations.sh
  ```

### 4. Generation and Conditioning Experiments

- **Position Generation:** Generate and evaluate positions from a single checkpoint.

  ```bash
  sbatch src/scripts/generation/generate_positions.sh
  ```

- **Partial-Board Conditioning:** Generate positions with a specified partial board, themes, rating, and optionally a best move.

  ```bash
  sbatch src/scripts/generation/generate_from_partial_board.sh
  ```

- **Diffusion Trajectories:** Generate branching diffusion trajectories and evaluate their final positions.

  ```bash
  sbatch src/scripts/generation/compute_diffusion_trajectories.sh
  ```

---

## Leela BT4 backbone (BT4 finetune vs. the original model)

`src/BT4/` adds Leela Chess Zero's BT4 transformer (`BT4-1024x15x32h`, 188M parameters with
smolgen, which the Chessformer paper calls GAB, the geometric attention bias) as a drop-in
replacement for the masked-diffusion network. It takes the same inputs and produces the same
outputs (81 tokens, theme and rating conditioning, per-token logits) and is trained with the same
masked-diffusion objective (random token masking, ELBO loss) and sampler, so the training, RL,
evaluation and generation scripts work unchanged (`Config.model_type = "bt4"`).

What changes relative to lc0 BT4, and nothing else:

- **Sequence:** all 81 tokens are attention positions: the 64 squares plus the 12 FEN
  metadata tokens and the 5 best-move tokens. Squares attend to squares exactly as in BT4
  (smolgen/GAB bias), and to the 17 extra tokens through a gated second attention whose gate
  starts at zero. The extra tokens attend to all 81 tokens. All of it uses BT4's pretrained
  q/k/v projections, so at initialisation the network computes exactly BT4.
- **Square input:** square tokens become lc0's classical 112 input planes (side-to-move frame
  and `fen_only` history as in lc0). Masked squares and the theme/rating conditioning enter
  through zero-initialised projections.
- **Outputs:** every position gets logits over the full vocabulary, like the original
  model's classifier, so the losses compare directly. The best-move tokens additionally start
  from the marginals of BT4's own policy head, restricted to moves consistent with what is
  known; a learned head on the move tokens (zero at init) adds to them.

`src/BT4/check_bt4.py` verifies the port against outputs recorded from the lc0 engine and checks
that, on a known board, the move head reproduces BT4's policy over legal moves.

### Running on a GPU node (8x B300)

```bash
git clone -b bt4-finetune https://github.com/daniel-monroe/Chess-Puzzle-Generation && cd Chess-Puzzle-Generation
bash setup_b300.sh    # env (CUDA build matched to the driver), Stockfish, BT4 weights (sha256-checked), Lichess dataset, self-check
bash launch_bt4.sh    # BT4 finetune on GPUs 0-3, original MaskedDiffusion model on GPUs 4-7 (runs in the background)
```

**Requirements:** 8 GPUs, about 40 GB of free disk, and internet access (PyTorch wheels,
Stockfish, the BT4 weights from storage.lczero.org, the Lichess puzzle database). Also `git`,
`curl`, `make` and `g++`. Check the end of `setup_b300.sh`'s output: the torch line should list
all GPUs, and it should finish with `BT4 check passed`.

**What runs** (both runs use identical settings; only the network and BT4's pretrained
initialisation differ):

| | |
|---|---|
| Data | 6.16M Lichess puzzles: 6.11M train, 50,000 random test |
| Targets | all 81 tokens: 64 squares, 12 FEN fields, 5 best-move tokens |
| Run length | **1,000,000 steps at a global batch of 2,048 per run** (4 GPUs x 512), about 2.05B samples or ~335 epochs. Same sample budget as the paper's 2,000,000 steps at batch 1,024 |
| Optimiser | AdamW, lr 3e-4 for every parameter, weight decay 1e-4, 1,000 warmup steps, cosine decay to 10% |
| Validation | full 50k test set every 5,000 steps |
| Puzzle evaluation | 20,000 generated positions scored with Stockfish every 50,000 steps (legal, unique, counter-intuitive, puzzle rate) |
| Checkpoints | `model_<step>.pt` every 20,000 steps (2 newest kept, for resuming) and `model_best.pt`, the lowest validation loss so far |

Expect the runs to take on the order of days. BT4 is about 1.7x faster per step than the
original model. To shorten, set e.g. `STEPS=200000 bash launch_bt4.sh`; all settings are
environment variables documented at the top of `launch_bt4.sh`.

**Logs** go to `src/runs/supervised/bt4_finetune/` and `src/runs/supervised/bt4_original/`:

```bash
tail -f src/runs/supervised/bt4_*/train.log | grep --line-buffered -E "==>|\[val\]|\[eval\]"
.venv/bin/tensorboard --logdir src/runs/supervised --bind_all   # curves
```

`[train]` lines appear every 100 steps, `[val]` every 5,000 and `[eval]` every 50,000.

**Stop / resume:** `kill $(cat src/runs/supervised/bt4_*/launcher.pid)`. Running
`bash launch_bt4.sh` again resumes each run from its newest checkpoint.

**If a run crashes in `torch.compile`** (new GPU architectures), relaunch with
`COMPILE=0 bash launch_bt4.sh`, adding `TORCHDYNAMO_DISABLE=1` if the original model's sampler
fails too. Training resumes from the last checkpoint.

**Under Slurm:** `sbatch src/scripts/training/bt4.sh`.
