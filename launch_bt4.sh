#!/usr/bin/env bash
# Launch BT4 puzzle-diffusion training runs side by side on one node (run setup_b300.sh first).
#
#   bash launch_bt4.sh                     # BT4 finetune vs. the repo's original model, GPUs split in half
#   WAIT=1 bash launch_bt4.sh              # stay in the foreground (e.g. inside an sbatch job)
#
# Runs:
#   finetune  Leela BT4 adapted to the puzzle tokens, initialised from the lc0 BT4 weights
#   original  the repo's original MaskedDiffusion transformer (16x1024, 8 heads), from scratch
# Both runs use the same settings: data, targets (PREDICT_MOVES), global batch, steps, AdamW
# with lr 3e-4 and weight decay 1e-4, WARMUP then cosine decay, same validation and
# evaluation. Only the network (and BT4's pretrained init) differ.
# Each run gets an equal share of the GPUs (GPUS=0,1,2,... to choose them) and is resumed
# from its newest checkpoint if the run directory already exists.
#
# Knobs (env vars, defaults in brackets):
#   PREFIX [bt4]  BATCH [2048 global per run]  STEPS [400000]  WARMUP [1000]  KEEP_CKPTS [2]
#   LR_NEW [3e-4] new layers of the finetune   LR_PRE [3e-4] pretrained BT4 layers (same as LR_ORIGINAL by default)
#   LR_ORIGINAL [3e-4]  WD [1e-4]  PREDICT_MOVES [1]  COMPILE [1]  CKPT_ACT [0]
#   SAVE_EVERY [20000]  VAL_EVERY [5000]  EVAL_EVERY [50000]  N_EVAL [20000]  SKIP_INITIAL_EVAL [1]
#   EXTRA_ARGS [""] appended to every run's command line
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)
# Optional first argument: a file of KEY=VALUE lines (same knobs as the env vars above).
if [ $# -ge 1 ]; then
  set -a; . "$1"; set +a
fi

RUNS=${RUNS:-"finetune original"}
PREFIX=${PREFIX:-bt4}
BATCH=${BATCH:-2048}
STEPS=${STEPS:-400000}
WARMUP=${WARMUP:-1000}
KEEP_CKPTS=${KEEP_CKPTS:-2}
LR_NEW=${LR_NEW:-3e-4}
LR_PRE=${LR_PRE:-3e-4}
LR_ORIGINAL=${LR_ORIGINAL:-3e-4}
WD=${WD:-1e-4}
PREDICT_MOVES=${PREDICT_MOVES:-1}
COMPILE=${COMPILE:-1}
CKPT_ACT=${CKPT_ACT:-0}
SAVE_EVERY=${SAVE_EVERY:-20000}
VAL_EVERY=${VAL_EVERY:-5000}
EVAL_EVERY=${EVAL_EVERY:-50000}
N_EVAL=${N_EVAL:-20000}
SKIP_INITIAL_EVAL=${SKIP_INITIAL_EVAL:-1}
EXTRA_ARGS=${EXTRA_ARGS:-}
WEIGHTS=${BT4_WEIGHTS:-$ROOT/weights/BT4-1024x15x32h-swa-6147500.pb.gz}
RUNS_DIR=${RUNS_DIR:-$ROOT/src/runs/supervised}
PY=$ROOT/.venv/bin/python

[ -x "$PY" ] || { echo "no .venv: run setup_b300.sh first"; exit 1; }
[ -f src/dataset/with_best_move/trainset.pt ] || { echo "no dataset: run setup_b300.sh first"; exit 1; }

if [ -n "${GPUS:-}" ]; then IFS=, read -ra gpu_list <<< "$GPUS"
else mapfile -t gpu_list < <(nvidia-smi --query-gpu=index --format=csv,noheader); fi
read -ra run_list <<< "$RUNS"
n_runs=${#run_list[@]}
per_run=$(( ${#gpu_list[@]} / n_runs ))
[ "$per_run" -ge 1 ] || { echo "need at least one GPU per run (${#gpu_list[@]} GPUs, $n_runs runs)"; exit 1; }
cpus_per_gpu=$(( $(nproc) / ${#gpu_list[@]} ))
[ $((BATCH % per_run)) -eq 0 ] || { echo "BATCH=$BATCH must be divisible by GPUs per run ($per_run)"; exit 1; }

mkdir -p "$RUNS_DIR"
pids=()
for i in "${!run_list[@]}"; do
  run=${run_list[$i]}
  gpus=$(IFS=,; echo "${gpu_list[*]:$((i * per_run)):$per_run}")
  name="${PREFIX}_${run}"
  common=(--run_name "$name" --runs_dir "$RUNS_DIR" --resume --batch_size "$BATCH" --n_steps "$STEPS"
          --warmup_steps "$WARMUP" --weight_decay "$WD" --keep_checkpoints "$KEEP_CKPTS" --predict_moves "$PREDICT_MOVES" --compile "$COMPILE"
          --checkpoint_activations "$CKPT_ACT" --save_interval "$SAVE_EVERY" --validation_interval "$VAL_EVERY"
          --evaluation_interval "$EVAL_EVERY" --n_evaluation_positions "$N_EVAL"
          --num_workers 4 --eval_cpus "$(( cpus_per_gpu > 6 ? cpus_per_gpu - 4 : 2 ))")
  [ "$SKIP_INITIAL_EVAL" = 1 ] && common+=(--skip_initial_eval)
  case $run in
    finetune) args=(--model bt4 --bt4_init lc0 --bt4_weights "$WEIGHTS" --lr "$LR_NEW" --lr_pretrained "$LR_PRE") ;;
    original) args=(--model masked_diffusion --lr "$LR_ORIGINAL") ;;
    *) echo "unknown run '$run' (finetune|original)"; exit 1 ;;
  esac
  mkdir -p "$RUNS_DIR/$name"
  logf="$RUNS_DIR/$name/train.log"
  echo "[$name] GPUs $gpus -> $logf"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$gpus OMP_NUM_THREADS=4 TORCHINDUCTOR_CACHE_DIR="$ROOT/.inductor_cache/$name" \
    nohup "$ROOT/.venv/bin/torchrun" --standalone --nproc_per_node "$per_run" \
      src/scripts/training/supervised_training.py "${common[@]}" "${args[@]}" $EXTRA_ARGS \
      >> "$logf" 2>&1 &
  pids+=($!)
  echo "$!" > "$RUNS_DIR/$name/launcher.pid"
done

cat <<EOF

Launched ${run_list[*]} (pids ${pids[*]}).
  follow:      tail -f $RUNS_DIR/${PREFIX}_*/train.log
  tensorboard: $ROOT/.venv/bin/tensorboard --logdir $RUNS_DIR --bind_all
  stop:        kill \$(cat $RUNS_DIR/${PREFIX}_*/launcher.pid)
  re-running this script resumes each run from its newest checkpoint.
EOF

if [ "${WAIT:-0}" = 1 ]; then
  status=0
  for p in "${pids[@]}"; do wait "$p" || status=$?; done
  exit $status
fi
