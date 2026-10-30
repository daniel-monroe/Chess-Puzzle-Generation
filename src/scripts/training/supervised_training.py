import sys
from pathlib import Path

# Keep src imports available when this file is launched directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp

from datetime import datetime
import os
import argparse
import math
import re
import time

from MaskedDiffusion.model import MaskedDiffusion
from models import build_model
from Config import Config
from tokenization.tokenization import tokens_to_fen, tokens_to_move, scale_ratings

import queue
from joblib import Parallel, delayed
from chess.engine import SimpleEngine
from metrics.themes import legal, get_unique_puzzle_from_fen, counter_intuitive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--checkpoint_name", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--n_evaluation_positions", type=int, default=50000)
    parser.add_argument("--evaluation_interval", type=int, default=100_000)
    # --- model selection (defaults reproduce the original MaskedDiffusion run) ---
    parser.add_argument("--model", choices=["masked_diffusion", "bt4"], default="masked_diffusion")
    parser.add_argument("--bt4_weights", type=str, default=None, help="lc0 BT4 .pb.gz")
    parser.add_argument("--bt4_init", choices=["lc0", "scratch"], default="lc0")
    parser.add_argument("--bt4_history", choices=["repeat", "none"], default="repeat")
    parser.add_argument("--predict_moves", type=int, choices=[0, 1], default=0)
    # --- optimisation (None = Config defaults) ---
    parser.add_argument("--batch_size", type=int, default=1024, help="global batch size")
    parser.add_argument("--n_steps", type=int, default=2_000_000)
    parser.add_argument("--lr", type=float, default=None, help="lr for new / from-scratch parameters")
    parser.add_argument("--lr_pretrained", type=float, default=None, help="lr for parameters loaded from lc0 (bt4_init=lc0)")
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=100_000)
    parser.add_argument("--validation_interval", type=int, default=10_000)
    parser.add_argument("--train_logging_interval", type=int, default=10)
    parser.add_argument("--checkpoint_activations", type=int, choices=[0, 1], default=0)
    parser.add_argument("--compile", type=int, choices=[0, 1], default=1)
    # --- infrastructure ---
    parser.add_argument("--dataset_dir", type=str, default="src/dataset/with_best_move")
    parser.add_argument("--runs_dir", type=str, default="src/runs/supervised")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--eval_cpus", type=int, default=None, help="Stockfish processes per GPU for puzzle evaluation")
    parser.add_argument("--stockfish", type=str, default="Stockfish/src/stockfish")
    parser.add_argument("--skip_initial_eval", action="store_true")
    parser.add_argument("--keep_checkpoints", type=int, default=0, help="keep only the newest N checkpoints (0 = keep all)")
    parser.add_argument("--resume", action="store_true", help="continue from the newest checkpoint in the run directory, if any")
    args = parser.parse_args()
    distributed = args.distributed or int(os.environ.get("WORLD_SIZE", "1")) > 1

    base_path = Path("./src")
    
    run_name = args.run_name
    if not run_name:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging_path = Path(args.runs_dir) / run_name
    if args.resume and args.checkpoint_name is None and logging_path.exists():
        ckpts = sorted(logging_path.glob("model_*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
        if ckpts:
            args.checkpoint_name = ckpts[-1].name
    continue_from_checkpoint = args.checkpoint_name != None

    # ====================== LOAD CHECKPOINT ======================
    if continue_from_checkpoint:
        checkpoint = torch.load(logging_path / args.checkpoint_name, map_location="cpu", weights_only=False)

    # ====================== DEVICE ======================
    if distributed:
        assert torch.cuda.is_available()
        if "LOCAL_RANK" in os.environ:  # torchrun (sets MASTER_ADDR/PORT itself)
            local_rank = int(os.environ["LOCAL_RANK"])
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
        else:  # srun
            local_rank = int(os.environ["SLURM_PROCID"])
            rank = local_rank  # should be something different with multiple nodes
            world_size = int(os.environ["SLURM_NTASKS"])
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "12355")
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        master_process = rank == 0
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        master_process = True
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    device = torch.device(device)

    # ====================== CONFIG ======================
    if continue_from_checkpoint:
        config = checkpoint["config"]
        # schedule / logging settings from the command line win over the checkpoint's
        config.n_steps = args.n_steps
        config.save_interval = args.save_interval
        config.validation_interval = args.validation_interval
        config.train_logging_interval = args.train_logging_interval
        if args.lr is not None:
            config.lr = args.lr
    else:
        config = Config(train_logging_interval=args.train_logging_interval, validation_interval=args.validation_interval,
                        n_steps=args.n_steps, save_interval=args.save_interval, batch_size=args.batch_size,
                        predict_moves=bool(args.predict_moves), model_type=args.model,
                        bt4_weights=args.bt4_weights, bt4_init=args.bt4_init, bt4_history=args.bt4_history)
        if args.lr is not None:
            config.lr = args.lr
        if args.weight_decay is not None:
            config.weight_decay = args.weight_decay
    if args.lr_pretrained is not None:
        config.lr_pretrained = args.lr_pretrained          # stored in checkpoints for --resume
    lr_pretrained = getattr(config, "lr_pretrained", None) or config.lr
        
    # ====================== SEED AND PRECISION ======================
    torch.manual_seed(rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rank)
    torch.set_float32_matmul_precision("high")

    # ====================== LOGGING ======================
    if master_process:
        logging_path.mkdir(parents=True, exist_ok=True)
        train_writer = SummaryWriter(logging_path / "train")
        validation_writer = SummaryWriter(logging_path / "validation")

        if not continue_from_checkpoint:
            import yaml
            hparams = {
                "predict_moves": config.predict_moves,
                "masking_schedule": config.schedule,
                "n_heads": config.n_heads,
                "n_layers": config.n_layers,
                "embed_dim": config.embed_dim,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
                "batch_size": config.batch_size,
                "n_steps": config.n_steps,
                "use_context": config.use_context,
                "model_type": getattr(config, "model_type", "masked_diffusion"),
                "bt4_init": getattr(config, "bt4_init", None),
                "bt4_history": getattr(config, "bt4_history", None),
                "lr_pretrained": lr_pretrained,
                "warmup_steps": args.warmup_steps,
                "world_size": world_size,
            }
            with open(logging_path / "config.yml", "w") as f:
                yaml.dump(hparams, f)

    # ====================== DATASET ======================
    dataset_path = Path(args.dataset_dir)
    trainset = torch.load(dataset_path / "trainset.pt", weights_only=False, map_location="cpu")
    testset = torch.load(dataset_path / "testset.pt", weights_only=False, map_location="cpu")
    trainsampler = DistributedSampler(trainset, shuffle=True, rank=rank, num_replicas=world_size) if distributed else None
    validationsampler = DistributedSampler(testset, shuffle=True, rank=rank, num_replicas=world_size) if distributed else None
    assert config.batch_size % world_size == 0
    local_batch_size = config.batch_size // world_size
    num_workers = args.num_workers if args.num_workers is not None else (10 if distributed else 0)
    trainloader = DataLoader(trainset, batch_size=local_batch_size, shuffle=not distributed, sampler=trainsampler, num_workers=num_workers, pin_memory=distributed, persistent_workers=num_workers > 0, drop_last=True)
    validationloader = DataLoader(testset, batch_size=8 * local_batch_size if config.model_type == "masked_diffusion" else 2 * local_batch_size, shuffle=not distributed, sampler=validationsampler, num_workers=num_workers, pin_memory=distributed)

    # ====================== MODEL ======================
    model = build_model(config, load_weights=not continue_from_checkpoint)
    if continue_from_checkpoint:
        state_dict = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in checkpoint["model"].items()}
        model.load_state_dict(state_dict)
    model.to(device=device)
    pretrained_names = (model.pretrained_parameter_names()
                        if getattr(config, "model_type", "") == "bt4" and getattr(config, "bt4_init", "") == "lc0" else set())
    if master_process:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[train] {config.model_type}{' (' + config.bt4_init + ' init)' if config.model_type == 'bt4' else ''} params={n_params / 1e6:.1f}M "
              f"world_size={world_size} global_batch={config.batch_size} predict_moves={config.predict_moves}", flush=True)

    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    if args.compile:
        model = torch.compile(model)

    # ====================== OPTIMIZER ======================
    named = [(n.removeprefix("_orig_mod.").removeprefix("module."), p) for n, p in model.named_parameters()]
    pre = [p for n, p in named if n in pretrained_names]
    new = [p for n, p in named if n not in pretrained_names]
    groups = [{"params": new, "lr": config.lr}]
    if pre:
        groups.append({"params": pre, "lr": lr_pretrained})
    optimizer = AdamW(groups, weight_decay=config.weight_decay, fused=device_type == "cuda")
    start_step = checkpoint["step"] if continue_from_checkpoint else 0
    warmup = args.warmup_steps

    def lr_factor(i):
        # linear warmup from 1%, then cosine decay to 10% (same shape as the original schedule)
        s = i + start_step
        if s < warmup:
            return 0.01 + 0.99 * s / warmup
        progress = min(1.0, (s - warmup) / max(1, config.n_steps - warmup))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    if continue_from_checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception as e:
            if master_process: print(f"[train] could not restore optimizer state: {e}", flush=True)
        for g, base in zip(optimizer.param_groups, [config.lr, lr_pretrained]):
            g["lr"] = base
            g["initial_lr"] = base
    lr_scheduler = LambdaLR(optimizer, lr_factor)

    # ====================== LOSS FUNCTION ======================
    def compute_loss(model: MaskedDiffusion, fens, moves, themes=None, ratings=None):
        # could use the variance reduced version in rl.espo, but for supervised learning, this is good enough (variance is not a problem)
        tokens = torch.cat([fens, moves], dim=1) if config.predict_moves else fens
        batch_size = len(fens)
        t = (torch.rand(1) + torch.arange(1, batch_size + 1, 1) / batch_size) % 1
        if master_process and model.training and step % config.train_logging_interval == 0:
            train_writer.add_scalars("t", {"min": t.min(), "max": t.max(), "mean": t.mean()}, step)
            train_writer.add_scalars("alpha_t", {"min": config.masking_schedule(t).min(), "max": config.masking_schedule(t).max(), "mean": config.masking_schedule(t).mean()}, step)
            train_writer.add_scalars("t_weight", {"min": config.masking_schedule.get_weight(t).min(), "max": config.masking_schedule.get_weight(t).max(), "mean": config.masking_schedule.get_weight(t).mean()}, step)
        alpha_t = config.masking_schedule(t).unsqueeze(1).to(device)

        random_mask = torch.rand(tokens.size(), device=device) < alpha_t
        masked_tokens = torch.where(random_mask, tokens, config.mask_token)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            if config.use_context:
                logits = model(masked_tokens, themes, ratings, checkpoint_activations=bool(args.checkpoint_activations))
            else:
                logits = model(masked_tokens, checkpoint_activations=bool(args.checkpoint_activations))
        unwrapped_model = unwrap(model)
        return unwrapped_model.elbo_loss(t, logits, tokens, masked_tokens)

    def unwrap(m):
        m = getattr(m, "_orig_mod", m)
        return m.module if isinstance(m, DistributedDataParallel) else m

    # ====================== VALIDATION ======================
    def write_logits(step):
        seq_len = config.fen_length + config.move_length if config.predict_moves else config.fen_length
        probs_sum = torch.zeros((seq_len, config.n_tokens), dtype=torch.float32, device=device)
        samples = torch.tensor(0, device=device)
        for validation_fen, validation_move, validation_theme, validation_rating in validationloader:
            validation_fen, validation_move, validation_theme, validation_rating = validation_fen.to(dtype=torch.long, device=device), validation_move.to(dtype=torch.long, device=device), validation_theme.to(dtype=torch.float32, device=device), validation_rating.to(dtype=torch.float32, device=device)
            tokens = torch.cat([validation_fen, validation_move], dim=1) if config.predict_moves else validation_fen
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    if config.use_context:
                        sub_logits = model(tokens, validation_theme, validation_rating)
                    else:
                        sub_logits = model(tokens)
                sub_probs = F.softmax(sub_logits, dim=2).to(dtype=torch.float32)
                probs_sum += sub_probs.sum(dim=0)
                samples += len(validation_fen)
        if distributed:
            all_reduce(probs_sum, op=ReduceOp.SUM)
            all_reduce(samples, op=ReduceOp.SUM)
        
        avg_probs = probs_sum / samples

        if master_process: validation_writer.add_image("Probabilities", avg_probs.unsqueeze(0), step)

    def write_fen(step):
        unwrapped_model = unwrap(model)
        if config.use_context:
            validation_themes = torch.zeros((config.n_validation_generations, config.n_themes), dtype=torch.float32, device=device)
            indices = torch.randint(0, config.n_themes, (config.n_validation_generations,))
            validation_themes[:, indices] = 1
            validation_ratings = scale_ratings(3000 * torch.rand(config.n_validation_generations, dtype=torch.float32, device=device) + 300)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                tokens = unwrapped_model.sample(validation_themes, validation_ratings, steps=512)
        else:
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                tokens = unwrapped_model.sample(batch_size=config.n_validation_generations, steps=512)
        for generated_tokens in tokens:
            try:
                fen_str = tokens_to_fen(generated_tokens[:config.fen_length])
                if config.predict_moves:
                    move_str = tokens_to_move(generated_tokens[config.fen_length:])
                    if master_process: validation_writer.add_text("Generations/fen", f"{fen_str} move: {move_str}", step)
                else:
                    if master_process: validation_writer.add_text("Generations/fen", fen_str, step)
            except:
                pass

    def compute_validation_loss():
        total = torch.zeros(2, dtype=torch.float32, device=device)
        with torch.no_grad():
            for validation_fen, validation_move, validation_theme, validation_rating in validationloader:
                validation_fen, validation_move, validation_theme, validation_rating = validation_fen.to(dtype=torch.long, device=device), validation_move.to(dtype=torch.long, device=device), validation_theme.to(dtype=torch.float32, device=device), validation_rating.to(dtype=torch.float32, device=device)
                if config.use_context:
                    validation_loss = compute_loss(model, validation_fen, validation_move, validation_theme, validation_rating).sum()
                else:
                    validation_loss = compute_loss(model, validation_fen, validation_move, None, None).sum()
                total[0] += validation_loss.detach()
                total[1] += len(validation_fen)
        if distributed:
            all_reduce(total, op=ReduceOp.SUM)
        return total[0] / total[1]

    def write_validation_loss(step):
        validation_loss = compute_validation_loss()
        if master_process:
            validation_writer.add_scalar("Loss", validation_loss, step)
            print(f"[val] step {step} validation loss {float(validation_loss):.4f}", flush=True)

    def save_state():
        checkpoint_path = logging_path / f"model_{step:07d}.pt"
        base_model = unwrap(model)
        checkpoint = {
            "model": base_model.state_dict(),
            "config": config,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step
        }
        torch.save(checkpoint, checkpoint_path)
        if args.keep_checkpoints > 0:
            old = sorted(logging_path.glob("model_*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
            for p in old[:-args.keep_checkpoints]:
                p.unlink(missing_ok=True)

    def evaluate_puzzles(step):
        model.eval()
        local_n = args.n_evaluation_positions // world_size
        if rank == 0:
            local_n += args.n_evaluation_positions % world_size
        
        n_jobs = args.eval_cpus or int(os.environ.get("SLURM_CPUS_PER_GPU", max(1, (os.cpu_count() or 8) // max(1, int(os.environ.get("LOCAL_WORLD_SIZE", world_size))) - 2)))
        stockfish_path = Path(args.stockfish)
        engine_pool = queue.Queue()
        for _ in range(n_jobs):
            engine = SimpleEngine.popen_uci(stockfish_path)
            engine.configure({"Threads": 1, "Hash": 32})
            engine_pool.put(engine)

        def process_position(fen_tokens):
            engine = engine_pool.get()
            try:
                fen = tokens_to_fen(fen_tokens.cpu())
                if not legal(fen):
                    return False, False, False
                engine.configure({"Clear Hash": None})
                is_counter_intuitive = counter_intuitive(fen, engine)
                puzzle = get_unique_puzzle_from_fen(fen, engine)
                is_unique = (puzzle is not None)
                return True, is_unique, (is_unique and is_counter_intuitive)
            except Exception:
                return False, False, False
            finally:
                engine_pool.put(engine)

        generated_count = 0
        local_legal = 0
        local_unique = 0
        local_counter_intuitive_unique = 0
        
        unwrapped_model = unwrap(model)
        batch_size = 1024 if config.model_type == "masked_diffusion" else 512
        
        while generated_count < local_n:
            b = min(batch_size, local_n - generated_count)
            
            if config.use_context:
                indices = torch.randint(0, len(testset), (b,)).tolist()
                sampled_items = [testset[idx] for idx in indices]
                themes_one_hot = torch.stack([torch.as_tensor(item[2]) for item in sampled_items]).to(device=device, dtype=torch.float32)
                scaled_ratings = torch.stack([torch.as_tensor(item[3]) for item in sampled_items]).to(device=device, dtype=torch.float32)
            else:
                themes_one_hot = None
                scaled_ratings = None
                
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    tokens = unwrapped_model.sample(themes_one_hot, scaled_ratings, batch_size=b, steps=256)
            
            if config.predict_moves:
                fen_tokens = tokens[:, :config.fen_length]
            else:
                fen_tokens = tokens
                
            batch_results = Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(process_position)(fen_tokens[i]) for i in range(b)
            )
            
            for is_legal, is_unique, is_ci_unique in batch_results:
                if is_legal:
                    local_legal += 1
                if is_unique:
                    local_unique += 1
                if is_ci_unique:
                    local_counter_intuitive_unique += 1
                    
            generated_count += b

        while not engine_pool.empty():
            engine = engine_pool.get()
            engine.quit()

        counts = torch.tensor([local_n, local_legal, local_unique, local_counter_intuitive_unique], dtype=torch.float32, device=device)
        if distributed:
            all_reduce(counts, op=ReduceOp.SUM)
            
        if master_process:
            total_gen = counts[0].item()
            total_legal = counts[1].item()
            total_unique = counts[2].item()
            total_ci_unique = counts[3].item()
            
            legal_rate = total_legal / total_gen if total_gen > 0 else 0.0
            uniqueness_rate = total_unique / total_gen if total_gen > 0 else 0.0
            counter_intuitiveness_rate = total_ci_unique / total_unique if total_unique > 0 else 0.0
            puzzle_rate = total_ci_unique / total_gen if total_gen > 0 else 0.0
            
            validation_writer.add_scalar("Metrics/legal_rate", legal_rate, step)
            validation_writer.add_scalar("Metrics/uniqueness_rate", uniqueness_rate, step)
            validation_writer.add_scalar("Metrics/counter_intuitiveness_rate_given_unique", counter_intuitiveness_rate, step)
            validation_writer.add_scalar("Metrics/puzzle_rate", puzzle_rate, step)
            print(f"[eval] step {step} legal {legal_rate:.4f} unique {uniqueness_rate:.4f} "
                  f"counter-intuitive|unique {counter_intuitiveness_rate:.4f} puzzle {puzzle_rate:.4f} "
                  f"({int(total_gen)} positions)", flush=True)
            
        model.train()

    # ====================== TRAINING LOOP ======================
    step = 0
    epoch = 0
    if continue_from_checkpoint: step = checkpoint["step"]
    if continue_from_checkpoint: epoch = checkpoint["epoch"]

    if not continue_from_checkpoint and not args.skip_initial_eval: evaluate_puzzles(step)
    ended = False
    total = torch.zeros(3, dtype=torch.float32, device=device)
    last_log = [step, time.time()]
    while not ended:
        epoch += 1
        if distributed:
            trainsampler.set_epoch(epoch)

        for fen, move, theme, rating in trainloader:
            fen, move, theme, rating = fen.to(dtype=torch.long, device=device), move.to(dtype=torch.long, device=device), theme.to(dtype=torch.float32, device=device), rating.to(dtype=torch.float32, device=device)

            optimizer.zero_grad()

            if config.use_context:
                loss = compute_loss(model, fen, move, theme, rating)
            else:
                loss = compute_loss(model, fen, move, None, None)
            loss.mean().backward()

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            step += 1

            total[0] += loss.detach().sum()
            total[1] += len(fen) * norm.detach()
            total[2] += len(fen)
            if step % config.train_logging_interval == 0 or step == 1:
                if distributed: all_reduce(total, op=ReduceOp.SUM)
                if master_process: train_writer.add_scalar("Loss", total[0] / total[2], step)
                if master_process: train_writer.add_scalar("Grad norm", total[1] / total[2], step)
                if master_process:
                    for gi, g in enumerate(optimizer.param_groups):
                        train_writer.add_scalar(f"lr/group{gi}", g["lr"], step)
                    now = time.time()
                    if step % (config.train_logging_interval * 10) == 0 or step == 1:
                        sps = (step - last_log[0]) / max(1e-9, now - last_log[1])
                        print(f"[train] step {step} loss {(total[0] / total[2]).item():.4f} grad_norm {(total[1] / total[2]).item():.3f} "
                              f"lr {optimizer.param_groups[0]['lr']:.2e} {sps:.2f} it/s", flush=True)
                        last_log[0], last_log[1] = step, now
                total = torch.zeros(3, dtype=torch.float32, device=device)

            # validation
            if step % config.validation_interval == 0 or step == 1:
                model.eval()
                if distributed:
                    validationsampler.set_epoch(step // config.validation_interval)

                write_logits(step)
                write_fen(step)
                write_validation_loss(step)

                model.train()
            
            if step % args.evaluation_interval == 0:
                evaluate_puzzles(step)
            
            # create a checkpoint
            if step % config.save_interval == 0:
                if master_process:
                    save_state()

            # are we finished?
            if step >= config.n_steps:
                ended = True
                break

    model.eval()
    validation_loss = compute_validation_loss()
    if master_process:
        train_writer.add_hparams({
            "masking_schedule": config.schedule,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "embed_dim": config.embed_dim,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "batch_size": config.batch_size,
            "n_steps": config.n_steps,
            "validation_interval": config.validation_interval,
            "train_logging_interval": config.train_logging_interval,
            "save_interval": config.save_interval,
            "n_validation_generations": config.n_validation_generations,
            "use_context": config.use_context,
        }, {
            "validation_loss": validation_loss
        })

    if master_process: train_writer.close()
    if master_process: validation_writer.close()
    if distributed: destroy_process_group()


if __name__ == "__main__":
    main()
