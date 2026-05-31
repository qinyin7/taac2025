"""
Training & eval

- Loads MyDataset from $TRAIN_DATA_PATH and builds BaselineModel
- AdamW with param grouping; warmup + cosine LR scheduler
- Curriculum negatives & hard top-k schedule
- Periodic full-corpus retrieval eval: reports Hit@K for token==1 and action!=0
- Writes logs to train.log and TensorBoard; saves model.pt and ckpt.pt under $TRAIN_CKPT_PATH

Env vars: TRAIN_DATA_PATH, TRAIN_CKPT_PATH, TRAIN_LOG_PATH, TRAIN_TF_EVENTS_PATH

Usage:
    python train.py
    python train.py --resume /path/to/ckpt.pt
"""


from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import random
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset
from model import BaselineModel

SEED = 13


# ------------------------------
# Utilities
# ------------------------------

def set_seed(seed: int = SEED) -> None:
    """Set RNG seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------
# Checkpoint I/O
# ------------------------------

def save_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    extra: Dict | None = None,
) -> None:
    """Save model/optimizer/scheduler + RNG states for exact resumability."""
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    if extra is not None:
        ckpt["extra"] = extra
    torch.save(ckpt, path)


def load_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    device: str = "cpu",
):
    """Load states and best-effort restore RNG."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    try:
        rng = ckpt.get("rng", {})
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("python") is not None:
            random.setstate(rng["python"])
    except Exception as e:
        print(f"[Warn] Failed to restore RNG states: {e}")

    return ckpt


def append_jsonl(path: Path, obj: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def plot_training_curves(log_dir: Path | str) -> None:
    """Write a compact PNG summary of loss, LR, curriculum, and eval metrics."""
    log_dir = Path(log_dir)
    train_path = log_dir / "train.log"
    eval_path = log_dir / "eval.log"
    if not train_path.exists():
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable, skip curve plot: {e}")
        return

    train_rows = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                train_rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    eval_rows = []
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    eval_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not train_rows:
        return

    steps = [int(r["global_step"]) for r in train_rows]
    losses = [float(r["loss"]) for r in train_rows]
    window = max(1, min(200, len(losses) // 20 or 1))
    kernel = np.ones(window, dtype=np.float64) / window
    smooth = np.convolve(np.asarray(losses, dtype=np.float64), kernel, mode="valid")
    smooth_steps = steps[window - 1 :]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    ax.plot(steps, losses, alpha=0.25, linewidth=0.8, label="loss")
    ax.plot(smooth_steps, smooth, linewidth=1.8, label=f"loss_ma{window}")
    ax.set_title("Train Loss")
    ax.set_xlabel("global_step")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    lr_rows = [r for r in train_rows if "lr_group0" in r]
    if lr_rows:
        ax.plot([int(r["global_step"]) for r in lr_rows], [float(r["lr_group0"]) for r in lr_rows], label="lr_group0")
    hard_rows = [r for r in train_rows if "hard_topk" in r]
    if hard_rows:
        ax2 = ax.twinx()
        ax2.plot(
            [int(r["global_step"]) for r in hard_rows],
            [float(r["hard_topk"]) for r in hard_rows],
            color="tab:orange",
            alpha=0.7,
            label="hard_topk",
        )
        ax2.set_ylabel("hard_topk")
    ax.set_title("LR / Curriculum")
    ax.set_xlabel("global_step")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    epoch_rows = [r for r in eval_rows if "avg_epoch_loss" in r]
    if epoch_rows:
        ax.plot(
            [int(r["epoch"]) for r in epoch_rows],
            [float(r["avg_epoch_loss"]) for r in epoch_rows],
            marker="o",
            label="avg_epoch_loss",
        )
    ax.set_title("Epoch Loss")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    metric_rows = [r for r in eval_rows if "hit@10[token==1]" in r or "hit@10[action!=0]" in r]
    if metric_rows:
        x = [int(r["epoch"]) for r in metric_rows]
        if "hit@10[token==1]" in metric_rows[0]:
            ax.plot(x, [float(r.get("hit@10[token==1]", 0.0)) for r in metric_rows], marker="o", label="Hit@10 token")
        if "hit@10[action!=0]" in metric_rows[0]:
            ax.plot(x, [float(r.get("hit@10[action!=0]", 0.0)) for r in metric_rows], marker="o", label="Hit@10 action")
    ax.set_title("Epoch Eval")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    out_path = log_dir / "training_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Wrote {out_path}")


# ------------------------------
# Args
# ------------------------------

def get_args() -> argparse.Namespace:
    env = os.environ
    parser = argparse.ArgumentParser(description="MNS Baseline Trainer")

    # Paths
    parser.add_argument("--data_path", type=str, default=env.get("TRAIN_DATA_PATH"),
                        help="Dataset root path (env: TRAIN_DATA_PATH)")
    parser.add_argument("--ckpt_root", type=str, default=env.get("TRAIN_CKPT_PATH"),
                        help="Checkpoint root (env: TRAIN_CKPT_PATH)")
    parser.add_argument("--log_dir", type=str, default=env.get("TRAIN_LOG_PATH"),
                        help="Directory to write train.log (env: TRAIN_LOG_PATH)")
    parser.add_argument("--tf_dir", type=str, default=env.get("TRAIN_TF_EVENTS_PATH"),
                        help="TensorBoard events dir (env: TRAIN_TF_EVENTS_PATH)")

    # Train params
    parser.add_argument("--batch_size", default=232, type=int)
    parser.add_argument("--lr", default=8e-4, type=float)
    parser.add_argument("--maxlen", default=101, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--val_ratio", default=0.01, type=float, help="Validation split ratio (default 1%)")

    # Baseline Model construction
    parser.add_argument("--hidden_units", default=128, type=int)
    parser.add_argument("--num_blocks", default=16, type=int)
    parser.add_argument("--num_epochs", default=4, type=int)
    parser.add_argument("--num_heads", default=8, type=int)
    parser.add_argument("--drop_path_rate", default=0.0, type=float)
    parser.add_argument("--emb_dropout", default=0.2, type=float)
    parser.add_argument("--ffn_dropout", default=0.2, type=float)
    parser.add_argument("--attn_out_dropout", default=0.0, type=float)
    parser.add_argument("--sdpa_dropout", default=0.0, type=float)
    parser.add_argument("--l2_emb", default=0.0, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--norm_first", action="store_true")

    # Run modes
    parser.add_argument("--inference_only", action="store_true", help="Skip training loop")
    parser.add_argument("--save_final", action="store_true", help="Save a final checkpoint when training exits")
    parser.add_argument("--save_interval", default=0, type=int,
                        help="Save/overwrite ckpt_root/latest every N training steps; 0 disables periodic latest saves")
    parser.add_argument("--resume", default=None, type=str,
                        help="Path to ckpt.pt to resume")
    parser.add_argument("--state_dict_path", default=None, type=str,
                        help="(Optional) initialize model weights from a state_dict")

    # Eval params
    parser.add_argument("--eval_batch_size", default=512, type=int)
    parser.add_argument("--item_emb_batch_size", default=8192, type=int)
    parser.add_argument("--topk", nargs="+", default=[10, 100], type=int)
    parser.add_argument("--eval_each_epoch", action="store_true",
                        help="Run retrieval eval after every training epoch")
    parser.add_argument("--plot_curves", action="store_true",
                        help="Write log_dir/training_curves.png after epoch eval and final save")

    # MM embedding Feature IDs
    parser.add_argument("--mm_emb_id", nargs="+", default=[], type=str, choices=[str(s) for s in range(81, 87)])

    return parser.parse_args()


# ------------------------------
# Curriculum
# ------------------------------

def set_curriculum(global_step: int) -> Tuple[int, float]:
    """Return (hardtopk, glb_neg_ratio) according to global_step schedule."""
    if global_step < 20000:
        hardtopk = 0
    elif global_step < 40000:
        hardtopk = 8192
    elif global_step < 60000:
        hardtopk = 4096
    elif global_step < 80000:
        hardtopk = 2048
    elif global_step < 100000:
        hardtopk = 1024
    elif global_step < 115000:
        hardtopk = 512
    elif global_step < 130000:
        hardtopk = 256
    else:
        hardtopk = 128
    glb_neg_ratio = 0.5 if global_step < 80000 else 0.25
    return hardtopk, glb_neg_ratio


# ------------------------------
# Eval helpers
# ------------------------------

@torch.no_grad()
def precompute_item_embeddings(
    model: BaselineModel,
    dataset: MyDataset,
    device: str,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Vectorize all items [1..itemnum]; L2-normalize; return CPU tensor of shape (itemnum, D)."""
    model.eval()
    itemnum = dataset.itemnum
    all_embs: list[torch.Tensor] = []
    for start in tqdm(range(1, itemnum + 1, batch_size), desc="Precompute item embs"):
        end = min(start + batch_size - 1, itemnum)
        ids = np.arange(start, end + 1, dtype=np.int64)
        item_seq = torch.from_numpy(ids).to(device).unsqueeze(0)  # (1, M)

        feats = []
        for iid in ids:
            raw = dataset.item_feat_dict.get(str(int(iid)), None)
            feats.append(dataset.fill_missing_feat(raw, int(iid)))
        feats = np.array(feats, dtype=object)

        emb = model.feat2emb(item_seq, [feats], include_user=False).squeeze(0)  # (M, D)
        emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def batched_topk_hits(
    q_vecs: torch.Tensor,
    item_embs_cpu: torch.Tensor,
    gt_item_ids: torch.Tensor,
    device: str,
    ks: Iterable[int] = (10, 100),
    chunk_items: int = 65536,
) -> dict[int, int]:
    """Compute Hit@k for multiple k via full-corpus retrieval (GPU-chunked)."""
    ks = list(sorted(set(int(k) for k in ks)))
    assert len(ks) > 0
    k_max = max(ks)

    B, _ = q_vecs.shape
    I = item_embs_cpu.shape[0]
    q = q_vecs.to(device)

    top_scores = torch.full((B, 0), float("-inf"), device=device)
    top_indices = torch.empty((B, 0), dtype=torch.long, device=device)

    for start in range(0, I, chunk_items):
        end = min(start + chunk_items, I)
        cand = item_embs_cpu[start:end].to(device, non_blocking=True)  # (m, D)
        scores = q @ cand.t()  # (B, m)

        if top_indices.numel() == 0:
            vals, idx = torch.topk(scores, k=min(k_max, scores.size(1)), dim=1)
            top_scores = vals
            top_indices = idx + start
        else:
            comb_scores = torch.cat([top_scores, scores], dim=1)
            chunk_idx = torch.arange(start, end, device=device).view(1, -1).expand(B, -1)
            comb_indices = torch.cat([top_indices, chunk_idx], dim=1)
            vals, new_pos = torch.topk(comb_scores, k=min(k_max, comb_scores.size(1)), dim=1)
            row_ids = torch.arange(B, device=device).unsqueeze(1)
            new_idx = comb_indices[row_ids, new_pos]
            top_scores, top_indices = vals, new_idx

    pred_ids = top_indices + 1  # convert back to 1-based item ids
    hits_by_k: dict[int, int] = {}
    gt = gt_item_ids.view(-1, 1)
    for k in ks:
        hits_k = (pred_ids[:, :k] == gt).any(dim=1).sum().item()
        hits_by_k[k] = hits_k
    return hits_by_k


# ------------------------------
# Build components
# ------------------------------

def build_model(args: argparse.Namespace, dataset: MyDataset) -> BaselineModel:
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    args.use_item_id_emb = True
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    # (Optional) load external state_dict
    if args.state_dict_path:
        state = torch.load(args.state_dict_path, map_location=args.device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[Init] Loaded state_dict from {args.state_dict_path} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        # Initialize train-from-scratch weights
        def init_weights(m):
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        model.apply(init_weights)

    # Zero out padding rows
    with torch.no_grad():
        if hasattr(model, "pos_emb"):
            model.pos_emb.weight[0].zero_()
        if hasattr(model, "item_emb") and getattr(args, "use_item_id_emb", False):
            model.item_emb.weight[0].zero_()
        if hasattr(model, "user_emb"):
            model.user_emb.weight[0].zero_()
        if hasattr(model, "sparse_emb"):
            for k in model.sparse_emb:
                model.sparse_emb[k].weight[0].zero_()

    return model


def build_optimizer(model: nn.Module, base_lr: float) -> torch.optim.Optimizer:
    embed_w = [m.weight for m in model.modules() if isinstance(m, nn.Embedding) and m.weight.requires_grad]
    decay_w, no_decay_w = [], []

    for m in model.modules():
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            for p in m.parameters(recurse=False):
                if p.requires_grad:
                    no_decay_w.append(p)
        elif isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            if m.weight is not None and m.weight.requires_grad:
                decay_w.append(m.weight)
            if m.bias is not None and m.bias.requires_grad:
                no_decay_w.append(m.bias)

    captured = set(map(id, decay_w + no_decay_w + embed_w))
    leftover = [p for p in model.parameters() if p.requires_grad and id(p) not in captured]
    decay_w += leftover

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_w, "lr": base_lr, "weight_decay": 1e-5},
            {"params": no_decay_w, "lr": base_lr, "weight_decay": 0.0},
            {"params": embed_w, "lr": base_lr * 0.5, "weight_decay": 0.0},
        ]
    )
    return optimizer


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int = 130_000, warmup_ratio: float = 0.05):
    warmup_steps = max(1, int(warmup_ratio * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        min_ratio = 0.0002
        return min_ratio + 0.5 * (1 - min_ratio) * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_dataloaders(
    dataset: MyDataset,
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader]:
    n = len(dataset)
    train_len = int((1.0 - args.val_ratio) * n)
    lengths = [train_len, n - train_len]
    g = torch.Generator().manual_seed(SEED)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, lengths, generator=g)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=dataset.collate_fn,
        persistent_workers=False,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=dataset.collate_fn,
        persistent_workers=False,
        drop_last=False,
    )
    return train_loader, valid_loader


# ------------------------------
# Train / Eval loops
# ------------------------------

def evaluate_once(
    model: BaselineModel,
    dataset: MyDataset,
    valid_loader: DataLoader,
    device: str,
    item_emb_batch_size: int,
    ks: Iterable[int] = (10, 100),
) -> dict[str, float]:
    """Run full evaluation; return dict of Hit@K metrics for token/action masks."""
    model.eval()
    item_embs_cpu = precompute_item_embeddings(model, dataset, device, batch_size=item_emb_batch_size)

    hit_cnt_tok = {k: 0 for k in ks}
    hit_cnt_act = {k: 0 for k in ks}
    eval_cnt_tok, eval_cnt_act = 0, 0

    with torch.no_grad():
        for _, vbatch in tqdm(enumerate(valid_loader), total=len(valid_loader), desc="Eval"):
            (
                seq, pos, neg,
                token_type, next_token_type, next_action_type,
                seq_time, next_time,
                seq_feat, pos_feat, neg_feat,
            ) = vbatch

            seq = seq.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            # 1) User representation at last step
            q = model.predict(seq, seq_feat, token_type, seq_time, next_action_type)  # (B, D)

            # 2) Ground-truth next item (last position)
            gt = pos[:, -1].to(device)

            # 3) Two evaluation masks
            mask_tok = (next_token_type[:, -1].to(device) == 1) & (gt > 0)
            mask_act = (next_action_type[:, -1].to(device) != 0) & (gt > 0)

            # 4) token_type == 1
            if mask_tok.any():
                hits = batched_topk_hits(
                    q_vecs=q[mask_tok],
                    item_embs_cpu=item_embs_cpu,
                    gt_item_ids=gt[mask_tok],
                    device=device,
                    ks=ks,
                    chunk_items=65536,
                )
                for k in ks:
                    hit_cnt_tok[k] += hits[k]
                eval_cnt_tok += int(mask_tok.sum().item())

            # 5) action_type != 0
            if mask_act.any():
                hits = batched_topk_hits(
                    q_vecs=q[mask_act],
                    item_embs_cpu=item_embs_cpu,
                    gt_item_ids=gt[mask_act],
                    device=device,
                    ks=ks,
                    chunk_items=65536,
                )
                for k in ks:
                    hit_cnt_act[k] += hits[k]
                eval_cnt_act += int(mask_act.sum().item())

    # Aggregate
    result = {
        f"hit@{k}[token==1]": (hit_cnt_tok[k] / eval_cnt_tok) if eval_cnt_tok else 0.0
        for k in ks
    }
    result.update({
        f"hit@{k}[action!=0]": (hit_cnt_act[k] / eval_cnt_act) if eval_cnt_act else 0.0
        for k in ks
    })
    result["n[token==1]"] = float(eval_cnt_tok)
    result["n[action!=0]"] = float(eval_cnt_act)
    return result


def train(args: argparse.Namespace) -> None:
    set_seed(SEED)

    # Paths
    log_dir = ensure_dir(args.log_dir)
    tf_dir = ensure_dir(args.tf_dir)
    ckpt_root = ensure_dir(args.ckpt_root)
    data_path = args.data_path

    # Dataset / loaders
    dataset = MyDataset(data_path, args)
    train_loader, valid_loader = build_dataloaders(dataset, args)

    # Model / optimizer / scheduler
    model = build_model(args, dataset)
    optimizer = build_optimizer(model, args.lr)
    scheduler = build_scheduler(optimizer, total_steps=130_000, warmup_ratio=0.05)

    # State
    epoch_start_idx = 1
    global_step = 0
    best_hit = 0.0

    def save_run_checkpoint(save_dir: Path, epoch: int, step: int, extra: Dict | None = None) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")
        save_checkpoint(
            save_dir / "ckpt.pt",
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            global_step=step,
            args=args,
            extra=extra or {"best_hit": float(best_hit)},
        )

    # Resume
    if args.resume is not None:
        ckpt_path = args.resume
        if os.path.isdir(ckpt_path):
            ckpt_path = str(Path(ckpt_path) / "ckpt.pt")
        ckpt = load_checkpoint(ckpt_path, model, optimizer, scheduler, device=args.device)
        epoch_start_idx = ckpt.get("epoch", 1)
        global_step = ckpt.get("global_step", 0)
        best_hit = ckpt.get("extra", {}).get("best_hit", 0.0) if "extra" in ckpt else 0.0
        print(f"[Resume] {ckpt_path}: epoch={epoch_start_idx}, global_step={global_step}, best_hit={best_hit}")

    eval_interval = 30000
    current_epoch = epoch_start_idx
    print("Start training")

    log_mode = "a" if args.resume else "w"
    with open(Path(log_dir, "train.log"), log_mode) as log_file, SummaryWriter(tf_dir) as writer:
        try:
            for epoch in range(epoch_start_idx, args.num_epochs + 1):
                current_epoch = epoch
                if args.inference_only:
                    break

                model.train()
                epoch_loss_sum = 0.0
                epoch_loss_count = 0
                pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
                for _, batch in pbar:
                    optimizer.zero_grad(set_to_none=True)
                    (
                        seq, pos, neg,
                        token_type, next_token_type, next_action_type,
                        seq_time, next_time,
                        seq_feat, pos_feat, neg_feat,
                    ) = batch

                    seq = seq.to(args.device)
                    pos = pos.to(args.device)
                    neg = neg.to(args.device)

                    # Curriculum & eval scheduling
                    hardtopk, glb_neg_ratio = set_curriculum(global_step)
                    if 0 < hardtopk < 256:
                        eval_interval = 1000

                    loss = model(
                        seq,
                        pos,
                        neg,
                        token_type,
                        next_token_type,
                        next_action_type,
                        seq_time,
                        seq_feat,
                        pos_feat,
                        neg_feat,
                        hard_topk=hardtopk,
                        glb_neg_ratio=glb_neg_ratio,
                    )

                    # Logging
                    current_loss = float(loss.item())
                    current_lrs = {f"lr_group{i}": float(pg["lr"]) for i, pg in enumerate(optimizer.param_groups)}
                    log_obj = {
                        "global_step": global_step,
                        "loss": current_loss,
                        "epoch": epoch,
                        "time": time.time(),
                        "hard_topk": int(hardtopk),
                        "glb_neg_ratio": float(glb_neg_ratio),
                        **current_lrs,
                    }
                    line = json.dumps(log_obj)
                    log_file.write(line + "\n")
                    log_file.flush()
                    epoch_loss_sum += current_loss
                    epoch_loss_count += 1
                    pbar.set_postfix(loss=f"{current_loss:.4f}")
                    writer.add_scalar("Loss/train", current_loss, global_step)
                    for i, pg in enumerate(optimizer.param_groups):
                        writer.add_scalar(f"LR/group{i}", pg["lr"], global_step)

                    # Step
                    global_step += 1
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, foreach=False)
                    optimizer.step()
                    scheduler.step()

                    if args.save_interval > 0 and global_step % args.save_interval == 0:
                        latest_dir = Path(ckpt_root, "latest")
                        save_run_checkpoint(latest_dir, epoch=epoch, step=global_step)
                        print(f"Saved latest checkpoint to {latest_dir}")

                    # Periodic eval
                    if global_step % eval_interval == 0:
                        metrics = evaluate_once(
                            model=model,
                            dataset=dataset,
                            valid_loader=valid_loader,
                            device=args.device,
                            item_emb_batch_size=args.item_emb_batch_size,
                            ks=args.topk,
                        )

                        # TB scalars
                        for k in args.topk:
                            writer.add_scalar(f"Metric/Hit@{k}[token==1]", metrics[f"hit@{k}[token==1]"], global_step)
                            writer.add_scalar(f"Metric/Hit@{k}[action!=0]", metrics[f"hit@{k}[action!=0]"], global_step)

                        print(
                            f"Hit@{args.topk}[token==1]: "
                            + ", ".join(f"{k}={metrics[f'hit@{k}[token==1]']:.4f}" for k in args.topk)
                            + f" (n={int(metrics['n[token==1]'])}) | "
                            f"Hit@{args.topk}[action!=0]: "
                            + ", ".join(f"{k}={metrics[f'hit@{k}[action!=0]']:.4f}" for k in args.topk)
                            + f" (n={int(metrics['n[action!=0]'])})"
                        )

                        # Save
                        save_dir = Path(
                            ckpt_root,
                            f"global_step{global_step}"
                            + "".join(f".hit{k}_tok={metrics[f'hit@{k}[token==1]']:.4f}" for k in args.topk)
                            + "".join(f".hit{k}_act={metrics[f'hit@{k}[action!=0]']:.4f}" for k in args.topk)
                        )

                        best_hit = max(best_hit, float(metrics.get("hit@10[token==1]", 0.0)))
                        save_run_checkpoint(
                            save_dir,
                            epoch=epoch,
                            step=global_step,
                            extra={
                                **{f"hit{k}_tok": float(metrics[f"hit@{k}[token==1]"]) for k in args.topk},
                                **{f"hit{k}_act": float(metrics[f"hit@{k}[action!=0]"]) for k in args.topk},
                                "best_hit": float(best_hit),
                            },
                        )
                        if global_step < 135000:
                            model.train()

                avg_epoch_loss = epoch_loss_sum / max(epoch_loss_count, 1)
                writer.add_scalar("Loss/epoch_avg", avg_epoch_loss, epoch)
                eval_obj = {
                    "event": "epoch_end",
                    "epoch": epoch,
                    "global_step": global_step,
                    "avg_epoch_loss": float(avg_epoch_loss),
                    "time": time.time(),
                }
                if args.eval_each_epoch:
                    metrics = evaluate_once(
                        model=model,
                        dataset=dataset,
                        valid_loader=valid_loader,
                        device=args.device,
                        item_emb_batch_size=args.item_emb_batch_size,
                        ks=args.topk,
                    )
                    eval_obj.update({k: float(v) for k, v in metrics.items()})
                    for k in args.topk:
                        writer.add_scalar(f"EpochMetric/Hit@{k}[token==1]", metrics[f"hit@{k}[token==1]"], epoch)
                        writer.add_scalar(f"EpochMetric/Hit@{k}[action!=0]", metrics[f"hit@{k}[action!=0]"], epoch)
                    print(
                        f"[Epoch {epoch}] avg_loss={avg_epoch_loss:.4f} | "
                        f"Hit@{args.topk}[token==1]: "
                        + ", ".join(f"{k}={metrics[f'hit@{k}[token==1]']:.4f}" for k in args.topk)
                        + " | "
                        f"Hit@{args.topk}[action!=0]: "
                        + ", ".join(f"{k}={metrics[f'hit@{k}[action!=0]']:.4f}" for k in args.topk)
                    )
                    best_hit = max(best_hit, float(metrics.get("hit@10[token==1]", 0.0)))
                    epoch_dir = Path(ckpt_root, f"epoch{epoch}_global_step{global_step}")
                    save_run_checkpoint(
                        epoch_dir,
                        epoch=epoch,
                        step=global_step,
                        extra={
                            **{f"hit{k}_tok": float(metrics[f"hit@{k}[token==1]"]) for k in args.topk},
                            **{f"hit{k}_act": float(metrics[f"hit@{k}[action!=0]"]) for k in args.topk},
                            "avg_epoch_loss": float(avg_epoch_loss),
                            "best_hit": float(best_hit),
                        },
                    )
                    model.train()
                append_jsonl(Path(log_dir, "eval.log"), eval_obj)
                if args.plot_curves:
                    plot_training_curves(log_dir)
        except KeyboardInterrupt:
            latest_dir = Path(ckpt_root, "latest")
            save_run_checkpoint(latest_dir, epoch=current_epoch, step=global_step)
            print(f"\nInterrupted. Saved latest checkpoint to {latest_dir}")
            raise

        if args.save_final:
            save_dir = Path(ckpt_root, f"final_global_step{global_step}")
            save_run_checkpoint(save_dir, epoch=args.num_epochs, step=global_step)
            print(f"Saved final checkpoint to {save_dir}")
        if args.plot_curves:
            plot_training_curves(log_dir)

        print("Done")


# ------------------------------
# Entry
# ------------------------------

if __name__ == "__main__":
    args = get_args()
    train(args)
