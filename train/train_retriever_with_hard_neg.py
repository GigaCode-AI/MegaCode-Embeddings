"""
Train retriever on contrastive loss only with in-batch negatives and a single hard negative.
One can choose loss
"""
import json
import os
import logging
import time
import argparse
import random
import glob
import shutil
from contextlib import nullcontext
from datetime import datetime
from typing import List, Tuple, Union

import tqdm
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist
from transformers import AutoModel, AutoTokenizer

EPS = 1e-8


def str2bool(x: Union[str, bool]) -> bool:
    """
    For Python logger
    """
    if isinstance(x, bool):
        return x
    elif isinstance(x, str):
        if x.lower() == "true":
            return True
        elif x.lower() == "false":
            return False
        else:
            raise ValueError(f"invalid arg value: {x}")
    else:
        raise TypeError(f"invalid arg type: {type(x)}")


def init_process_group(backend: str = "nccl") -> Tuple[int, int, torch.device]:
    torch.distributed.init_process_group(backend=backend, init_method="env://")
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    return world_size, local_rank, device


def get_logger(name: str = None, level: Union[int, str] = logging.INFO, log_path: str = None):
    """
    Format taken from hydra
    """
    logger = logging.getLogger(name)
    formatter = logging.Formatter(fmt='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.setLevel(level)
    if log_path is not None:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    return logger


def get_paths(shards_mask: str) -> List[str]:
    paths = []
    for mask in shards_mask.split(","):
        paths_i = sorted(glob.glob(mask))
        assert len(paths_i) > 0, f"there are no paths matching mask {mask}"
        paths += paths_i
    return paths


def train_batches_gen(
        paths: List[str],
        tokenizer,
        batch_size: int = 16,
        buffer_size: int = 100000,
        max_tokens_query: int = 64,
        max_tokens_code: int = 1024,
        seed: int = 228,
        rank: int = 0,
        device: Union[str, torch.device] = "cpu"
):
    # setup som constants
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id

    # setup rng
    rng = random.Random(seed)

    # init buffers
    buffer = {
        "question": [],
        "pos": [],
        "neg": []
    }

    # init data streams
    def cycle(path):
        while True:
            with open(path) as f:
                for line in f:
                    yield line

    streams = [cycle(path) for path in paths]

    def ensure_eos(ids: List[int]):
        if ids[-1] != eos:
            return ids + [eos]
        return ids

    def yield_next() -> Tuple[List[int], List[int], List[int]]:
        x = json.loads(next(rng.choice(streams)))
        q = x["question"][:max_tokens_query - 1]
        p = rng.choice(x["positives"])["context"][:max_tokens_code - 1]
        n = rng.choice(x["negatives"])["context"][:max_tokens_code - 1]
        return ensure_eos(q), ensure_eos(p), ensure_eos(n)

    def pad_sequences(sequences: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = []
        attention_mask = []
        m = max(map(len, sequences))
        for i in range(len(sequences)):
            x = sequences[i]
            input_ids.append(x + [pad] * (m - len(x)))
            attention_mask.append([1] * len(x) + [0] * (m - len(x)))
        return torch.tensor(input_ids), torch.tensor(attention_mask)

    def sample():
        # sample batch indices
        idx = rng.sample(range(buffer_size), batch_size)

        # build batch
        questions = []
        pos = []
        neg = []
        for i in idx:
            questions.append(buffer["question"][i].copy())
            pos.append(buffer["pos"][i].copy())
            neg.append(buffer["neg"][i].copy())

        res = dict()
        res["input_ids_q"], res["attention_mask_q"] = pad_sequences(questions)
        res["input_ids_p"], res["attention_mask_p"] = pad_sequences(pos)
        res["input_ids_n"], res["attention_mask_n"] = pad_sequences(neg)
        res = {k: v.to(device) for k, v in res.items()}
        return res, idx

    def fill():
        for _ in tqdm.trange(buffer_size, desc=f"[rank {rank}] fill buffer", position=rank, leave=False):
            q, p, n = yield_next()
            buffer["question"].append(q)
            buffer["pos"].append(p)
            buffer["neg"].append(n)

    def update(idx: List[int]):
        for i in idx:
            q, p, n = yield_next()
            buffer["question"][i] = q
            buffer["pos"][i] = p
            buffer["neg"][i] = n

    fill()
    while True:
        batch, idx_ = sample()
        yield batch
        update(idx_)


def train_batches_gen_v2(
        paths: List[str],
        tokenizer,
        batch_size: int = 16,
        max_tokens_query: int = 64,
        max_tokens_code: int = 1024,
        seed: int = 228,
        rank: int = 0,
        device: Union[str, torch.device] = "cpu",
        **kwargs
):
    # setup some constants
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id

    # setup rng
    rng = random.Random(seed)

    # load all data
    data = []
    for path in tqdm.tqdm(paths, desc=f"[rank {rank}] read lines", position=rank, leave=False):
        with open(path) as f:
            for line in f:
                data.append(line)

    def ensure_eos(ids: List[int]):
        if ids[-1] != eos:
            return ids + [eos]
        return ids

    def pad_sequences(sequences: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = []
        attention_mask = []
        m = max(map(len, sequences))
        for i in range(len(sequences)):
            x = sequences[i]
            input_ids.append(x + [pad] * (m - len(x)))
            attention_mask.append([1] * len(x) + [0] * (m - len(x)))
        return torch.tensor(input_ids), torch.tensor(attention_mask)

    def sample():
        # build batch
        questions = []
        pos = []
        neg = []
        for line_ in rng.sample(data, batch_size):
            x = json.loads(line_)
            q = x["question"][:max_tokens_query - 1]
            p = rng.choice(x["positives"])["context"][:max_tokens_code - 1]
            n = rng.choice(x["negatives"])["context"][:max_tokens_code - 1]
            questions.append(ensure_eos(q))
            pos.append(ensure_eos(p))
            neg.append(ensure_eos(n))

        res = dict()
        res["input_ids_q"], res["attention_mask_q"] = pad_sequences(questions)
        res["input_ids_p"], res["attention_mask_p"] = pad_sequences(pos)
        res["input_ids_n"], res["attention_mask_n"] = pad_sequences(neg)
        res = {k: v.to(device) for k, v in res.items()}
        return res

    while True:
        batch = sample()
        yield batch


def describe(x):
    quantiles_list = [0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 0.999]
    quantiles = torch.tensor(quantiles_list, device=x.device)
    with torch.no_grad():
        # RuntimeError: quantile() input tensor must be either float or double dtype
        quantiles_values = torch.quantile(x.float(), quantiles).tolist()
        res = {
            "min": x.min(),
            "max": x.max(),
            "mean": x.mean()
        }
        for q, v in zip(quantiles_list, quantiles_values):
            res[f"q{q}"] = v
    return res


def clip_loss(q, p, n, tau: float, compute_stats: bool = False):
    """
    https://arxiv.org/pdf/2103.00020
    """
    # scores
    qp = q @ torch.cat([p, n], dim=0).T / tau

    # loss
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    labels = torch.arange(q.shape[0], device=q.device)
    loss_row = loss_fn(qp, labels)
    loss_col = loss_fn(qp[:, labels].T, labels)
    # Sum instead of average because row loss strongly dominates column loss due to hard negatives
    loss = loss_row + loss_col

    # stats
    stats = {}
    if compute_stats:
        stats.update({f"loss_{k}": v for k, v in describe(loss).items()})
        stats.update({f"loss_row_{k}": v for k, v in describe(loss_row).items()})
        stats.update({f"loss_col_{k}": v for k, v in describe(loss_col).items()})
    return loss.mean(), stats


def gte_loss(q, p, n, tau: float, compute_stats: bool = False):
    """
    https://arxiv.org/pdf/2308.03281
    """
    # scores
    qp = q @ torch.cat([p, n], dim=0).T
    qq = q @ q.T
    pp = p @ p.T

    # loss
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")  # for computing statistics
    labels = torch.arange(q.shape[0], device=q.device)
    mask_eye = torch.eye(q.shape[0], device=q.device, dtype=q.dtype)
    loss_q = loss_fn(torch.cat([qp, qq - mask_eye * 10000.0], dim=1) / tau, labels)
    loss_p = loss_fn(torch.cat([qp[:, labels].T, pp - mask_eye * 10000.0], dim=1) / tau, labels)
    # Also sum instead of average because loss_q strongly dominates loss_p
    loss = loss_q + loss_p

    # stats
    stats = {}
    if compute_stats:
        stats.update({f"loss_{k}": v for k, v in describe(loss).items()})
        stats.update({f"loss_q_{k}": v for k, v in describe(loss_q).items()})
        stats.update({f"loss_p_{k}": v for k, v in describe(loss_p).items()})
    return loss.mean(), stats


loss_map = {
    "clip": clip_loss,
    "gte": gte_loss
}


def main(args):
    # setup process group
    world_size, local_rank, device = init_process_group("nccl")

    # setup some constants
    num_negatives = 1  # hardcoded one hard negative
    mbs = args.per_device_train_batch_size  # batch that fits in the model (micro batch size)
    bs = mbs * args.gradient_accumulation_steps  # accumulated batch on one rank
    gbs = bs * world_size  # global batch

    # create output dir
    if local_rank == 0:
        print(args)
        assert not os.path.exists(args.output_dir), f"please rename or remove output dir: {args.output_dir}"
        now = datetime.now()
        log_dir = os.path.join(args.output_dir, "runs", now.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(log_dir)  # both output_dir and log_dir will be created
        writer = SummaryWriter(log_dir=log_dir)
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        logger = get_logger("training", logging.INFO, os.path.join(args.output_dir, "train.log"))
    else:
        logger = get_logger("training", logging.WARN)
        writer = None
    # dist.barrier()

    # setup logger
    logger.info(f"nce batch: {gbs} queries, {gbs * (1 + num_negatives)} candidates per query")

    def _all_gather(x):
        """
        In this script this aggregation is over tensors obtained in no_grad mode,
        so we don't need to worry about preserving gradients w.r.t. x
        """
        xs = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(xs, x)
        return torch.cat(xs, dim=0)

    # model and tokenizer
    model = AutoModel.from_pretrained(args.pretrained_dir, torch_dtype=torch.bfloat16).to(device)
    model = DistributedDataParallel(model)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_dir)

    def vectorize(input_ids, attention_mask):
        """Run forward and take eos vector"""
        outputs = model(input_ids, attention_mask)
        x = outputs.last_hidden_state
        xs = torch.arange(x.shape[0], device=device)
        ys = attention_mask.long().sum(1)
        return F.normalize(x[xs, ys - 1])

    # data pipeline nce
    # each process reads its subset of files
    paths = [x for i, x in enumerate(get_paths(args.train_data_path)) if i % world_size == local_rank]
    logger.info(f"num train paths: {len(paths)}")
    batches_gen_fn = train_batches_gen if args.buffer_size > 0 else train_batches_gen_v2
    batches_gen = batches_gen_fn(
        paths=paths,
        tokenizer=tokenizer,
        batch_size=mbs,
        buffer_size=args.buffer_size,
        max_tokens_query=args.max_tokens_query,
        max_tokens_code=args.max_tokens_code,
        seed=args.seed,
        rank=local_rank,
        device=device
    )

    # optimizer
    if args.zero2:
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.AdamW,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # scaler
    scaler = torch.amp.GradScaler("cuda", enabled=args.torch_dtype == "fp16")
    alias2dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32
    }
    dtype = alias2dtype[args.torch_dtype]

    # scheduler
    def _get_linear_schedule_with_warmup_lr_lambda(current_step):
        if current_step < args.num_warmup_steps:
            return float(current_step) / float(max(1, args.num_warmup_steps))
        return max(0.0, float(args.num_steps - current_step) / float(max(1, args.num_steps - args.num_warmup_steps)))

    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: _get_linear_schedule_with_warmup_lr_lambda(step))

    def maybe_write_scalar(*args):
        if writer is not None:
            writer.add_scalar(*args)

    def save(step: int):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))
        for name in ["config.json", "generation_config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"]:
            shutil.copyfile(os.path.join(args.pretrained_dir, name), os.path.join(checkpoint_dir, name))
        # if not args.save_only_model:
        #     if args.zero2:
        #         optimizer.consolidate_state_dict()
        #     torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))  # optimizer.consolidate_state_dict() in case of zero2
        #     torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
        logger.info(f"[step {global_step}] checkpoint saved to {checkpoint_dir}")

    # run!
    loss_fn = loss_map[args.loss_fn]
    t0 = time.time()  # initialize step start time
    pid = os.getpid()
    h = model.module.config.hidden_size
    grad_acc_steps = args.gradient_accumulation_steps
    for global_step in tqdm.trange(1, args.num_steps + 1, disable=local_rank != 0):
        # compute representations for global batch in inference mode
        model.eval()
        q = torch.zeros((bs, h), dtype=torch.float32, device=device)
        p = torch.zeros((bs, h), dtype=torch.float32, device=device)
        n = torch.zeros((bs * num_negatives, h), dtype=torch.float32, device=device)
        micro_batches = []
        for i in range(grad_acc_steps):
            b = next(batches_gen)
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                q[i * mbs:(i + 1) * mbs] = vectorize(b["input_ids_q"], b["attention_mask_q"])
                p[i * mbs:(i + 1) * mbs] = vectorize(b["input_ids_p"], b["attention_mask_p"])
                n[i * mbs * num_negatives:(i + 1) * mbs * num_negatives] = vectorize(b["input_ids_n"], b["attention_mask_n"])
            micro_batches.append(b)
        q = _all_gather(q)  # [gbs, d]
        p = _all_gather(p)  # [gbs, d]
        n = _all_gather(n)  # [gbs * num_negatives, d]

        # compute derivatives w.r.t. model
        # We multiply q by p.T grad_acc_steps times, but overall this is very small compared to the model forward
        model.train()
        o1 = bs * local_rank
        o2 = bs * local_rank * num_negatives
        stats_loss = {}  # to avoid PyCharm unused variable warning
        for i in range(grad_acc_steps):
            b = micro_batches[i]
            context = nullcontext if i == grad_acc_steps - 1 else model.no_sync
            with context():
                with torch.autocast("cuda", dtype=dtype):
                    qc = q.clone()
                    pc = p.clone()
                    nc = n.clone()
                    qc[i * mbs + o1:(i + 1) * mbs + o1] = vectorize(b["input_ids_q"], b["attention_mask_q"])
                    pc[i * mbs + o1:(i + 1) * mbs + o1] = vectorize(b["input_ids_p"], b["attention_mask_p"])
                    nc[i * mbs * num_negatives + o2:(i + 1) * mbs * num_negatives + o2] = vectorize(b["input_ids_n"], b["attention_mask_n"])
                    loss, stats = loss_fn(qc, pc, nc, tau=args.tau, compute_stats=i == 0)
                loss = loss * float(world_size)
                loss = scaler.scale(loss)
                loss.backward()
            # Same loss is computed at each step, so only need to compute statistics once
            if i == 0:
                stats_loss.update(stats)

        # update weights
        # 1. clip gradients: 1.1. divide by scale_factor  1.2. clip
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # 2. update weights (maybe)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        optimizer_was_run = scale_before <= scale_after
        model.zero_grad()

        # 3. if weights were updated, update scheduler
        if optimizer_was_run:
            scheduler.step()
        else:
            logger.warning(f"[rank {local_rank}, step {global_step}] step ignored. "
                           f"scale_before: {scale_before}; "
                           f"scale after: {scale_after}")

        # 4. log various training metrics
        global_step_time = time.time() - t0
        if (global_step == 1) or (global_step % args.log_steps == 0):  # want first log after first step
            for k, v in stats_loss.items():
                maybe_write_scalar(f"train/{k}", v, global_step)
            maybe_write_scalar("train/seconds_per_step", global_step_time, global_step)
            maybe_write_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
            x = os.popen(f"ps -o rss= {pid}").read().strip()
            ram = float(x) / 1024 / 1024
            maybe_write_scalar("train/ram_usage", ram, global_step)

        # save
        if (local_rank == 0) and (global_step % args.save_steps == 0):
            save(global_step)

        # update step start time
        t0 = time.time()

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    g = parser.add_argument_group("paths")
    g.add_argument("--train_data_path")
    g.add_argument("--pretrained_dir")
    g.add_argument("--output_dir")

    g = parser.add_argument_group("dataset")
    g.add_argument("--max_tokens_query", type=int, default=64)
    g.add_argument("--max_tokens_code", type=int, default=1024)
    g.add_argument("--seed", type=int, default=228)

    g = parser.add_argument_group("training_args")
    g.add_argument("--per_device_train_batch_size", type=int, default=16)
    g.add_argument("--num_steps", type=int, default=20000)
    g.add_argument("--num_warmup_steps", type=int, default=2000)
    g.add_argument("--gradient_accumulation_steps", type=int, default=1)
    g.add_argument("--save_steps", type=int, default=1000)
    g.add_argument("--log_steps", type=int, default=100)
    g.add_argument("--learning_rate", type=float, default=2e-5)
    g.add_argument("--weight_decay", type=float, default=0.01)
    g.add_argument("--max_grad_norm", type=float, default=1.0)
    g.add_argument("--buffer_size", type=int, default=100000)
    g.add_argument("--torch_dtype", type=str, choices=["fp16", "bf16", "fp32"], default="fp16")
    g.add_argument("--zero2", type=str2bool, default=False)

    g = parser.add_argument_group("loss")
    g.add_argument("--loss_fn", type=str, choices=["clip", "gte"], default="clip")
    g.add_argument("--tau", type=float, default=0.01)

    main(parser.parse_args())
