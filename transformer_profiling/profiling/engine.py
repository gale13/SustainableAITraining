"""Training worker. Imported only after CUDA_VISIBLE_DEVICES is set by its parent."""
from contextlib import nullcontext
import hashlib
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .common import atomic_json, export_block, write_csv
from .energy import Meter, Sensor
from .model import TransformerLM


def batches(tokens, seq_len, batch_size):
    """Every complete sequence appears once, including an incomplete final batch."""
    n = (len(tokens) - 1) // seq_len
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        yield (tokens[start * seq_len:stop * seq_len].view(-1, seq_len),
               tokens[start * seq_len + 1:stop * seq_len + 1].view(-1, seq_len))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_hash(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class Engine:
    def __init__(self, config, manifest, data_dir, batch_size, device="cuda:0"):
        self.config, self.manifest = config, manifest
        self.data_dir, self.batch_size = Path(data_dir), batch_size
        self.device = torch.device(device)
        self.seq_len = config["model"]["seq_len"]
        self.train_file = np.memmap(self.data_dir / "train.bin", dtype="<u4", mode="r")
        valid = np.memmap(self.data_dir / "validation.bin", dtype="<u4", mode="r")
        # Preload token IDs onto GPU outside training windows, ~17 MB for one WT2-sized block.
        self.validation = torch.from_numpy(np.array(valid, dtype=np.int64)).to(self.device)
        self.tokens_seen = self.steps = 0
        self.reset()

    def reset(self):
        # Seed before CPU initialization so all GPU workers begin with identical parameters.
        seed_all(self.config["seed"])
        self.model = TransformerLM(self.manifest["vocab_size"], **self.config["model"]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"],
                                          betas=(0.9, 0.999), eps=1e-8, weight_decay=0, foreach=False)
        self.tokens_seen = self.steps = 0

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def context(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16) if self.config["precision"] == "bf16" else nullcontext()

    def block(self, block_id):
        meta = self.manifest["blocks"][block_id]
        data = self.train_file[meta["start"]:meta["start"] + meta["target_tokens"] + 1]
        if hashlib.sha256(data.tobytes()).hexdigest() != meta["sha256"]:
            raise ValueError(f"Block {block_id} checksum mismatch")
        return torch.from_numpy(np.array(data, dtype=np.int64)).to(self.device)

    def step(self, x, y):
        if "warmup_steps" in self.config:
            warmup, position = self.config["warmup_steps"], self.steps + 1
        else:
            warmup, position = self.config["warmup_tokens"], self.tokens_seen + y.numel()
        rate = self.config["learning_rate"] * (min(1.0, position / warmup) if warmup else 1.0)
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        self.optimizer.zero_grad(set_to_none=True)
        with self.context():
            logits = self.model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip"], error_if_nonfinite=True)
        self.optimizer.step()
        self.tokens_seen += y.numel()
        self.steps += 1
        return loss.detach()

    def warmup(self, tokens):
        self.model.train()
        x, y = next(batches(tokens, self.seq_len, self.batch_size))
        start = time.perf_counter()
        for step in range(1000000):
            self.step(x, y)
            self.sync()
            if step >= 9 and time.perf_counter() - start >= self.config["warmup_seconds"]:
                break
        # Warmup must not change the formal experiment's initial model, optimizer or RNG state.
        self.reset()
        self.sync()

    def epoch(self, tokens, train=True):
        self.model.train(train)
        total = torch.zeros((), device=self.device, dtype=torch.float64)
        count = 0
        bs = self.batch_size if train else self.config["eval_batch"]
        with nullcontext() if train else torch.no_grad():
            for x, y in batches(tokens, self.seq_len, bs):
                if train:
                    loss = self.step(x, y)
                else:
                    with self.context():
                        logits = self.model(x)
                        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                total += loss.detach().double() * y.numel()
                count += y.numel()
        self.sync()
        mean = float(total.item()) / count
        if not math.isfinite(mean):
            raise FloatingPointError("Nonfinite loss; experiment stopped")
        return mean, count

    def measured_epoch(self, tokens, meter, block_id, epoch, epochs_per_block, run_id):
        start_wall = time.perf_counter()
        previous_steps = self.steps
        self.sync()
        meter.begin("train", block_id, epoch)
        train_loss, count = self.epoch(tokens)
        train = meter.end()
        self.sync()
        meter.begin("validation", block_id, epoch)
        val_loss, _ = self.epoch(self.validation, train=False)
        validation = meter.end()
        row = {"epoch_id": f"b{block_id:04d}_e{epoch:05d}", "run_id": run_id,
               "batch_sequences": self.batch_size, "block_id": block_id,
               "epoch_in_block": epoch, "global_epoch": block_id * epochs_per_block + epoch,
               "train_loss": train_loss, "validation_loss": val_loss,
               "validation_perplexity": math.exp(val_loss) if val_loss < 700 else None,
               "train_tokens": count, "unknown_fraction": self.manifest["blocks"][block_id]["unknown_fraction"],
               "optimizer_steps": self.steps - previous_steps,
               "global_steps": self.steps, "global_tokens": self.tokens_seen,
               "learning_rate": self.optimizer.param_groups[0]["lr"],
               "tokens_per_second": count / train["seconds"],
               "epoch_energy_j": train["energy_j"] + validation["energy_j"],
               "epoch_energy_wh": train["energy_wh"] + validation["energy_wh"],
               "epoch_wall_seconds": time.perf_counter() - start_wall}
        for prefix, values in [("train", train), ("validation", validation)]:
            row.update({f"{prefix}_{k}": v for k, v in values.items()})
        return row

    def save(self, path, completed_blocks, fingerprint):
        self.sync()
        state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "steps": self.steps, "tokens_seen": self.tokens_seen, "completed_blocks": completed_blocks,
                 "fingerprint": fingerprint, "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
                 "python_rng": random.getstate(), "numpy_rng": np.random.get_state()}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def restore(self, path, fingerprint):
        # Only load self-created local checkpoints; these include trusted optimizer/RNG objects.
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["fingerprint"] != fingerprint:
            raise ValueError("Checkpoint configuration/data fingerprint mismatch")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.steps, self.tokens_seen = state["steps"], state["tokens_seen"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        return state["completed_blocks"]


def worker(connection, config, manifest, data_dir, output, gpu, batch_size, mode, plan, committed, fingerprint):
    """The coordinator commits a block only after every worker has durably saved it."""
    torch.set_num_threads(config["cpu_threads_per_worker"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in the worker; check its GPU allocation and CUDA-enabled PyTorch")
    torch.cuda.set_device(0)
    actual_uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
    if actual_uuid.lower() != gpu["uuid"].lower() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"Worker must see exactly its assigned GPU {gpu['uuid']}; found {actual_uuid}")
    if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported on this GPU")
    root = Path(output) / f"batch_{batch_size:03d}"
    root.mkdir(parents=True, exist_ok=True)
    attempt = str(time.time_ns())
    sensor = Sensor(gpu["uuid"])
    meter = Meter(sensor, root / "telemetry" / f"{attempt}.csv", config["power_interval_seconds"])
    try:
        engine = Engine(config, manifest, data_dir, batch_size)
        initial_hash = model_hash(engine.model)
        block0 = engine.block(0)
        meter.label("warmup")
        engine.warmup(block0)
        del block0
        torch.cuda.empty_cache()
        state_dir = root / "_state"
        cp_path = lambda k: state_dir / f"checkpoint_{k:04d}.pt"
        if mode == "run":
            if committed or cp_path(0).exists():
                if engine.restore(cp_path(committed), fingerprint) != committed:
                    raise ValueError("Checkpoint progress mismatch")
            else:
                engine.save(cp_path(0), 0, fingerprint)
        atomic_json(root / f"environment_{attempt}.json", {
            "gpu": gpu, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "arch_list": torch.cuda.get_arch_list(),
            "capability": torch.cuda.get_device_capability(), "parameter_count": sum(p.numel() for p in engine.model.parameters()),
            "initial_model_sha256": initial_hash, "precision": config["precision"], "tf32": False,
            "sensor_note": sensor.init_note, "batch_sequences": batch_size,
            "batch_tokens": batch_size * engine.seq_len, "pid": os.getpid(), "fingerprint": fingerprint})
        meter.label("coordinator_wait")
        connection.send({"event": "ready", "initial_hash": initial_hash})
        while True:
            command = connection.recv()
            if command["action"] == "stop":
                break
            block_id = command["block_id"]
            tokens = engine.block(block_id)
            # Only prune checkpoints older than the globally committed starting state.
            if mode == "run":
                for path in state_dir.glob("checkpoint_*.pt"):
                    if int(path.stem.split("_")[-1]) < block_id:
                        path.unlink()
            engine.sync()
            while time.perf_counter() < command["start_at"]:
                time.sleep(0.01)
            torch.cuda.reset_peak_memory_stats()
            rows = []
            start = time.perf_counter()
            epoch = 1
            epoch_limit = plan["epochs_per_block"] if mode == "run" else 1000000
            while epoch <= epoch_limit:
                row = engine.measured_epoch(tokens, meter, block_id, epoch, epoch_limit,
                                            os.environ.get("PROFILE_RUN_ID", Path(output).name))
                rows.append(row)
                # An atomic table after EVERY epoch preserves complete measurements on interruption.
                write_csv(root / "attempts" / f"block_{block_id:04d}_{attempt}.csv", rows)
                print(f"batch={batch_size} block={block_id} epoch={epoch} loss={row['train_loss']:.4f} "
                      f"train={row['train_seconds']:.2f}s energy={row['train_energy_j']:.1f}J", flush=True)
                if mode == "pilot" and epoch >= config["pilot_min_epochs"] and time.perf_counter() - start >= config["pilot_seconds"]:
                    break
                epoch += 1
            meter.label("checkpoint", block_id)
            peak = torch.cuda.max_memory_reserved()
            if peak > torch.cuda.get_device_properties(0).total_memory * config["memory_fraction_limit"]:
                raise RuntimeError("Measured memory headroom below configured requirement")
            if mode == "run":
                engine.save(cp_path(block_id + 1), block_id + 1, fingerprint)
                export_block(root, block_id, rows)
            else:
                result = {"batch_sequences": batch_size, "gpu_uuid": gpu["uuid"],
                          "train_epoch_seconds": [r["train_seconds"] for r in rows],
                          "validation_epoch_seconds": [r["validation_seconds"] for r in rows],
                          "peak_reserved_bytes": peak, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                          "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                          "initial_hash": initial_hash}
                atomic_json(root / "pilot.json", result)
            duration = time.perf_counter() - start
            del tokens
            meter.label("coordinator_wait", block_id)
            connection.send({"event": "done", "block_id": block_id, "block_wall_seconds": duration,
                             "train_block_seconds": sum(r["train_seconds"] for r in rows)})
    finally:
        meter.close()
