import copy
import hashlib
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from profiling.common import atomic_json, export_block, make_plan, read_json
from profiling.energy import integrate, summarize
from profiling.engine import Engine, batches, model_hash
from profiling.model import TransformerLM
from profiling.prepare import vocabulary, words


class MeterMathTests(unittest.TestCase):
    def sample(self, t, watts, counter=None):
        return {"monotonic_s": t, "power_w": watts, "energy_mj": counter}

    def test_nonuniform_power_integration_and_units(self):
        samples = [self.sample(0, 100), self.sample(2, 200), self.sample(5, 100)]
        self.assertEqual(integrate(samples), 750)
        result = summarize(samples, 1)
        self.assertEqual(result["energy_j"], 750)
        self.assertAlmostEqual(result["energy_wh"], 750 / 3600)
        self.assertEqual(result["mean_w"], 150)

    def test_counter_priority_and_stalled_fallback(self):
        a = [self.sample(0, 100, 2000), self.sample(10, 100, 1002000)]
        self.assertEqual(summarize(a, 1)["energy_method"], "nvml_counter")
        a[-1]["energy_mj"] = 1000
        self.assertEqual(summarize(a, 1)["energy_method"], "power_trapezoid")
        self.assertFalse(summarize(a, 1)["energy_valid"])

    def test_bad_timestamps(self):
        with self.assertRaises(ValueError):
            integrate([self.sample(1, 100), self.sample(0, 100)])


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_batch_partition_preserves_targets_and_tail(self):
        tokens = torch.arange(41)
        for batch_size in (1, 2, 3, 6, 20):
            paired = list(batches(tokens, 4, batch_size))
            self.assertTrue(torch.equal(torch.cat([y.flatten() for _, y in paired]), tokens[1:]))
            for x, y in paired:
                self.assertTrue(torch.equal(x + 1, y))
            self.assertEqual(len(paired), math.ceil(10 / batch_size))

    def test_causal_mask_and_parameter_count(self):
        model = TransformerLM(32, layers=2, d_model=16, d_ff=32, heads=4, seq_len=8, dropout=0).eval()
        x = torch.randint(32, (2, 8))
        modified = x.clone()
        modified[:, 5:] = (modified[:, 5:] + 1) % 32
        self.assertTrue(torch.allclose(model(x)[:, :5], model(modified)[:, :5], atol=1e-6))
        self.assertEqual(sum(p.numel() for p in TransformerLM(32768).parameters()), 13127168)

    def fixture(self, path):
        config = read_json(Path(__file__).parents[1] / "configs/default.json")
        config["model"] = dict(layers=2, d_model=16, d_ff=32, heads=4, seq_len=4, dropout=0.2)
        config["eval_batch"] = 3
        config["warmup_steps"] = 100
        ids = np.arange(81, dtype="<u4") % 32
        ids.tofile(path / "train.bin")
        ids[:25].tofile(path / "validation.bin")
        blocks = [dict(block_id=i, start=40 * i, target_tokens=40,
                       sha256=hashlib.sha256(ids[40 * i:40 * i + 41].tobytes()).hexdigest()) for i in range(2)]
        manifest = dict(vocab_size=32, blocks=blocks, block_target_tokens=40)
        return config, manifest

    def test_warmup_uses_optimizer_steps_across_block_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, manifest = self.fixture(Path(tmp))
            for bs in (1, 4):
                engine = Engine(config, manifest, tmp, bs, "cpu")
                for block_id in range(2):
                    for x, y in batches(engine.block(block_id), 4, bs):
                        engine.step(x, y)
                        self.assertAlmostEqual(engine.optimizer.param_groups[0]["lr"],
                                               config["learning_rate"] * engine.steps / 100)

    def test_checkpoint_resume_matches_uninterrupted_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            config, manifest = self.fixture(path)
            a = Engine(config, manifest, path, 3, device="cpu")
            a.epoch(a.block(0))
            a.save(path / "checkpoint.pt", 1, "same")
            loss_expected, _ = a.epoch(a.block(1))
            expected_hash, expected_steps = model_hash(a.model), a.steps
            b = Engine(config, manifest, path, 3, device="cpu")
            self.assertEqual(b.restore(path / "checkpoint.pt", "same"), 1)
            loss, count = b.epoch(b.block(1))
            self.assertEqual(loss, loss_expected)
            self.assertEqual(model_hash(b.model), expected_hash)
            self.assertEqual(b.steps, expected_steps)
            self.assertEqual(b.tokens_seen, 80)
            self.assertEqual(count, 40)
            with self.assertRaises(ValueError):
                b.restore(path / "checkpoint.pt", "different")

    def test_evaluation_loss_is_token_weighted_and_batch_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, manifest = self.fixture(Path(tmp))
            engine = Engine(config, manifest, tmp, 3, "cpu")
            loss_a, tokens = engine.epoch(engine.validation, train=False)
            engine.config["eval_batch"] = 1
            loss_b, _ = engine.epoch(engine.validation, train=False)
            self.assertAlmostEqual(loss_a, loss_b, places=6)
            self.assertEqual(tokens, 24)

    def test_only_training_tokens_enter_vocabulary(self):
        source = list(words([{"text": "a b a"}, {"text": ""}]))
        self.assertEqual(source, ["a", "b", "a", "<eos>", "<eos>"])
        vocab = vocabulary(source, 3)
        self.assertEqual(vocab, {"<unk>": 0, "<eos>": 1, "a": 2})
        self.assertEqual(vocab.get("future-word", 0), 0)


class PlanTests(unittest.TestCase):
    def test_shared_epoch_and_budget(self):
        config = read_json(Path(__file__).parents[1] / "configs/default.json")
        pilots = [dict(batch_sequences=b, train_epoch_seconds=[10, 11, 12],
                       validation_epoch_seconds=[1, 1, 1], peak_reserved_bytes=10,
                       total_memory_bytes=100) for b in config["batches"]]
        manifest = dict(block_target_tokens=2097152, blocks=list(range(40)))
        plan = make_plan(config, manifest, pilots)
        self.assertEqual(plan["paper_equivalent_epochs"], 25)
        self.assertEqual(plan["epochs_per_block"], 144)
        self.assertGreaterEqual(plan["predicted_fastest_train_block_seconds"], 1200)
        self.assertLessEqual(plan["predicted_total_hours_with_margin"], 24)
        pilots[0]["peak_reserved_bytes"] = 99
        with self.assertRaises(ValueError):
            make_plan(config, manifest, pilots)

    def test_infeasible_budget_rejected(self):
        config = read_json(Path(__file__).parents[1] / "configs/default.json")
        pilots = [dict(batch_sequences=b, train_epoch_seconds=[1 if b == 4 else 10000],
                       validation_epoch_seconds=[1], peak_reserved_bytes=10,
                       total_memory_bytes=100) for b in config["batches"]]
        with self.assertRaises(ValueError):
            make_plan(config, dict(block_target_tokens=2000000, blocks=[0]), pilots)


if __name__ == "__main__":
    unittest.main()
