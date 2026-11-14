"""Prepare immutable, batch-independent word-level chunks from the training split only."""
import argparse
from collections import Counter
import hashlib
import itertools
from pathlib import Path

import numpy as np

from .common import atomic_json, file_sha256


def words(dataset):
    for row in dataset:
        yield from row["text"].split()
        yield "<eos>"  # Includes empty rows; the same rule is applied to WT2 and WT103.


def vocabulary(tokens, size):
    counts = Counter(tokens)
    for special in ("<unk>", "<eos>"):
        counts.pop(special, None)
    ordered = sorted(counts, key=lambda token: (-counts[token], token))
    ordered = ordered if size == 0 else ordered[:size - 2]
    return {token: i for i, token in enumerate(["<unk>", "<eos>"] + ordered)}


def encode_to_file(dataset, vocab, path):
    count = unknown = 0
    with open(path, "wb") as f:
        stream = words(dataset)
        while batch := list(itertools.islice(stream, 100000)):
            ids = np.fromiter((vocab.get(w, 0) for w in batch), dtype="<u4")
            count += len(ids)
            unknown += int((ids == 0).sum())
            ids.tofile(f)
    return {"tokens": count, "unknown_tokens": unknown, "sha256": file_sha256(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--vocab-source", choices=["first-block", "full-train"], default="first-block")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    if args.seq_len <= 0 or (args.vocab_size != 0 and args.vocab_size < 2):
        parser.error("Invalid sequence length or vocabulary size")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error("Output directory must be empty; prepared datasets are immutable")
    from datasets import load_dataset
    from huggingface_hub import HfApi
    # Resolve the moving ref once, then use the exact same immutable revision for both corpora.
    revision = HfApi().dataset_info("Salesforce/wikitext", revision=args.revision).sha
    wt2 = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", revision=revision, split="train")
    wt103 = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", revision=revision)
    wt2_tokens = sum(1 for _ in words(wt2))
    block_tokens = (wt2_tokens // args.seq_len) * args.seq_len
    stream = words(wt103["train"])
    vocab_tokens = stream if args.vocab_source == "full-train" else itertools.islice(stream, block_tokens + 1)
    vocab = vocabulary(vocab_tokens, args.vocab_size)
    if args.vocab_size and len(vocab) != args.vocab_size:
        raise ValueError(f"Requested {args.vocab_size} vocabulary entries but found only {len(vocab)}")
    atomic_json(out / "vocab.json", vocab)
    train = encode_to_file(wt103["train"], vocab, out / "train.bin")
    valid = encode_to_file(wt103["validation"], vocab, out / "validation.bin")
    arr = np.memmap(out / "train.bin", dtype="<u4", mode="r")
    blocks = []
    for block_id in range((len(arr) - 1) // block_tokens):
        start = block_id * block_tokens
        chunk = arr[start:start + block_tokens + 1]
        blocks.append({"block_id": block_id, "start": start, "target_tokens": block_tokens,
                       "sha256": hashlib.sha256(chunk.tobytes()).hexdigest(),
                       "unknown_tokens": int((chunk[1:] == 0).sum()),
                       "unknown_fraction": float((chunk[1:] == 0).mean())})
    manifest = {
        "dataset": "Salesforce/wikitext", "revision": revision,
        "config": "wikitext-103-raw-v1", "block_reference": "wikitext-2-raw-v1",
        "tokenizer": "whitespace + EOS after every dataset row",
        "vocab_size": len(vocab), "vocab_source": args.vocab_source,
        "vocab_sha256": file_sha256(out / "vocab.json"),
        "seq_len": args.seq_len, "wikitext2_train_tokens_same_tokenizer": wt2_tokens,
        "block_target_tokens": block_tokens, "train": train, "validation": valid,
        "discarded_train_tail_targets": (len(arr) - 1) % block_tokens,
        "validation_targets": ((valid["tokens"] - 1) // args.seq_len) * args.seq_len,
        "discarded_validation_tail_targets": (valid["tokens"] - 1) % args.seq_len,
        "blocks": blocks,
        "boundary_rule": "Each block reuses one preceding input token; prediction targets never overlap. Sequence order is never shuffled.",
        "simulation_scope": "Ordered block availability, not timestamped production arrivals. Test split is unused."
    }
    if not blocks or not manifest["validation_targets"]:
        raise ValueError("Dataset too small")
    atomic_json(out / "manifest.json", manifest)
    print(f"Prepared {len(blocks)} blocks, {block_tokens:,} targets/block, {len(vocab):,} vocabulary entries", flush=True)


if __name__ == "__main__":
    main()
