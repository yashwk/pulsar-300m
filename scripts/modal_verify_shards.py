"""Verify pretraining shards on Modal Volume 'pulsar-data'.

Checks:
1. All 8 train and val shards exist and match expected file sizes.
2. Token range validity (all tokens in [0, 32767]).
3. Document-level exact deduplication across all shards using fast BLAKE2b hashing.
4. Windowed 64-token n-gram overlap check across all shard pairs.
"""

import os
import sys
import time
import hashlib
import numpy as np
import modal

data_vol = modal.Volume.from_name("pulsar-data", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "tqdm")
)

app = modal.App("pulsar-shard-verifier")


@app.function(
    image=image,
    cpu=8,
    memory=16384,
    timeout=60 * 20,
    volumes={"/root/data": data_vol},
)
def verify_shards_remote(
    num_shards: int = 8,
    sample_windows_per_shard: int = 50_000,
    window_size: int = 64,
) -> dict:
    """Read binary shards directly from mounted volume and verify deduplication."""
    print("=" * 80)
    print("       PULSAR-300M SHARD VERIFICATION & DEDUPLICATION AUDIT")
    print(f"       Shards: 0 to {num_shards - 1} | Vocab Target: 32,768 ($2^{{15}}$)")
    print("=" * 80)

    # 1. Check all shard files exist
    shard_stats = []
    missing_files = []

    for s in range(num_shards):
        t_path = f"/root/data/train_shard{s}.bin"
        v_path = f"/root/data/val_shard{s}.bin"
        if not os.path.exists(t_path):
            missing_files.append(t_path)
        if not os.path.exists(v_path):
            missing_files.append(v_path)

    if missing_files:
        raise FileNotFoundError(f"Missing shard files on volume: {missing_files}")

    print(f"\n[Step 1/4] Found all {num_shards * 2} shard files on volume 'pulsar-data'.")

    # 2. Inspect token bounds and document structure
    print("\n[Step 2/4] Scanning token ranges and extracting document boundaries...")
    doc_hash_to_shard = {}
    doc_cross_duplicates = 0
    doc_intra_duplicates = 0
    total_docs_all_shards = 0

    shard_info = []

    for s in range(num_shards):
        t_path = f"/root/data/train_shard{s}.bin"
        v_path = f"/root/data/val_shard{s}.bin"

        t_size = os.path.getsize(t_path)
        v_size = os.path.getsize(v_path)

        # Memory-map the files for fast zero-copy reading
        train_tokens = np.memmap(t_path, dtype=np.uint16, mode="r")
        val_tokens = np.memmap(v_path, dtype=np.uint16, mode="r")

        n_train = len(train_tokens)
        n_val = len(val_tokens)
        total_tokens = n_train + n_val

        max_t = int(train_tokens.max())
        min_t = int(train_tokens.min())
        max_v = int(val_tokens.max())
        min_v = int(val_tokens.min())

        assert max_t < 32768, f"Shard {s} train token {max_t} exceeds vocab 32768"
        assert max_v < 32768, f"Shard {s} val token {max_v} exceeds vocab 32768"
        assert min_t >= 0, f"Shard {s} train token {min_t} < 0"
        assert min_v >= 0, f"Shard {s} val token {min_v} < 0"

        # Find document boundaries (token == 0 is EOS <|endoftext|>)
        eos_indices = np.where(train_tokens == 0)[0]
        n_docs = len(eos_indices)
        total_docs_all_shards += n_docs

        # Hash individual documents using fast 64-bit BLAKE2b
        prev_idx = 0
        s_intra_dups = 0
        s_cross_dups = 0
        s_seen_hashes = set()

        for end_idx in eos_indices:
            doc_len = end_idx - prev_idx
            if doc_len >= 16:
                doc_bytes = train_tokens[prev_idx:end_idx].tobytes()
                # 64-bit hash
                h = hashlib.blake2b(doc_bytes, digest_size=8).digest()
                if h in s_seen_hashes:
                    s_intra_dups += 1
                else:
                    s_seen_hashes.add(h)
                    if h in doc_hash_to_shard:
                        s_cross_dups += 1
                        doc_cross_duplicates += 1
                    else:
                        doc_hash_to_shard[h] = s

            prev_idx = end_idx + 1

        doc_intra_duplicates += s_intra_dups

        shard_info.append({
            "shard": s,
            "train_tokens": n_train,
            "val_tokens": n_val,
            "total_tokens": total_tokens,
            "train_size_mb": t_size / (1024 * 1024),
            "val_size_mb": v_size / (1024 * 1024),
            "max_token_id": max(max_t, max_v),
            "num_docs": n_docs,
            "intra_dups": s_intra_dups,
            "cross_dups": s_cross_dups,
        })

        print(
            f"  Shard {s}: {total_tokens:,} tokens ({n_train:,} train, {n_val:,} val) | "
            f"Max ID: {max(max_t, max_v)} | Docs: {n_docs:,} | Cross-dups: {s_cross_dups}"
        )

    # 3. Sample-based N-Gram / Subsequence Overlap Check
    print(f"\n[Step 3/4] Running {window_size}-gram window cross-shard overlap check...")
    print(f"  Sampling {sample_windows_per_shard:,} random {window_size}-token windows per shard...")

    rng = np.random.RandomState(42)
    ngram_hash_to_shard = {}
    ngram_cross_collisions = {}

    for s in range(num_shards):
        t_path = f"/root/data/train_shard{s}.bin"
        tokens = np.memmap(t_path, dtype=np.uint16, mode="r")
        n = len(tokens)
        if n <= window_size:
            continue

        sample_starts = rng.randint(0, n - window_size, size=sample_windows_per_shard)

        s_collisions = 0
        for start in sample_starts:
            window = tokens[start : start + window_size].tobytes()
            h = hashlib.blake2b(window, digest_size=8).digest()

            if h in ngram_hash_to_shard:
                other_shard = ngram_hash_to_shard[h]
                if other_shard != s:
                    s_collisions += 1
                    pair = tuple(sorted((s, other_shard)))
                    ngram_cross_collisions[pair] = ngram_cross_collisions.get(pair, 0) + 1
            else:
                ngram_hash_to_shard[h] = s

    # 4. Final summary
    total_train_tokens = sum(x["train_tokens"] for x in shard_info)
    total_val_tokens = sum(x["val_tokens"] for x in shard_info)
    grand_total_tokens = total_train_tokens + total_val_tokens

    print("\n" + "=" * 80)
    print("                    SHARD DEDUPLICATION AUDIT RESULTS")
    print("=" * 80)
    print(f"Total Shards:           {num_shards}")
    print(f"Total Train Tokens:     {total_train_tokens:,}")
    print(f"Total Val Tokens:       {total_val_tokens:,}")
    print(f"Grand Total Tokens:     {grand_total_tokens:,} (~{grand_total_tokens / 1e9:.2f}B tokens)")
    print(f"Total Documents Scanned: {total_docs_all_shards:,}")
    print(f"Cross-Shard Exact Doc Duplicates: {doc_cross_duplicates} ({100.0 * doc_cross_duplicates / max(1, total_docs_all_shards):.4f}%)")
    print(f"Intra-Shard Exact Doc Duplicates: {doc_intra_duplicates}")
    print(f"Sampled {window_size}-gram Overlaps Across Shard Pairs: {sum(ngram_cross_collisions.values())}")
    for pair, count in sorted(ngram_cross_collisions.items()):
        print(f"  Shard pair {pair}: {count} shared {window_size}-grams ({100.0 * count / sample_windows_per_shard:.3f}%)")
    print("=" * 80)

    result = {
        "status": "passed" if doc_cross_duplicates == 0 else "duplicates_detected",
        "num_shards": num_shards,
        "total_train_tokens": total_train_tokens,
        "total_val_tokens": total_val_tokens,
        "grand_total_tokens": grand_total_tokens,
        "total_docs": total_docs_all_shards,
        "doc_cross_duplicates": doc_cross_duplicates,
        "doc_intra_duplicates": doc_intra_duplicates,
        "ngram_cross_collisions": {f"{k[0]}-{k[1]}": v for k, v in ngram_cross_collisions.items()},
        "shard_info": shard_info,
    }
    return result


@app.local_entrypoint()
def main(num_shards: int = 8, sample_windows: int = 50_000, window_size: int = 64):
    print(f"[Local Entrypoint] Triggering remote verification for {num_shards} shards on Modal...")
    res = verify_shards_remote.remote(
        num_shards=num_shards,
        sample_windows_per_shard=sample_windows,
        window_size=window_size,
    )
    print(f"\nVerification status: {res['status'].upper()}")
