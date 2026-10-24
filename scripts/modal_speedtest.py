"""Benchmark real-world Hugging Face download speed inside Modal worker."""

import time
import modal

app = modal.App("pulsar-hf-speedtest")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]", "tqdm")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_XET_HIGH_PERFORMANCE": "1",
    })
)


@app.function(image=image, cpu=4, memory=4096, timeout=60)
def benchmark_hf_download(hf_token: str = None):
    import os
    import shutil
    from huggingface_hub import hf_hub_download

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    print("\n" + "=" * 60)
    print("🚀 MODAL ➔ HUGGING FACE NETWORK BENCHMARK")
    print("=" * 60)

    # Benchmark downloading a 130 MB real Parquet slice from OpenCodeReasoning
    repo_id = "nvidia/OpenCodeReasoning"
    filename = "split_0/train-00000-of-00030.parquet"

    print(f"Testing authenticated download of {repo_id}/{filename}...")
    t0 = time.time()
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    elapsed = max(time.time() - t0, 0.001)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    mbps = (size_mb * 8) / elapsed
    mb_per_sec = size_mb / elapsed

    print(f"  Downloaded: {size_mb:.2f} MB in {elapsed:.2f} seconds")
    print(f"  Throughput: {mb_per_sec:.2f} MB/s  ({mbps:.2f} Mbps / {mbps/1000:.2f} Gbps)")
    print("=" * 60 + "\n")

    return {
        "size_mb": size_mb,
        "elapsed": elapsed,
        "mb_per_sec": mb_per_sec,
        "gbps": mbps / 1000,
    }


@app.local_entrypoint()
def main():
    benchmark_hf_download.remote()
