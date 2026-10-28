"""Test NVIDIA H200 SXM GPU allocation and runtime on Modal."""

import modal

app = modal.App("test-h200-access")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4.0")
)


@app.function(
    image=image,
    gpu="H200",
    timeout=120,
)
def probe_h200():
    import torch
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
    cuda_cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)

    print("=" * 60)
    print(f"Device Name:        {device_name}")
    print(f"Total VRAM:         {vram_gb:.2f} GB")
    print(f"Compute Capability: {cuda_cap[0]}.{cuda_cap[1]}")
    print("=" * 60)

    # BF16 Tensor Core test
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    y = x @ x
    torch.cuda.synchronize()
    print("BF16 Tensor Core Matrix Multiplication: PASSED")

    return {
        "device": device_name,
        "vram_gb": round(vram_gb, 2),
        "compute_capability": f"{cuda_cap[0]}.{cuda_cap[1]}",
        "status": "success",
    }


@app.local_entrypoint()
def main():
    result = probe_h200.remote()
    print(f"\n[Modal Probe Output]: {result}")
