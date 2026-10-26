"""Inspect and compare Pulsar-300M latent reasoning and token predictions pulse-by-pulse (Pulse 1 vs Pulse 2 vs Pulse 3)."""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import PulsarConfig, PulsarForCausalLM


def load_model_and_tokenizer(checkpoint_path: str = "checkpoints/milestone_step_500.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "checkpoints/latest.pt"

    print(f"\n📦 Loading Pulsar Checkpoint: {checkpoint_path} on {device}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    tokenizer = AutoTokenizer.from_pretrained("data/pulsar_tokenizer")
    cfg = PulsarConfig(
        max_seq_len=2048,
        base_seq_len=1024,
        default_pulses=3,
    )
    model = PulsarForCausalLM(cfg).to(device=device)
    model.to_training_precision(torch.bfloat16)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    step = ckpt.get("step", 500)
    tokens_seen = ckpt.get("tokens_seen", 65536000)
    val_loss = ckpt.get("val_loss", 2.6761)
    print(f"✅ Checkpoint Loaded: Step {step} | Tokens: {tokens_seen/1e6:.1f}M | Val Loss: {val_loss:.4f}\n")

    return model, tokenizer, device


def analyze_next_token_pulses(model, tokenizer, prompt: str, device: str):
    """Analyze top-5 token predictions, probabilities, and entropy across Pulse 1, 2, and 3."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    print("=" * 80)
    print("🎯 NEXT-TOKEN PREDICTION ANALYSIS ACROSS PULSES (8 vs 16 vs 24 effective layers)")
    print("=" * 80)
    print(f"Prompt: {repr(prompt)}\n")

    with torch.no_grad():
        for k in [1, 2, 3]:
            eff_layers = k * 8
            logits, _, _ = model(input_ids, num_pulses=k)
            last_logits = logits[0, -1, :].float()

            # Softmax probabilities
            probs = F.softmax(last_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-12)).sum().item()

            # Top 5 candidates
            topk_probs, topk_indices = torch.topk(probs, 5)
            topk_tokens = [tokenizer.decode([idx.item()]) for idx in topk_indices]

            top1_margin = (last_logits[topk_indices[0]] - last_logits[topk_indices[1]]).item()

            print(f"--- Pulse {k} [{eff_layers:2d} effective layers] ---")
            print(f"  Entropy: {entropy:.3f} nats | Top-1 Confidence Margin: {top1_margin:+.2f}")
            print("  Top-5 Candidates:")
            for rank, (tok, p) in enumerate(zip(topk_tokens, topk_probs), 1):
                clean_tok = repr(tok).replace("'", "")
                bar = "█" * int(p.item() * 30)
                print(f"    {rank}. {clean_tok:<18s} | {p.item()*100:5.1f}% | {bar}")
            print()


def generate_with_pulse(model, tokenizer, prompt: str, k_pulses: int, max_tokens: int = 40, temperature: float = 0.7, repetition_penalty: float = 1.15, device: str = "cuda"):
    """Generate completion using a fixed number of recurrent pulses (k_pulses)."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    curr_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_tokens):
            context_window = curr_ids[:, -1024:]
            logits, _, _ = model(context_window, num_pulses=k_pulses)
            next_token_logits = logits[0, -1, :].clone().float()

            # Repetition penalty
            seen_tokens = set(curr_ids[0].tolist())
            for tok_id in seen_tokens:
                val = next_token_logits[tok_id]
                if val > 0:
                    next_token_logits[tok_id] = val / repetition_penalty
                else:
                    next_token_logits[tok_id] = val * repetition_penalty

            scaled_logits = next_token_logits / max(temperature, 1e-5)
            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    completion = tokenizer.decode(curr_ids[0, prompt_len:], skip_special_tokens=False)
    return completion


def compare_generations(model, tokenizer, prompt: str, tag: str, max_tokens: int = 40, device: str = "cuda"):
    print("=" * 80)
    print(f"📝 FULL TEXT GENERATION: [{tag}]")
    print("=" * 80)
    print(f"Prompt:\n{prompt}\n")

    torch.manual_seed(42)
    for k in [1, 2, 3]:
        eff = k * 8
        torch.manual_seed(42)  # identical sampling seed for direct comparison
        comp = generate_with_pulse(model, tokenizer, prompt, k_pulses=k, max_tokens=max_tokens, device=device)
        print(f"⚡ Pulse {k} [{eff} layers]:")
        print(f"   {comp.strip()}\n")
    print("-" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/milestone_step_500.pt")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt to inspect")
    parser.add_argument("--max-tokens", type=int, default=45)
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(args.ckpt)

    test_prompts = [
        {
            "tag": "Python / Algorithm Implementation",
            "prompt": "def quicksort(arr: list[int]) -> list[int]:\n    if len(arr) <= 1:\n        return arr\n    ",
        },
        {
            "tag": "Agentic MCP Tool Invocation",
            "prompt": "<|im_start|>user\nFind recent python errors in system logs<|im_end|>\n<|im_start|>assistant\n<|tool_call|>",
        },
        {
            "tag": "Mathematical Problem / Latent Reasoning",
            "prompt": "<|im_start|>user\nIf 3x + 5 = 20, what is the value of x?<|im_end|>\n<|im_start|>assistant\n<|think|>\n",
        },
        {
            "tag": "Linux CLI Command",
            "prompt": "<|im_start|>user\nWrite a command to check the size of all directories in /var/log in human-readable format.<|im_end|>\n<|im_start|>assistant\n```bash\n",
        },
    ]

    if args.prompt:
        analyze_next_token_pulses(model, tokenizer, args.prompt, device)
        compare_generations(model, tokenizer, args.prompt, "Custom Prompt", max_tokens=args.max_tokens, device=device)
    else:
        # First analyze next-token distribution for the first prompt
        analyze_next_token_pulses(model, tokenizer, test_prompts[0]["prompt"], device)

        # Run multi-pulse completions across all canaries
        for p in test_prompts:
            compare_generations(model, tokenizer, p["prompt"], p["tag"], max_tokens=args.max_tokens, device=device)


if __name__ == "__main__":
    main()
