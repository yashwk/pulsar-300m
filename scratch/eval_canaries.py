import os
import sys
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from src.model import PulsarConfig, PulsarForCausalLM

def run_canaries():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Pulsar Canary Evaluator] Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    ckpt_path = "checkpoints/milestone_step_500.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/latest.pt"
    print(f"[Pulsar Canary Evaluator] Loading checkpoint: {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    tokenizer = AutoTokenizer.from_pretrained("data/pulsar_tokenizer")
    print(f"[Pulsar Canary Evaluator] Tokenizer loaded (vocab: {len(tokenizer):,})")

    cfg = PulsarConfig(
        max_seq_len=32768,
        base_seq_len=1024,
        default_pulses=3,
    )
    model = PulsarForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    canary_prompts = [
        {
            "id": 1,
            "tag": "Python / Algorithms",
            "category": "Python",
            "prompt": "def quicksort(arr: list[int]) -> list[int]:\n    if len(arr) <= 1:\n        return arr\n    ",
        },
        {
            "id": 2,
            "tag": "Agentic MCP Tool Call",
            "category": "Tool Calling",
            "prompt": "<|im_start|>user\nRename scan_01.pdf to 2026_Invoice.pdf in /home/user/docs<|im_end|>\n<|im_start|>assistant\n<|tool_call|>",
        },
        {
            "id": 3,
            "tag": "Math / Latent CoT",
            "category": "Math",
            "prompt": "<|im_start|>user\nIf a train travels 120 km in 2 hours, what is its speed in km/h?<|im_end|>\n<|im_start|>assistant\n<|think|>\n",
        },
        {
            "id": 4,
            "tag": "Bash / Linux CLI",
            "category": "CLI",
            "prompt": "<|im_start|>user\nWrite a bash command to find all .pdf files in ~/Downloads modified in the last 7 days.<|im_end|>\n<|im_start|>assistant\n",
        },
        {
            "id": 5,
            "tag": "STEM / World Knowledge",
            "category": "STEM",
            "prompt": "The primary function of mitochondria in a biological cell is to",
        },
    ]

    torch.manual_seed(42)
    results = []

    print("\n" + "=" * 80)
    print("      PULSAR MILESTONE STEP 500: CANARY PROMPT EVALUATION")
    print("=" * 80)

    for c in canary_prompts:
        prompt = c["prompt"]
        tag = c["tag"]
        category = c["category"]
        canary_id = c["id"]

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            curr_ids = input_ids.clone()
            for _ in range(50):
                context_window = curr_ids[:, -1024:]
                logits, _, _ = model(context_window, labels=None)
                next_token_logits = logits[:, -1, :].clone()

                # Repetition penalty
                seen_tokens = set(curr_ids[0].tolist())
                for tok_id in seen_tokens:
                    val = next_token_logits[0, tok_id]
                    if val > 0:
                        next_token_logits[0, tok_id] = val / 1.15
                    else:
                        next_token_logits[0, tok_id] = val * 1.15

                scaled_logits = next_token_logits / 0.7
                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                curr_ids = torch.cat([curr_ids, next_token], dim=1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

        full_decoded = tokenizer.decode(curr_ids[0], skip_special_tokens=False)
        completion_tokens = curr_ids[0, prompt_len:]
        completion_decoded = tokenizer.decode(completion_tokens, skip_special_tokens=False)

        print(f"\n--- [{tag}] ({category}) ---")
        print(f"Prompt:\n{prompt}")
        print(f"\nCompletion:\n{completion_decoded.strip()}")
        print("-" * 80)

        results.append({
            "id": canary_id,
            "tag": tag,
            "category": category,
            "prompt": prompt,
            "completion": full_decoded,
        })

    # Update live_status.json so the web dashboard displays them
    live_status_path = "checkpoints/live_status.json"
    if os.path.exists(live_status_path):
        try:
            with open(live_status_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["canaries"] = results
            with open(live_status_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print("\n[Pulsar Canary Evaluator] Updated checkpoints/live_status.json with latest canaries.")
        except Exception as e:
            print(f"[Notice] Could not update live_status.json: {e}")

if __name__ == "__main__":
    run_canaries()
