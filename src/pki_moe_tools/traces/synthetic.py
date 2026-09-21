import math
import random
from datetime import datetime, timezone
from typing import Any

from pki_moe_tools.schema.models import MANIFEST_SCHEMA_VERSION, make_trace


def make_synthetic_trace(seed: int = 7, token_count: int = 4) -> dict[str, Any]:
    rng = random.Random(seed)
    run_id = f"synthetic-{seed}"
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "experiment-0.0",
        "condition": "synthetic",
        "synthetic": True,
        "prompt": "synthetic neutral router fixture",
        "prompt_id": "synthetic-prompt",
        "trajectory_id": "synthetic-trajectory",
        "turn_id": 0,
        "model": {"repository": "synthetic/olmoe-like", "checkpoint": None, "revision": None, "architecture": "olmoe-like", "config_fingerprint": None},
        "software": {"pki_moe_tools": "0.1.0", "git_commit": None, "python": None, "os": None},
        "backend": {"name": "synthetic", "version": None, "transformers": None, "torch": None, "llama_cpp": None},
        "weights": {"format": None, "dtype": "float32", "quantization": None, "metadata": None},
        "hardware": {"cpu": None, "gpu": None, "vram_bytes": None, "ram_bytes": None, "compute_backend": "none"},
        "generation": {"seed": seed, "do_sample": False, "temperature": None, "top_p": None, "top_k": None, "max_new_tokens": token_count, "parameters": {}},
        "instrumentation": {"implementation": "synthetic-fixture", "version": "0.1.0", "captured": ["router_logits", "selected_expert_ids", "selected_expert_weights"], "limitations": ["synthetic fixture"]},
    }
    events: list[dict[str, Any]] = []
    event_order = 0
    for token_position in range(token_count):
        for layer_id in range(16):
            logits = [rng.uniform(-2.0, 2.0) for _ in range(64)]
            ids = sorted(range(64), key=lambda index: logits[index], reverse=True)[:8]
            exp_values = [math.exp(logits[index]) for index in ids]
            total = sum(exp_values)
            weights = [value / total for value in exp_values]
            events.append({
                "run_id": run_id,
                "prompt_id": manifest["prompt_id"],
                "trajectory_id": manifest["trajectory_id"],
                "turn_id": manifest["turn_id"],
                "forward_call_index": token_position,
                "layer_id": layer_id,
                "token_position": token_position,
                "token_id": token_position,
                "token_text": None,
                "phase": "synthetic",
                "selected_expert_ids": ids,
                "selected_expert_weights": weights,
                "router_logits": logits,
                "input_shape": [1, token_count, 2048],
                "selected_shape": [8],
                "event_order": event_order,
            })
            event_order += 1
    return make_trace(manifest, events)
