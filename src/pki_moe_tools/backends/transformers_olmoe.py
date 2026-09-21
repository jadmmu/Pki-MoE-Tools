from typing import Any, Iterable

from .base import TraceSource


class TransformersOlmoeTraceSource(TraceSource):
    condition = "reference"
    backend_name = "transformers"

    def __init__(self, model: Any, tokenizer: Any, **metadata: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.metadata = metadata

    def capture(self, prompt: str, **kwargs: Any) -> Iterable[dict[str, Any]]:
        raise NotImplementedError("Phase 2 defines the hook contract; real model capture is Phase 3")

    @staticmethod
    def hook_path() -> str:
        return "model.model.layers[layer_id].mlp.gate"

    @staticmethod
    def output_contract() -> tuple[str, str, str]:
        return "router_logits", "selected_weights", "selected_ids"

    @staticmethod
    def token_mapping_status() -> str:
        return "unresolved: prefill and cached decode calls require explicit input/cache position tracking"
