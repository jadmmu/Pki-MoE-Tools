# PKI-MoE-Tools

PKI-MoE-Tools is research infrastructure for measuring expert routing in sparse mixture-of-experts language models. The current research question is whether the previous active routing state contains useful information about the next active state during a coherent task trajectory.

The project currently measures routing. It does not implement PKI, expert residency, paging, eviction, prefetching, prediction, training, or semantic labeling.

## Phase 2 status

The backend-independent pipeline is implemented for synthetic OLMoE-like traces. It supports a reference Transformers/PyTorch condition and a local exploratory llama.cpp condition at the schema boundary. No model weights are downloaded by the project tests.

The reference model is `allenai/OLMoE-1B-7B-0125`. Its planned capture path is a passive hook on each `OlmoeTopKRouter`. The local llama.cpp path remains a future source-level instrumentation task.

## Reproduce the synthetic pipeline

```text
uv run --extra dev pki-moe synthetic --output data/raw/synthetic.json
uv run --extra dev pki-moe validate data/raw/synthetic.json
uv run --extra dev pki-moe inspect data/raw/synthetic.json --limit 3
uv run --extra dev pki-moe heatmap data/raw/synthetic.json --output figures/synthetic-layer-expert.png
uv run --extra dev pytest
```

The generated trace is explicitly marked `synthetic: true` and is not an experimental observation.

## Conditions

`reference` denotes direct Transformers/PyTorch instrumentation of the unquantized or explicitly documented OLMoE checkpoint. `local_exploratory` denotes llama.cpp execution of an explicitly identified GGUF quantization. These conditions must not be merged silently.

See [docs/architecture.md](docs/architecture.md), [docs/experiment-0.0.md](docs/experiment-0.0.md), and [docs/llama-cpp-reconnaissance.md](docs/llama-cpp-reconnaissance.md).
