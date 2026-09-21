# llama.cpp reconnaissance

Current llama.cpp supports the `OLMOE` architecture. The model loader and graph dispatch are present in `src/llama-model.cpp`, and the OLMoE-specific model loading and graph construction are in `src/models/olmoe.cpp`.

The OLMoE graph calls the shared `llm_graph_context::build_moe_ffn` implementation in `src/llama-graph.cpp`. For OLMoE it passes softmax gating and `norm_w=false`, matching the checkpoint configuration `norm_topk_prob=false`.

The relevant graph stages are:

- `ffn_moe_logits`: router matrix multiplication;
- `ffn_moe_probs`: softmax probabilities;
- `ffn_moe_argsort` and `ffn_moe_topk`: selected expert indices;
- `ffn_moe_weights`: selected, unnormalized router weights.

The graph callback records tensor nodes internally, but the public llama.cpp API does not currently expose these routing tensors as a stable inference result. The minimum future instrumentation is a read-only callback or export path at these named graph nodes, with layer ID, token positions, tensor shape, and copied host values. It must not replace, reorder, or modify any graph tensor.

OLMoE support is therefore architecturally feasible, but a canonical routing trace is not available from the stock public API. A source-level change is required for selected IDs and weights, and likely for logits and probabilities if they are to be preserved directly.

For Windows and an RX 6600, CPU is the conservative supported path. Vulkan is the practical local GPU candidate because llama.cpp advertises MoE support in its Vulkan feature matrix and RX 6600 Vulkan execution is documented by the project community. ROCm/HIP on this specific Windows GPU is less attractive because support and packaging are more conditional. The local backend should initially be recorded as CPU or Vulkan, never assumed.

The 7B checkpoint is too large for ordinary float32 execution in 16 GB RAM. A 4-bit or 5-bit GGUF is the realistic local range. Approximate weight storage is 3.5 to 5 GB before runtime overhead, KV cache, allocator space, and metadata. An 8 GB RX 6600 may fit a 4-bit model with partial CPU offload, but exact VRAM use must be measured. This is an estimate, not a measured result.

The local and reference semantics are close in structure but not identical in execution. Both use softmax and top-8 selection for this architecture. They can differ through quantized expert and router weights, accumulator precision, backend kernels, tensor layouts, tie behavior, batching, and prefill/decode scheduling. The llama.cpp path must not be treated as a reference-equivalent source until cross-backend validation is performed.

Sources: [OLMoE graph](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/models/olmoe.cpp), [shared MoE graph](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-graph.cpp), [model architecture dispatch](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp), [feature matrix](https://github.com/ggml-org/llama.cpp/wiki/Feature-matrix), [GGUF model documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/models.md).
