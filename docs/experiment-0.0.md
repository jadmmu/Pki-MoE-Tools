# Experiment 0.0: Instrumentation Feasibility

Experiment 0.0 asks whether expert-routing behavior can be observed and stored from a pretrained MoE without changing normal inference behavior.

The eventual PKI question distinguishes observed activation demand $A_t$ from physical residency $R_t$. This phase measures neither residency nor predictive value. It establishes the measurement instrument.

OLMoE uses 16 routed layers, 64 experts per layer, and top-8 routing. The Transformers router computes a linear router logit vector, applies softmax, and selects eight experts. Its router return tuple contains raw logits, selected weights, and selected IDs.

The reference condition is direct Transformers/PyTorch instrumentation. The local exploratory condition is llama.cpp with an explicitly identified GGUF quantization. Quantized local routes are not assumed equivalent to the reference route.

Synthetic fixtures provide software tests only. They are marked `synthetic: true` and cannot support scientific interpretation.

The first trustworthy real trace requires a fixed prompt, deterministic generation, recorded environment metadata, complete layer and event identity, explicit token mapping, stored selected IDs and weights, validation, and an uninstrumented comparison.

The main unresolved issue is token mapping during prefill and cached decoding. Hook order is not sufficient evidence. The trace must retain forward-call shape and source-specific position metadata, or leave `token_position` null.

Future cross-backend validation will compare generated tokens, selected IDs, weights, layer frequencies, aggregate activation vectors, and routing similarity under the same prompt and comparable deterministic settings. Differences may arise from quantization, arithmetic precision, backend kernels, tensor layout, batching, and sampling implementation.
