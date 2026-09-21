# Future cross-backend validation

Use the same model family and prompt with deterministic generation settings.

The local condition will use an identified OLMoE GGUF file and llama.cpp on CPU or Vulkan. The reference condition will use the original checkpoint in Transformers/PyTorch on an NVIDIA GPU with explicitly recorded BF16 or FP16 settings.

Compare generated token IDs first. Then compare selected expert IDs, selected weights, layer-wise activation frequencies, aggregate activation vectors, and routing similarity. Report exact mismatches and numerical tolerances. Do not pool observations across conditions until equivalence has been evaluated.

A mismatch is an observation about execution conditions. It is not evidence for or against the PKI hypothesis.
