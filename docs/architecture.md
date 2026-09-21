# Architecture

```text
Transformers + PyTorch  ----\
                             -> canonical raw trace -> validation -> analysis -> figures
llama.cpp ------------------/
```

Backend-specific code ends at the trace source boundary. The schema, JSON storage, validation, aggregation, visualization, and experiment metadata are shared.

The canonical trace preserves raw router quantities when available. `null` means unavailable. It is not an inferred value.

The reference source will attach passive hooks to `model.model.layers[i].mlp.gate`. The hook observes `(router_logits, selected_weights, selected_ids)`. The local source is not implemented yet because llama.cpp requires source-level callbacks or tensor export to expose those intermediate graph values.

Token positions are a first-class mapping problem. A Transformers prefill call contains multiple positions, while cached decode calls commonly contain one. A llama.cpp batch can contain multiple tokens, positions, and sequence IDs. Hook or graph callback ordering is therefore retained as event order but is not treated as proof of canonical token position.

`reference`, `local_exploratory`, and `synthetic` are separate conditions. No analysis function combines them implicitly.
