import math
from typing import Any

from .models import MANIFEST_SCHEMA_VERSION, TRACE_SCHEMA_VERSION

REQUIRED_MANIFEST = {
    "run_id", "timestamp", "experiment_id", "condition", "prompt_id", "trajectory_id", "turn_id",
    "prompt", "model", "software", "backend", "weights", "hardware", "generation", "instrumentation",
}
REQUIRED_EVENT = {
    "run_id", "prompt_id", "trajectory_id", "turn_id", "forward_call_index", "layer_id", "token_position",
    "token_id", "selected_expert_ids", "selected_expert_weights", "router_logits", "input_shape",
    "selected_shape", "event_order",
}


def _fail(message: str) -> None:
    raise ValueError(message)


def _finite(values: Any, path: str) -> None:
    if values is None:
        return
    if isinstance(values, list):
        for index, value in enumerate(values):
            _finite(value, f"{path}[{index}]")
        return
    if not isinstance(values, (int, float)) or not math.isfinite(float(values)):
        _fail(f"non-finite value at {path}")


def validate_manifest(manifest: dict[str, Any]) -> None:
    missing = REQUIRED_MANIFEST - manifest.keys()
    if missing:
        _fail(f"manifest missing fields: {sorted(missing)}")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        _fail("invalid manifest schema version")
    if manifest["condition"] not in {"reference", "local_exploratory", "synthetic"}:
        _fail("invalid execution condition")
    if not isinstance(manifest["run_id"], str) or not manifest["run_id"]:
        _fail("missing run ID")
    if not isinstance(manifest["model"], dict) or not isinstance(manifest["backend"], dict):
        _fail("model and backend metadata must be objects")


def validate_trace(trace: dict[str, Any]) -> None:
    if trace.get("trace_type") != "canonical_raw_routing_trace":
        _fail("invalid trace type")
    if trace.get("schema_version") != TRACE_SCHEMA_VERSION:
        _fail("invalid trace schema version")
    if not isinstance(trace.get("events"), list):
        _fail("events must be a list")
    manifest = trace.get("manifest")
    if not isinstance(manifest, dict):
        _fail("trace manifest must be an object")
    validate_manifest(manifest)
    if bool(trace.get("synthetic", False)) != bool(manifest.get("synthetic", False)):
        _fail("trace and manifest synthetic flags disagree")
    expected_order = 0
    for event in trace["events"]:
        missing = REQUIRED_EVENT - event.keys()
        if missing:
            _fail(f"event missing fields: {sorted(missing)}")
        if event["run_id"] != manifest["run_id"]:
            _fail("event and manifest run IDs disagree")
        if event["prompt_id"] != manifest["prompt_id"]:
            _fail("event and manifest prompt IDs disagree")
        if event["trajectory_id"] != manifest["trajectory_id"] or event["turn_id"] != manifest["turn_id"]:
            _fail("event and manifest trajectory identity disagree")
        if event["event_order"] != expected_order:
            _fail("inconsistent event ordering")
        expected_order += 1
        if not isinstance(event["forward_call_index"], int) or event["forward_call_index"] < 0:
            _fail("invalid forward call index")
        if not isinstance(event["layer_id"], int) or not 0 <= event["layer_id"] < 16:
            _fail("invalid layer ID")
        ids = event["selected_expert_ids"]
        weights = event["selected_expert_weights"]
        logits = event["router_logits"]
        if ids is not None:
            if not isinstance(ids, list) or len(ids) != 8:
                _fail("selected expert IDs must contain top-8 values")
            if any(not isinstance(value, int) or not 0 <= value < 64 for value in ids):
                _fail("invalid expert ID")
        if weights is not None:
            if not isinstance(weights, list) or len(weights) != 8:
                _fail("selected expert weights must contain top-8 values")
            _finite(weights, "selected_expert_weights")
            if any(value < 0 for value in weights):
                _fail("negative selected expert weight")
        if logits is not None:
            if not isinstance(logits, list) or len(logits) != 64:
                _fail("router logits must contain 64 values")
            _finite(logits, "router_logits")
        if ids is not None and weights is not None and len(ids) != len(weights):
            _fail("selected expert IDs and weights disagree")
