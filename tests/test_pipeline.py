import json

import pytest

from pki_moe_tools.analysis.aggregation import activation_frequency
from pki_moe_tools.schema.validation import validate_trace
from pki_moe_tools.traces.io import load_trace, save_trace
from pki_moe_tools.traces.synthetic import make_synthetic_trace


def test_synthetic_trace_shape_and_frequency() -> None:
    trace = make_synthetic_trace(seed=3, token_count=2)
    validate_trace(trace)
    assert trace["synthetic"] is True
    assert len(trace["events"]) == 32
    assert sum(sum(row) for row in activation_frequency(trace)) == 16 * 2 * 8


def test_round_trip(tmp_path) -> None:
    path = tmp_path / "trace.json"
    save_trace(make_synthetic_trace(), path)
    assert load_trace(path) == json.loads(path.read_text())


def test_invalid_expert_id() -> None:
    trace = make_synthetic_trace()
    trace["events"][0]["selected_expert_ids"][0] = 64
    with pytest.raises(ValueError, match="invalid expert ID"):
        validate_trace(trace)


def test_invalid_identity() -> None:
    trace = make_synthetic_trace()
    trace["events"][0]["run_id"] = "other"
    with pytest.raises(ValueError, match="run IDs disagree"):
        validate_trace(trace)


def test_invalid_nonfinite_logit() -> None:
    trace = make_synthetic_trace()
    trace["events"][0]["router_logits"][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_trace(trace)
