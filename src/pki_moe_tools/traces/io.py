import json
from pathlib import Path
from typing import Any

from pki_moe_tools.schema.validation import validate_trace


def save_trace(trace: dict[str, Any], path: str | Path) -> None:
    validate_trace(trace)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_trace(path: str | Path) -> dict[str, Any]:
    trace = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_trace(trace)
    return trace
