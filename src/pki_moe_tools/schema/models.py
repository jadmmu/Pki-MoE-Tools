from typing import Any

TRACE_SCHEMA_VERSION = "1.0.0"
MANIFEST_SCHEMA_VERSION = "1.0.0"


def make_trace(manifest: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "trace_type": "canonical_raw_routing_trace",
        "schema_version": TRACE_SCHEMA_VERSION,
        "synthetic": bool(manifest.get("synthetic", False)),
        "manifest": manifest,
        "events": events,
    }
