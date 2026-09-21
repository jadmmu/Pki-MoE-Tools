import argparse
import json
from pathlib import Path

from pki_moe_tools.traces.io import load_trace, save_trace
from pki_moe_tools.traces.synthetic import make_synthetic_trace
from pki_moe_tools.visualization.heatmap import save_activation_heatmap


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pki-moe")
    subparsers = parser.add_subparsers(dest="command", required=True)
    synthetic = subparsers.add_parser("synthetic")
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--seed", type=int, default=7)
    synthetic.add_argument("--tokens", type=int, default=4)
    for name in ("validate", "summarize"):
        command = subparsers.add_parser(name)
        command.add_argument("trace")
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("trace")
    inspect.add_argument("--limit", type=int, default=10)
    heatmap = subparsers.add_parser("heatmap")
    heatmap.add_argument("trace")
    heatmap.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "synthetic":
        path = Path(args.output)
        save_trace(make_synthetic_trace(args.seed, args.tokens), path)
        print(f"Wrote synthetic trace: {path}")
        return
    trace = load_trace(args.trace)
    manifest = trace["manifest"]
    if args.command == "validate":
        print(f"Valid: {args.trace}")
        return
    if args.command == "summarize":
        print(json.dumps({"run_id": manifest["run_id"], "condition": manifest["condition"], "backend": manifest["backend"]["name"], "model": manifest["model"]["repository"], "synthetic": trace["synthetic"], "events": len(trace["events"])}, indent=2))
        return
    if args.command == "inspect":
        print(f"Run: {manifest['run_id']}")
        print(f"Condition: {manifest['condition']}")
        print(f"Backend: {manifest['backend']['name']}")
        print(f"Model: {manifest['model']['repository']}")
        if trace["synthetic"]:
            print("SYNTHETIC: true")
        for event in trace["events"][:args.limit]:
            print(f"Forward call: {event['forward_call_index']}")
            print(f"Layer: {event['layer_id']}")
            print(f"Token position: {event['token_position'] if event['token_position'] is not None else 'unknown'}")
            print(f"Selected experts: {event['selected_expert_ids']}")
            print(f"Selected weights: {event['selected_expert_weights']}")
        return
    if args.command == "heatmap":
        save_activation_heatmap(trace, args.output)
        print(f"Wrote heatmap: {args.output}")
