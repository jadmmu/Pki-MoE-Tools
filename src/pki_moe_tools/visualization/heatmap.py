from pathlib import Path
from typing import Any

from pki_moe_tools.analysis.aggregation import activation_frequency


def save_activation_heatmap(trace: dict[str, Any], output: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = activation_frequency(trace)
    figure, axis = plt.subplots(figsize=(12, 5))
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    axis.set_xlabel("Expert ID")
    axis.set_ylabel("Layer ID")
    title = f"Layer × expert activation frequency | run={trace['manifest']['run_id']}"
    if trace.get("synthetic"):
        title = f"SYNTHETIC | {title}"
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="Selected-token count")
    figure.tight_layout()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, metadata={"run_id": trace["manifest"]["run_id"], "schema_version": trace["schema_version"], "synthetic": str(trace.get("synthetic", False))})
    plt.close(figure)
