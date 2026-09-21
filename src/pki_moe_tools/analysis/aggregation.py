from typing import Any


def activation_frequency(trace: dict[str, Any]) -> list[list[int]]:
    matrix = [[0 for _ in range(64)] for _ in range(16)]
    for event in trace["events"]:
        for expert_id in event["selected_expert_ids"] or []:
            matrix[event["layer_id"]][expert_id] += 1
    return matrix
