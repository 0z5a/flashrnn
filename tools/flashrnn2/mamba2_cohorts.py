"""Select reproducible Mamba2 request groups from the pinned input file."""

import hashlib
import json


def select_groups(inputs: dict, batch: int) -> list[list[list[int]]]:
    source_batch = max(batch, 16)
    groups = inputs["batches"][str(source_batch)]["input_ids"]
    assert len(groups) == 3
    selected = [group[:batch] for group in groups]
    assert all(
        len(group) == batch
        and all(len(prompt) == inputs["prompt_length"] for prompt in group)
        for group in selected
    )
    return selected


def cohort_sha256(groups: list[list[list[int]]]) -> str:
    payload = json.dumps(groups, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
