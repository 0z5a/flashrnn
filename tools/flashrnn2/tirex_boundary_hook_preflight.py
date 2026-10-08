"""Check the pinned Torch keyword-state pre-hook contract on CPU."""

import json

import torch
from tirex_boundary_hook import capture_cell_inputs


class SampleCell(torch.nn.Module):
    def forward(self, gates, state=None):
        return gates if state is None else gates + state


def main():
    cell = SampleCell()
    records = []
    hook = cell.register_forward_pre_hook(
        capture_cell_inputs(records), with_kwargs=True
    )
    gates = torch.tensor([[1.0, 2.0]])
    state = torch.tensor([[3.0, 4.0]])
    try:
        assert torch.equal(cell(gates, state=None), gates)
        assert torch.equal(cell(gates, state=state), gates + state)
    finally:
        hook.remove()
    assert len(records) == 2
    assert torch.equal(records[0][0], gates) and records[0][1] is None
    assert torch.equal(records[1][0], gates)
    assert torch.equal(records[1][1], state)
    print(json.dumps({"status": "PASS", "torch": torch.__version__, "captures": 2}))


if __name__ == "__main__":
    main()
