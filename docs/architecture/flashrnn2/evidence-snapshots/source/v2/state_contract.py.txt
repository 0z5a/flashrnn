"""State precision and initialization scope for persistent chunk continuation."""

from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class StateContract:
    cell: str
    semantics: str
    input_dtype: torch.dtype
    compute_dtype: torch.dtype
    transport_dtype: torch.dtype
    output_dtype: torch.dtype
    components: tuple[str, ...]
    normalizer_scope: str


def state_contract(cell: str, dtype: torch.dtype, semantics: str = "flashrnn"):
    """Describe arithmetic; this declaration is not a GPU qualification."""
    if cell not in ("lstm", "slstm"):
        raise ValueError("persistent state contracts support LSTM and sLSTM")
    if semantics not in ("flashrnn", "tirex", "xlstm"):
        raise ValueError("unknown persistent arithmetic semantics")
    if semantics != "flashrnn" and cell != "slstm":
        raise ValueError("model-specific semantics require sLSTM")
    allowed = {
        "flashrnn": (torch.float16, torch.bfloat16),
        "tirex": (torch.bfloat16,),
        "xlstm": (torch.float32,),
    }
    if dtype not in allowed[semantics]:
        raise ValueError("input dtype does not match arithmetic semantics")
    compute = torch.bfloat16 if semantics == "tirex" else torch.float32
    return StateContract(
        cell,
        semantics,
        dtype,
        compute,
        compute,
        dtype,
        ("h", "c", "n", "m") if cell == "slstm" else ("h", "c"),
        "elementwise_each_step"
        if semantics == "xlstm"
        else ("logical_call_first_step" if cell == "slstm" else "none"),
    )


@dataclass(frozen=True)
class SequenceContext:
    """One logical call's reduction, shared by all batch/head shards.

    Construct before sharding. Keep the scalar tensor immutable, and advance
    the context only when the corresponding chunk has completed. Distributed
    callers must broadcast the full-call scalar to each rank's device.
    """

    contract: StateContract
    zero_normalizer: torch.Tensor
    step_offset: int = 0

    def __post_init__(self):
        if self.zero_normalizer.dtype != torch.bool or self.zero_normalizer.ndim != 0:
            raise ValueError("zero_normalizer must be a scalar boolean tensor")
        if type(self.step_offset) is not int or self.step_offset < 0:
            raise ValueError("step_offset must be a nonnegative integer")

    def advance(self, steps: int):
        if type(steps) is not int or steps < 1:
            raise ValueError("a completed chunk must contain at least one step")
        return replace(self, step_offset=self.step_offset + steps)


def begin_sequence(
    initial: torch.Tensor,
    cell: str,
    input_dtype: torch.dtype,
    semantics: str = "flashrnn",
) -> SequenceContext:
    """Compute the initialization predicate over the *unsharded* initial state."""
    contract = state_contract(cell, input_dtype, semantics)
    if (
        initial.ndim != 5
        or initial.shape[0] != len(contract.components)
        or initial.shape[2] != 1
        or initial.numel() == 0
    ):
        raise ValueError("expected a nonempty initial state [S,B,1,H,D]")
    if initial.dtype not in (input_dtype, contract.compute_dtype):
        raise ValueError("initial state dtype does not match the contract")
    zero = (
        torch.all(initial[2] == 0)
        if contract.normalizer_scope == "logical_call_first_step"
        else torch.zeros((), dtype=torch.bool, device=initial.device)
    )
    return SequenceContext(contract, zero)
