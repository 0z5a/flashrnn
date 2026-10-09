"""Capture TiRex cell inputs, including its keyword state argument."""


def capture_cell_inputs(records):
    def capture(_module, args, kwargs):
        assert len(args) == 1 and set(kwargs) == {"state"}
        gates = args[0]
        state = kwargs["state"]
        records.append(
            (
                gates.detach().cpu().clone(),
                None if state is None else state.detach().cpu().clone(),
            )
        )

    return capture
