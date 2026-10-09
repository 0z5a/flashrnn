"""One static decode graph with request-owned returned tensors."""

from collections.abc import Callable

import torch

Output = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
Decode = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Output]


class GraphDecode:
    def __init__(self, decode: Decode, sample: Output) -> None:
        self.inputs = (sample[0].clone(), sample[1].clone(), sample[2].clone())
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                decode(*self.inputs)
        torch.cuda.current_stream().wait_stream(stream)
        for target, value in zip(self.inputs, sample):
            target.copy_(value)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.outputs = decode(*self.inputs)

    def __call__(
        self, ids: torch.Tensor, conv: torch.Tensor, ssm: torch.Tensor
    ) -> Output:
        for target, value in zip(self.inputs, (ids, conv, ssm)):
            target.copy_(value)
        self.graph.replay()
        # Replays share static storage; queued requests must own their snapshots.
        return self.outputs[0].clone(), self.outputs[1].clone(), self.outputs[2].clone()
