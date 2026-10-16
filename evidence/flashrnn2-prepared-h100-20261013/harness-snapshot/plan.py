"""Finite optimization matrix; training and evaluation payloads are distinct."""
from flashrnn.flashrnn2.execution_policy import KernelConfig

SHAPES = [(1,1,64,1),(2,32,32,4),(2,64,128,4),(3,17,32,4),(3,17,48,2),
          (3,17,65,2),(16,32,64,2),(16,128,64,2),(32,128,64,2),(16,512,128,1)]
CONFIGS = {
    "mma4": KernelConfig(), "mma8": KernelConfig(warps=8),
    "mma_a4": KernelConfig(weight_operand="a"),
    "simt4": KernelConfig(family="simt",batch_tile=1),
    "simt8": KernelConfig(family="simt",batch_tile=1,warps=8),
    "mma32": KernelConfig(batch_tile=32),
}
MODEL_SHAPES = [(2,32,32,4),(16,32,64,2),(16,128,64,2),(32,128,64,2)]
MODEL_ARMS = ("legacy","dispatch","hidden","small","prepared","legacy_graph","optimized_graph")
CELL_PAIRS = (("legacy","mma4"),("mma4","mma8"),("mma4","mma_a4"),
              ("mma4","hidden"),("mma4","last"),("mma4","simt4"),("mma4","simt8"),
              ("mma4","mma32"),("mma4","selected"))
MODEL_PAIRS = (("legacy","dispatch"),("dispatch","hidden"),("hidden","small"),
               ("small","prepared"),("legacy","legacy_graph"),("prepared","optimized_graph"),
               ("legacy_graph","optimized_graph"),("legacy","optimized_graph"))


def configs_for(case):
    names = ["mma4","mma8","mma_a4"]
    if case["batch"] <= 3:
        names += ["simt4","simt8"]
    if case["batch"] >= 32:
        names += ["mma32"]
    return names


def chunk_lengths(t):
    return (1,3,t-4) if t > 4 else (1,)*(t)
