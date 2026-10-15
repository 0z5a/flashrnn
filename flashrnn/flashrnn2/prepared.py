"""Preallocated forward-only sequence execution with explicit output policy.

Inputs, weights and outputs have fixed addresses. Results are borrowed buffers:
the next execution overwrites them. One instance must be used serially on its
bound CUDA stream. Update its input/initial buffers on that stream before run.
The default public recurrence API remains unchanged.
"""
import torch
import triton

from . import triton_rtiled
from .execution_policy import default_config
from .simt_persistent import _sequence_simt
from .state_contract import state_contract
from .triton_persistent import _sequence


class PreparedSequence:
    def __init__(
        self, wx, recurrent, bias, initial, cell="lstm", *,
        output_mode="full", snapshot_dtype=None, config=None, chunks=None,
    ):
        if output_mode not in ("full", "hidden", "last"):
            raise ValueError("output mode must be full, hidden or last")
        if wx.ndim != 5:
            raise ValueError("expected Wx [B,T,4,H,D]")
        batch, steps, gates, heads, width = wx.shape
        contract = state_contract(cell, wx.dtype)
        states = len(contract.components)
        config = config or default_config(width)
        rtiled = config.family == "mma_rtiled"
        cluster = config.family == "tilelang"
        if gates != 4 or batch < 1 or steps < 1 or heads < 1 or width < 16:
            raise ValueError("expected nonempty B/T/H and D >= 16")
        if (not (rtiled or cluster) and width > 128) or (rtiled and width % 16):
            raise ValueError("single-CTA families need D <= 128; R tiling needs D a multiple of 16")
        if cluster:
            try:
                from . import tilelang_cluster
            except ImportError as error:  # chosen before the call: no silent switch to another backend
                raise RuntimeError("the selected TileLang backend is not installed") from error
            ok = (tilelang_cluster.supported(width, config.slice_width, config.batch_tile) if config.sync == "cluster"
                  else tilelang_cluster.supported_grid(width, config.slice_width, tilelang_cluster.sm_count(),
                                                       config.batch_tile))
            if not ok:
                raise ValueError(f"the TileLang {config.sync} path does not support this (D, slice width)")
        if recurrent.shape != (4, heads, width, width) or bias.shape != (4, heads, width):
            raise ValueError("invalid recurrent-weight or bias shape")
        if initial.shape != (states, batch, 1, heads, width):
            raise ValueError("invalid initial-state shape")
        tensors = (wx, recurrent, bias, initial)
        if not wx.is_cuda or any(x.device != wx.device or not x.is_contiguous() for x in tensors):
            raise ValueError("prepared execution requires contiguous tensors on one CUDA device")
        if recurrent.dtype != wx.dtype or bias.dtype != wx.dtype or initial.dtype not in (wx.dtype, torch.float32):
            raise ValueError("input dtype does not match generic computation contract")
        if any(x.requires_grad for x in tensors):
            raise ValueError("prepared execution is forward-only")
        dtype = wx.dtype if snapshot_dtype is None else snapshot_dtype
        if dtype not in (wx.dtype, torch.float32):
            raise ValueError("snapshot dtype must be input dtype or FP32")
        lengths = (steps,) if chunks is None else tuple(chunks)
        if not lengths or any(type(n) is not int or n < 1 for n in lengths) or sum(lengths) != steps:
            raise ValueError("chunk lengths must be positive integers totaling T")
        self.wx, self.recurrent, self.bias, self.initial = tensors
        self.cell, self.shape, self.chunks = cell, (batch, steps, heads, width), lengths
        self.config = config
        self.output_mode = {"full": 0, "hidden": 1, "last": 2}[output_mode]
        shape = (states, batch, steps, heads, width) if output_mode == "full" else (
            batch, 1 if output_mode == "last" else steps, heads, width
        )
        self.output = torch.empty(shape, dtype=dtype, device=wx.device)
        self.carries = tuple(torch.empty(initial.shape, dtype=torch.float32, device=wx.device) for _ in range(2))
        self.final = self.carries[(len(lengths) - 1) % 2]
        self.zero = torch.empty((), dtype=torch.bool, device=wx.device)
        self.zero_mask = torch.empty(initial[0].shape, dtype=torch.bool, device=wx.device)
        if rtiled:
            sms = torch.cuda.get_device_properties(wx.device).multi_processor_count
            self.rtiled_plan = triton_rtiled.plan(batch, heads, width, config.slice_width, sms=sms)
            self.k_parts = triton_rtiled.k_parts(width, config.pad_k)
            slots = self.rtiled_plan[2]
            # Exchange buffer, per-chunk barrier counters and the bounded-spin error flag.
            self.xbuf = torch.empty((slots, 2, 16, width), dtype=wx.dtype, device=wx.device)
            self.sync_counts = torch.zeros((len(lengths), slots), dtype=torch.int32, device=wx.device)
            self.error_flag = torch.zeros((1,), dtype=torch.int32, device=wx.device)
        if cluster:
            names = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}
            if wx.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError("the TileLang cluster path supports FP16 and BF16")
            self.tilelang_kernel = tilelang_cluster.kernel(
                batch, steps, heads, width, config.slice_width, cell=cell, dtype=names[wx.dtype],
                mode=output_mode, snapshot=names[dtype], sync=config.sync, bm=config.batch_tile,
                threads=32 * config.warps, prefetch=config.prefetch)
            # Fixed FP32 initial state and int32 normalizer flag, refreshed inside every (captured) run.
            self.initial32 = torch.empty(initial.shape, dtype=torch.float32, device=wx.device)
            self.zero32 = torch.empty((1,), dtype=torch.int32, device=wx.device)
            # Grid sync: double-buffered global h exchange, rewritten before it is read in every chunk.
            tiles = (batch + config.batch_tile - 1) // config.batch_tile
            self.tilelang_extra = () if config.sync == "cluster" else (
                torch.empty((2, tiles * heads, width, config.batch_tile), dtype=wx.dtype, device=wx.device),)
        self._graph = None
        self.compiler_records = []

    def _execute(self, *, diagnostics=False):
        if self.cell == "slstm":
            torch.eq(self.initial[2], 0, out=self.zero_mask)
            torch.all(self.zero_mask, out=self.zero)
        else:
            self.zero.zero_()
        batch, total, heads, width = self.shape
        cfg = self.config
        previous = self.initial
        start = 0
        if cfg.family == "mma_rtiled":
            self.sync_counts.zero_()
        if cfg.family == "tilelang":
            # Input-dtype -> FP32 is exact, so the FP32 copy leaves the arithmetic unchanged.
            self.initial32.copy_(self.initial)
            self.zero32.copy_(self.zero.reshape(1))
            previous = self.initial32
        for index, length in enumerate(self.chunks):
            carry = self.carries[index % 2]
            if cfg.family == "tilelang":
                self.tilelang_kernel(self.wx, self.recurrent, self.bias, previous, self.zero32, self.output, carry,
                                     *self.tilelang_extra, length, start, int(start == 0))
                kernel = None
            elif cfg.family == "mma_rtiled":
                splits, groups, slots = self.rtiled_plan
                kernel = triton_rtiled._rtiled[(splits, slots)](
                    self.wx, self.recurrent, self.bias, previous, self.output, carry, self.xbuf,
                    self.sync_counts[index], self.error_flag, self.zero,
                    batch, length, total, start, int(start == 0), groups, slots, 1,
                    heads, width, self.cell == "slstm", 16, cfg.slice_width, *self.k_parts,
                    self.output_mode, triton_rtiled.SPIN_LIMIT,
                    num_warps=cfg.warps, enable_fp_fusion=False, launch_cooperative_grid=True,
                )
            elif cfg.family == "mma":
                kernel = _sequence[(triton.cdiv(batch, cfg.batch_tile), heads)](
                    self.wx, self.recurrent, self.bias, previous, self.output, carry,
                    batch, length, heads, width, self.cell == "slstm", self.zero,
                    cfg.batch_tile, triton.next_power_of_2(width), cfg.weight_operand == "a",
                    False, False, True, start == 0,
                    TOTAL_T=total, STEP_BASE=start, OUTPUT_MODE=self.output_mode,
                    num_warps=cfg.warps, enable_fp_fusion=False,
                )
            else:
                kernel = _sequence_simt[(batch, heads)](
                    self.wx, self.recurrent, self.bias, previous, self.output, carry,
                    batch, length, heads, width, self.cell == "slstm", self.zero,
                    triton.next_power_of_2(width), total, start, self.output_mode, start == 0,
                    num_warps=cfg.warps, enable_fp_fusion=False,
                )
            if diagnostics and kernel is not None:
                self.compiler_records.append({
                    "step_base": start, "steps": length, "registers": kernel.n_regs,
                    "spills": kernel.n_spills, "shared_bytes": kernel.metadata.shared,
                    "ptx": kernel.asm["ptx"],
                })
            previous, start = carry, start + length
        return self.output, self.final

    def check(self):
        """Synchronize and raise if an R-tiled barrier reported a timeout."""
        if self.config.family == "mma_rtiled" and int(self.error_flag.item()):
            raise RuntimeError("R-tiled group barrier timed out; outputs are invalid")

    def run(self):
        if self._graph is not None:
            self._graph.replay()
            return self.output, self.final
        return self._execute()

    def capture(self):
        if self._graph is not None:
            raise ValueError("sequence is already captured")
        with torch.cuda.device(self.wx.device):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._execute()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._execute()
            self._graph = graph
        return self

    def close(self):
        if self._graph is not None:
            torch.cuda.synchronize(self.wx.device)
            self._graph.reset()
            self._graph = None
