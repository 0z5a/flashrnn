# TiRex first-divergence boundary probe

The [C32 diagnostic](tirex-e2e-r2-diagnostic-results.md) found six groups
whose cell outputs first differ bitwise late in one 64-step patch. This
source-only probe records the input gates and incoming state at each first
different cell. It then runs the official Torch cell and FlashRNN2 cell on
the **same** input and initial state through the preceding prefix, followed
by an isolated single step from the official prefix state.

| Group | Cell | Isolated step |
|---|---|---:|
| G1 | L3/P1 | 61 |
| G4 | L3/P1 | 59 |
| G6 | L10/P1 | 61 |
| G7 | L9/P1 | 61 |
| G8 | L9/P0 | 61 |
| G11 | L3/P0 | 61 |

The probe binds the same pinned checkpoint, source, fixture, previously
passing fixed-input gate, and failed C32 diagnostic by SHA256. It saves both
arms' full forecast, cell-boundary tensors, prefix output and four final
states, and isolated-step output and four final states. Its separate NumPy
auditor checks all 72 tensor comparisons and the reported input/state
equality flags. A forecast flag change relative to the prior diagnostic is
reported explicitly. The unchanged budget is `1e-4 × (1 + |reference|)`.

The first frozen auditor failed a CPU-only import preflight because it
imported the GPU benchmark module, which imports Triton. No GPU science run
started from that packet. The corrected auditor imports only CPU Torch and
NumPy helpers. Its local fixture generator has the same parsed Python AST
as the pinned benchmark generator (SHA256
`4ec739a2abdfd229fb9f99bef635d215fc0f8a0a859872f9395235c8eb0154d5`),
and it recomputes the fixture SHA256 from the generated tensors before
accepting an audit.

The next distinct attempt passed that import preflight but exited before
producing a boundary report. The pinned TiRex cell passes its `state` as a
keyword argument, while the probe's forward pre-hook unpacked it as a second
positional argument. No boundary tensor pairs were completed. The corrected
hook requests keyword arguments explicitly and a separate CPU-only preflight
checks the same hook against `state=None` and a tensor state. On any later
failure, the probe now saves the arrays captured so far and the failing phase
before propagating the exception. Previous failed source and exit evidence
remain separate from this revision.

This experiment separates upstream input/state differences from a
same-input cell difference and checks whether a different incoming state is
needed for the first output mismatch. It does not yet distinguish recurrent
matrix multiplication from pointwise arithmetic. No GPU result or timing is
claimed by this source-only change. The corrected producer still needs CPU
hook preflight, a fresh finite GPU run, and a complete independent audit.
