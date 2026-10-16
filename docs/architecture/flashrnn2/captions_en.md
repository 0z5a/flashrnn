# Captions and accessible descriptions

## fig01_architecture_overview

The pinned triton_fused inference baseline (0b84fc3) and the SHA-bound generic FlashRNN2 path both fuse the timestep loop, reuse recurrent weights, retain FP32 computation state and feed state locally. These are inherited mechanisms. Full history remains the default on both sides. The implemented delta is the opt-in separation of FP32 computation carry from observable snapshots, explicit full/hidden/last output modes, and full-call initialization context. This permits chunk continuation without using a rounded final snapshot. Local feedback denotes source-level dependence, not a claim that every shape is entirely register resident. Independent-prefetch overlap remains Proposed; C_t is backward-only and absent from the selected inference paths.

Alt text: Two homologous panels show R/b feeding recurrent MMA, U_t feeding state update, local S_t feedback and separate Y_t writes. The baseline exposes a final snapshot view; FlashRNN2 can expose independent FP32 chunk carry. Notes distinguish inheritance, implemented changes and proposed overlap.

## fig02_state_residency

The upper panel separates a CTA execution scope from physical storage. Representative H100 PTX confirms register MMA fragments, shared-memory ldmatrix exchange and CTA barriers; it does not establish a universal S_t/R/b/A_t allocation or zero spills. Global backing, required output history and boundary carry are separate objects. The lifetime strip covers init, t, t+1 and finish: R/b are reused inside a chunk and reloaded by a later kernel, while independent FP32 carry is exported at a boundary and read by the next serial chunk. Two carry arrays are not an asynchronous input-staging pipeline. Output lifetime follows full/hidden/last and the prepared borrowed-buffer contract. C_t is absent in this inference scope.

Alt text: A CTA scope contains register fragments, shared exchange and compute resources, with global weights, inputs and outputs below. Six lifetime rows identify load, ready, consume, carry and output boundaries. Full physical object mapping is explicitly TBD.

## fig03_dependency_pipeline

Three unrolled timesteps preserve the same-recurrence dependency. Both A_t and U_t must be available before state update; required Y_t stores share the actual execution path. The next hidden operand is cast and made visible before recurrent MMA, while c/n/m remain dependencies of the next update. Adjacent full recurrent MMAs are not arbitrarily parallel. This is a source-derived logical schedule, not a measured timeline; compiler layout/exchange may vary. C_t is absent from inference. The separate Proposed panel gives only a legal independent U_(t+1) prefetch lifecycle: issue, ready, consumer complete, reusable. Consumption requires input and state readiness; reuse waits for every consumer. Protocol and independent resources are TBD. Sequential chunk carry buffers do not prove overlap.

Alt text: Four logical lanes show MMA, input load, update with required stores, and cast/state-ready over three timesteps. Purple edges retain next-MMA and next-update dependencies. A separate dashed candidate strip shows issue, ready, consume and reuse without measured time units.

## fig04_execution_dispatch

The upper panel shows the implemented exact-shape ExecutionPolicy: nonempty offline tables require matching execution identity, keys contain cell/dtype/B/T/H/D, a missing key uses existing MMA16/4/B, and identity mismatch raises an error. No batch threshold is invented. The lower panel shows explicit caller-supplied configuration entering PreparedSequence. Shape, dtype, contiguity, one CUDA device, forward-only semantics, output mode and chunk checks precede the selected MMA/SIMT kernel and the preserved Y/carry contract. Invalid requests do not automatically switch to all upstream backends. Compiled resources constrain feasibility; there is no current runtime-budget query or complete automatic-fallback registry. Capability/resource-aware dispatch and cluster/WGMMA/FP8 remain Proposed. The connection is the experiment caller, not a request-time autotuner built into the public API.

Alt text: An exact-shape table branches to measured configuration, conservative MMA default, or identity error. The caller passes configuration to a separate prepared-execution panel, which validates inputs, launches MMA/SIMT and returns output/carry. Unknown resource limits and future policy extensions are isolated and labeled.
