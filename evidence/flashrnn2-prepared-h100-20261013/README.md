# Source-bound H100 evidence views

See [the result and limitations document](../../docs/flashrnn2/prepared-h100-results.md).
`summary.json`, `statistics-audit.json` and `RESULTS.md` are byte-identical
copies of the independent optimization analysis. `qualified-source.json` binds
the exercised core files. CSVs and the state-qualification summary come from
separate recorded cohorts and must not be silently combined as one experiment.

The harness snapshot and its SHA manifest preserve the frozen scientific code.
Large saved arrays and complete raw archives remain separately preserved under
the hashes referenced by their original reports. This source PR does not ship
those raw arrays or turn a summary into a fresh GPU qualification.
