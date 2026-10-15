# FlashRNN2 R tiling and high-concurrency evidence (curated)

One directory per run, named by the run labels used in `docs/flashrnn2/` (M0 … M7, HC0 … HC4, dispatch check,
table fill). Each holds the paired-timing summaries (`analysis/m4-summary.*`), frozen comparison and policy tables,
qualification records (`science/phase0/*/m6.json`), selection records, run receipts and archive manifests. Keys that
identify hosts, processes or storage paths were removed. Raw per-start timings and task logs are not published; each
run's `control/finalize-result.json` keeps the SHA256 of its verified raw archive. JSON records here are scrubbed
copies, so their bytes differ from the SHA256 values listed in the archive manifests, which describe the originals.

Commit hashes cited in `docs/flashrnn2/` ("development commit …") and `source_commit` in the packaged tables name
the development history each run's source was staged from. That history is not published because its evidence
carries cluster identifiers; the code in this branch is token-identical to it apart from comments and docstrings
that cited run identifiers.
