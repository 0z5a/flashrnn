# H20 C128 stacked-model E2E evidence

The complete offbox is committed here as three ordered parts. Reassemble it
with `cat own-offbox-raw.tar.gz.part-* > own-offbox-raw.tar.gz` and verify the
51,152,692-byte archive with `shasum -a 256 own-offbox-raw.tar.gz`. Its SHA256
is `be8444c01bb97cb7f9e18c05317035c82192ee75d39c22a5371727cb2cc8792d`.

| Part | Bytes | SHA256 |
| --- | ---: | --- |
| `part-00` | 20,000,000 | `56f702b58fc9c05fb98db30243b75263e82e4dd893f8aba5fc51cd8fd1179724` |
| `part-01` | 20,000,000 | `d30109a5cd3a9efb26e863de04c5602c8ab14a3060365f3d147372ac9d55bd8d` |
| `part-02` | 11,152,692 | `6b411e3d2b1f445a2b5f756da0b94952893f78a7307b7f7ab8384917c20b34bb` |

The archive contains 159 payload files plus its manifest. The manifest and
each archived payload byte were rechecked; the frozen source manifest has
123 files with matching hashes. The six `.qualification.pt` files are in this
archive and bind to the committed metadata by SHA.

This directory contains all 120 raw paired JSONL rows, six metadata files,
six child logs, six paired-analysis JSON and Markdown files, six local NumPy
audit results, the cuDNN packing probe, controller receipt, independent review,
offbox manifest, [whole-device handback](WHOLE.json) and fresh resource-clear
receipt. The local NumPy rerun
recomputed 252 tensor pairs and agreed exactly with the independent offbox
review. The paired analyzer reproduced the independent speed summaries.

The client's grant JSON and the remotely saved sorted-key JSON have different
byte hashes but identical parsed values. The controller receipt binds the
remote grant hash; the handback records both hashes and verifies the original
GPU and IO locks, empty compute-app list, nine absent actors and natural exit 0
for all seven children plus the wrapper/controller. No successor grant was
inferred from this handback.

The installed cuDNN backend source is in the archive with SHA256
`c23b6956839f10ae62913c199e4cd6b0acad64819665e6849903754d185c1a48`.
Its BF16 exclusion applies to this PyTorch 2.12.1+cu130 runtime; the probe
records the resulting unpacked weights and repeated warnings.
