# Supplied author source

`codes_for_BCDI_dataset_creation/` contains the user-supplied simulation source
for the high-strain BCDI paper. These are the original files, not our adapter.
All eight Python modules, the example notebook (including its original outputs),
and both supplied gold potential resources are copied byte-for-byte.

`author_source_manifest.json` records relative paths, byte sizes, and SHA-256
hashes. `.gitattributes` disables line-ending conversion for these files so a
Linux checkout preserves the same content. Do not edit the vendor files to
change generation behavior; keep compatibility/CLI changes in `simulation/`.

Generated `__pycache__` files and the unused `Main_files/atomsk` executable are
excluded. The active source creates atoms through Python/ASE and does not call
that executable. No standalone LICENSE file was present in the supplied source;
existing author/resource notices are retained, and no new license is asserted.

The generation and author-evaluation CLIs use this directory by default.
`--author-code-dir` remains available to override it for a source comparison.
PyNX/PyCUDA/CUDA and the scientific Python dependencies still need to be
installed separately; vendoring the source does not install its dependencies.
