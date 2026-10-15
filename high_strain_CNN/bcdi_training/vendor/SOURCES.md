# Simulator Source

`simulator_source.py` preserves the user-supplied
`deliverables/bcdi_2d_simulator_512/generate_data.py`. The production wrapper
loads only its geometry/configuration/FFT definitions through an AST allowlist;
the plotting imports and standalone CLI are not executed. All generated
datasets record the SHA256 of this local pinned file.

Original algorithm: Xi Yu et al., *Ultrafast Bragg coherent diffraction imaging
of epitaxial thin films using deep complex-valued neural networks*, npj
Computational Materials 10, 24 (2024), DOI: 10.1038/s41524-024-01208-7.

Original code: https://github.com/XFELDataScience/Complex-NNphase/blob/main/Simulated%20Date/generate_data.py

The user-supplied adaptation adds configurable pixel geometry, 512-square native
sampling, validation and the author/radians phase switch. Its physical formulas
and random draw order are reused without changes. This notice does not relicense
the original authors' work.
