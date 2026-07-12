# IEEE Paper Draft

This folder contains an IEEE-style LaTeX paper skeleton for the AutoPhaseNN and
UMamba phase retrieval experiments.

## Files

- `main.tex`: main IEEEtran document.
- `sections/*.tex`: modular paper sections.
- `references.bib`: BibTeX references.
- `figures/`: place experiment figures here.

## Suggested Figure Filenames

The current draft references these filenames:

- `figures/fig_autophasenn_reproduction.png`
- `figures/fig_umamba_original.png`
- `figures/fig_umamba_soft_threshold.png`
- `figures/fig_training_curves.png`

If a figure is missing, `main.tex` will render a placeholder box.

## Build

Use a TeX distribution with `IEEEtran` installed. For example:

```bash
latexmk -pdf main.tex
```

or:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The current Windows workspace did not have `latexmk`, `pdflatex`, or `xelatex`
available, so compilation was not run locally.
