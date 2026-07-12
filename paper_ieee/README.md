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

This workspace uses MiKTeX installed at:

```text
D:\Program Files\MiKTeX\miktex\bin\x64
```

In TeXworks:

1. Open `main.tex`.
2. Select `pdfLaTeX` in the green compile drop-down.
3. Run `pdfLaTeX`.
4. Run `BibTeX`.
5. Run `pdfLaTeX` twice more.

From PowerShell:

```powershell
$bin = "D:\Program Files\MiKTeX\miktex\bin\x64"
$env:Path = "$bin;$env:Path"
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

If MiKTeX asks to install a missing package, choose install. The compiled PDF is
`main.pdf`.

## Editing

- Write the paper body in `sections/*.tex`.
- Keep `main.tex` for document setup, title, abstract, and section imports.
- Put images in `figures/` using the filenames listed above.
- Missing figures render as placeholder boxes so the draft remains compilable.
