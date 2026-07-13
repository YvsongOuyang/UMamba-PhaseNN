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

From PowerShell, use the bundled build script:

```powershell
cd D:\code\PYTHON\UMamba-PhaseNN\paper_ieee
.\build_paper.ps1
```

To remove generated LaTeX files before compiling:

```powershell
.\build_paper.ps1 -Clean
```

The script disables SyncTeX for command-line builds. This avoids the common
Windows case where TeXworks or a PDF viewer keeps `main.synctex.gz` locked and
makes the next build appear to hang.

The equivalent manual commands are:

```powershell
$bin = "D:\Program Files\MiKTeX\miktex\bin\x64"
$env:Path = "$bin;$env:Path"
pdflatex -synctex=0 -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -synctex=0 -interaction=nonstopmode -halt-on-error main.tex
pdflatex -synctex=0 -interaction=nonstopmode -halt-on-error main.tex
```

If MiKTeX asks to install a missing package, choose install. The compiled PDF is
`main.pdf`. If the package dialog is open behind another window, the compiler
will wait for it; bring the MiKTeX dialog to the front and finish or cancel it.

## Editing

- Write the paper body in `sections/*.tex`.
- Keep `main.tex` for document setup, title, abstract, and section imports.
- Put images in `figures/` using the filenames listed above.
- Missing figures render as placeholder boxes so the draft remains compilable.
