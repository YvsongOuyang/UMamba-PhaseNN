param(
    [string]$MiKTeXBin = "D:\Program Files\MiKTeX\miktex\bin\x64",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$paperDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pdflatex = Join-Path $MiKTeXBin "pdflatex.exe"
$bibtex = Join-Path $MiKTeXBin "bibtex.exe"

if (-not (Test-Path -LiteralPath $pdflatex)) {
    throw "pdflatex.exe was not found in: $MiKTeXBin"
}
if (-not (Test-Path -LiteralPath $bibtex)) {
    throw "bibtex.exe was not found in: $MiKTeXBin"
}

Set-Location -LiteralPath $paperDir

if ($Clean) {
    $generatedExtensions = @(
        ".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls",
        ".log", ".out", ".synctex.gz"
    )
    foreach ($extension in $generatedExtensions) {
        Get-ChildItem -LiteralPath $paperDir -File -Filter "*$extension" |
            Remove-Item -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Program,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Program $($Arguments -join ' ')"
    }
}

# SyncTeX is disabled for command-line builds because an open PDF/TeXworks
# session can keep main.synctex.gz locked on Windows.
$latexArguments = @(
    "-synctex=0",
    "-interaction=nonstopmode",
    "-halt-on-error",
    "main.tex"
)

Write-Host "[1/4] Running pdfLaTeX"
Invoke-CheckedCommand -Program $pdflatex -Arguments $latexArguments

Write-Host "[2/4] Running BibTeX"
Invoke-CheckedCommand -Program $bibtex -Arguments @("main")

Write-Host "[3/4] Resolving references"
Invoke-CheckedCommand -Program $pdflatex -Arguments $latexArguments

Write-Host "[4/4] Finalizing PDF"
Invoke-CheckedCommand -Program $pdflatex -Arguments $latexArguments

Write-Host "Build complete: $(Join-Path $paperDir 'main.pdf')"
