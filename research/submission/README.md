# habitID-Net paper: compile in Overleaf or with pdflatex

Compile from this folder (the package root). Paths are relative to it.

Compile order (run twice so references resolve):

    pdflatex main.tex
    pdflatex main.tex

Overleaf: upload this whole folder, set main.tex as the main document, and compile with pdfLaTeX.
The IEEEtran class is needed (it is included with Overleaf). The bibliography is inline (thebibliography);
entries that still need manual verification are listed in REFERENCES_TO_VERIFY.md in the research folder.

Contents:
- main.tex                 the paper
- paper_assets/tables/     generated LaTeX tables (from results/ only)
- paper_assets/figures/    generated figures (PDF and PNG)
- paper_assets/numbers.tex generated \newcommand macros for numbers quoted in the text

The TeX source has one remaining \TBD: the acknowledgment (course, instructor, compute resources).
