"""Build the clean AlignTac paper: python build.py [--figures]."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    """Run a TeX tool from the project root and report useful failure output."""
    try:
        result = subprocess.run(
            command, cwd=ROOT, text=True, encoding='utf-8', errors='replace',
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        raise SystemExit(f'Unable to start {command[0]}: {exc}') from exc
    if result.returncode:
        print('\n'.join(result.stdout.splitlines()[-80:]), file=sys.stderr)
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures', action='store_true',
                        help='Also rebuild the standalone TikZ figure PDF exports.')
    args = parser.parse_args()
    latex = shutil.which('pdflatex')
    bibtex = shutil.which('bibtex') or shutil.which('bibtex.original')
    if not latex:
        raise SystemExit('PDFLaTeX is required. Install TeX Live or MiKTeX, or compile main.tex in Overleaf.')
    flags = ['-interaction=nonstopmode', '-halt-on-error', '-file-line-error']
    if args.figures:
        # Standalone TikZ wrappers live in figures/src/; the paper includes the exported PDFs in figures/.
        for source, target in [
            ('fig4_kshot_standalone', 'fig4_kshot'),
            ('fig5_labels_standalone', 'fig5_labels'),
            ('fig6_qual_standalone', 'fig6_qual'),
        ]:
            run([latex, *flags, '-output-directory=figures/src', f'figures/src/{source}.tex'])
            shutil.copy2(ROOT / f'figures/src/{source}.pdf', ROOT / f'figures/{target}.pdf')
    run([latex, *flags, 'main.tex'])
    if bibtex:
        run([bibtex, 'main'])
    elif not (ROOT / 'main.bbl').is_file():
        raise SystemExit('BibTeX is missing and main.bbl was not found.')
    else:
        print('Using the supplied main.bbl. Bibliography edits require BibTeX.')
    run([latex, *flags, 'main.tex'])
    run([latex, *flags, 'main.tex'])
    print('Created:', ROOT / 'main.pdf')


if __name__ == '__main__':
    main()
