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
        intro_dir = ROOT / 'figures' / 'intro_figure'
        run([latex, *flags, '-output-directory=figures/intro_figure',
             'figures/intro_figure/motivation_standalone.tex'])
        shutil.copy2(intro_dir / 'motivation_standalone.pdf',
                     intro_dir / 'AlignTac_Figure1_Updated.pdf')
        for source, target in [
            ('pipeline_standalone', 'fig_arch_training_inference'),
            ('fig_qual_standalone', 'fig_qual'),
            ('fig_kshot_standalone', 'fig_kshot'),
            ('fig_labels_standalone', 'fig_labels'),
            ('fig_renders_standalone', 'fig_renders'),
        ]:
            run([latex, *flags, '-output-directory=figures', f'figures/{source}.tex'])
            shutil.copy2(ROOT / f'figures/{source}.pdf', ROOT / f'figures/{target}.pdf')
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
