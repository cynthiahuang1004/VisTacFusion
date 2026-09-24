#!/bin/bash
# Two-way sync with the Overleaf project (id 6ab1c9afe47f399544728ef3).
#   ./sync_overleaf.sh pull   # Overleaf -> this directory
#   ./sync_overleaf.sh push   # this directory -> Overleaf (commits with message $2)
# Needs an Overleaf git token in $OVERLEAF_TOKEN (Overleaf > Account > Git integration).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
URL="https://git:${OVERLEAF_TOKEN:?set OVERLEAF_TOKEN}@git.overleaf.com/6ab1c9afe47f399544728ef3"
TMP=$(mktemp -d); git clone -q "$URL" "$TMP"
FILES="main.tex refs.bib ieeeconf.cls .gitignore sections figures data"
case "$1" in
  pull) for f in $FILES; do rm -rf "$HERE/$f"; cp -r "$TMP/$f" "$HERE/$f"; done; echo "pulled";;
  push) for f in $FILES; do rm -rf "$TMP/$f"; cp -r "$HERE/$f" "$TMP/$f"; done
        (cd "$TMP" && git add -A && git commit -q -m "${2:-sync from paper_RoboSoft}" && git push -q origin HEAD); echo "pushed";;
  *) echo "usage: $0 pull|push [message]"; exit 1;;
esac
rm -rf "$TMP"
