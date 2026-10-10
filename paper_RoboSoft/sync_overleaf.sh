#!/bin/bash
# Two-way sync with the Overleaf project (id 6ab1c9afe47f399544728ef3).
#   ./sync_overleaf.sh pull   # Overleaf -> this directory
#   ./sync_overleaf.sh push   # this directory -> Overleaf (commits with message $2)
# Needs an Overleaf git token in $OVERLEAF_TOKEN (Overleaf > Account > Git integration).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
URL="https://git:${OVERLEAF_TOKEN:?set OVERLEAF_TOKEN}@git.overleaf.com/6ab1c9afe47f399544728ef3"
TMP=$(mktemp -d); git clone -q "$URL" "$TMP"
FILES="main.tex main_long.tex architecture_diagram.tex architecture_diagram_training_inference.tex teaser_diagram.tex table_style.tex refs.bib ieeeconf.cls .gitignore sections sections_long figures data"
case "$1" in
  pull) for f in $FILES; do rm -rf "$HERE/$f"; cp -r "$TMP/$f" "$HERE/$f"; done
        git -C "$TMP" rev-parse HEAD > "$HERE/.overleaf_head"; echo "pulled";;
  push) # refuse to overwrite edits made on Overleaf since the last pull/push (pass "force" as $3 to override)
        if [ -f "$HERE/.overleaf_head" ] && [ "$(git -C "$TMP" rev-parse HEAD)" != "$(cat "$HERE/.overleaf_head")" ] && [ "$3" != force ]; then
          echo "Overleaf has new commits since the last sync:"; git -C "$TMP" log --oneline "$(cat "$HERE/.overleaf_head")..HEAD" | head; echo "run 'pull' first (or push ... force)"; rm -rf "$TMP"; exit 2; fi
        for f in $FILES; do rm -rf "$TMP/$f"; cp -r "$HERE/$f" "$TMP/$f"; done
        (cd "$TMP" && git add -A && git commit -q -m "${2:-sync from paper_RoboSoft}" && git push -q origin HEAD)
        git -C "$TMP" rev-parse HEAD > "$HERE/.overleaf_head"; echo "pushed";;
  *) echo "usage: $0 pull|push [message]"; exit 1;;
esac
rm -rf "$TMP"
