#!/usr/bin/env bash
# Evaluate a checkpoint, reporting metrics per modality config (both / tactile / rgb).
# [TODO/USER] wire checkpoint loading into a small eval entrypoint; for now eval runs at the
# end of each train cycle (see vistacfusion/engine/eval.py::evaluate).
set -euo pipefail
echo "Per-config evaluation runs inside training (vistacfusion/engine/eval.py)."
echo "Add a --resume/--eval-only entrypoint when checkpointing is wired (CLAUDE.md 10)."
