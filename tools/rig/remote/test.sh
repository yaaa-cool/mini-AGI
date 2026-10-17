#!/bin/bash
# test.sh : the CPU test suite on the rig's synced code. Exit is pytest's.
# shellcheck source-path=SCRIPTDIR source=lib.sh
. "$(dirname "$0")/lib.sh"
cd "$REPO" || exit 1
CUDA_VISIBLE_DEVICES="" "$PY" -m pytest -q tests 2>&1 | tail -n 5
exit "${PIPESTATUS[0]}"
