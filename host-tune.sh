#!/usr/bin/env bash
# Host setting the published numbers were measured with (run as root;
# start.sh calls it through sudo unless HOST_TUNE=0).
#   host-tune.sh <gpu, e.g. 0>
# Sets the 170 W power limit on the card. Nothing else on the host changes.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "host-tune.sh: run as root" >&2; exit 1; }
gpu=${1:?gpu}
nvidia-smi -i "$gpu" -pl 170 >/dev/null
echo "host tuned: 170 W on GPU $gpu"
