#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f /home/caohy/repositories/tashan_Manipulation/scripts/activate_curobo_v2_conda_env.sh ]]; then
  set +u
  # shellcheck source=/dev/null
  source /home/caohy/repositories/tashan_Manipulation/scripts/activate_curobo_v2_conda_env.sh
  set -u
else
  echo "missing CuroboV2 activation script" >&2
  exit 2
fi

python scripts/headless_smoke.py "$@"
