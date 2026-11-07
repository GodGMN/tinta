#!/usr/bin/env bash
# The session half of the integration-smoke job. Run it inside `dbus-run-session`
# in a throwaway container, with the .deb installed.
#
# It unlocks a fresh login keyring with an empty password (the session is not
# reachable from outside), runs the engine protocol smoke test, and then runs
# the tinta-core test that creates the library key through that keyring.
#
#   WORK=/tmp/work CLIP=/tmp/work/speech.wav \
#     TINTA_CORE_TEST=/path/to/tinta_core-<hash> \
#     dbus-run-session -- scripts/fork/integration-session.sh
set -euo pipefail

: "${WORK:?set WORK to a writable folder}"
: "${CLIP:?set CLIP to a speech audio file}"
: "${TINTA_CORE_TEST:?set TINTA_CORE_TEST to the built tinta-core test binary}"
ENGINE="${TINTA_ENGINE:-/usr/bin/tinta-engine}"

# `--unlock` creates the login keyring when it does not exist and unlocks it
# with the empty password. The daemon it starts serves org.freedesktop.secrets
# on the session bus that dbus-run-session provides.
eval "$(printf '\n' | gnome-keyring-daemon --unlock --components=secrets)"
sleep 1

python3 "$(dirname "$0")/engine-smoke.py" \
  --engine "$ENGINE" \
  --work "$WORK" \
  --clip "$CLIP"

# The keyring test creates the item, reads it back, and deletes it again.
TINTA_SECRET_SERVICE_TEST=1 "$TINTA_CORE_TEST" \
  --ignored --exact --nocapture \
  keys::tests::creates_a_library_key_through_the_secret_service
