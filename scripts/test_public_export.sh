#!/usr/bin/env bash
# Clean public-style source export -> its own locked environment -> its own tests.
# A public test that imports a private module, or reads a file the export leaves
# out, fails here rather than in the first stranger's clone.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

cd "$root"
if [[ -f scripts/scrub_check.py ]]; then
  # In the private working repository, test the exact proposed public tree.
  python scripts/scrub_check.py --export "$tmp/export"
  checkout="$tmp/export"
else
  # In the fresh-history public repository, the checkout is already that tree.
  checkout="$root"
fi
test ! -e "$checkout/deploy"
test ! -e "$checkout/evidence"
test ! -e "$checkout/harness"
test ! -e "$checkout/qa/redteam"
test ! -e "$checkout/.github/workflows/ci.yml"

cd "$checkout"
python scripts/public_boundary.py .
# No inherited environment: a fresh clone has none of these, and each one changes
# what the tests or chord's startup would see (profile, keys, ports).
clean=(env -u CHORD_CONFIG -u CHORD_MANIFEST -u CHORD_REGISTRY -u CHORD_DATA_DIR
       -u PYTHONPATH -u VIRTUAL_ENV
       -u CHORD_API_KEY -u CHORD_CLIENT_KEYS_JSON -u CHORD_ACCEPT_LEGACY_CLIENT_KEY
       -u PUBLIC_HOST -u PUBLIC_PORT -u INTERNAL_PORT)
"${clean[@]}" uv sync --locked --all-groups
"${clean[@]}" uv run pytest -q -rs -p no:cacheprovider
"${clean[@]}" uv run ruff check src/chord
"${clean[@]}" uv run ruff check --select F tests conftest.py
"${clean[@]}" uv run pyright
"${clean[@]}" uv run python -m compileall -q src tests

# The README text-only quickstart, run as written: its runtime-only install,
# example config unchanged, no API key, default loopback ports, under the same
# clean environment, so no runner variable can stand in for a promised default.
lock_before=$(python -c 'import hashlib;print(hashlib.sha256(open("uv.lock","rb").read()).hexdigest())')
"${clean[@]}" uv sync --frozen --no-dev
lock_after=$(python -c 'import hashlib;print(hashlib.sha256(open("uv.lock","rb").read()).hexdigest())')
test "$lock_before" = "$lock_after" || { echo "uv sync --frozen --no-dev changed uv.lock"; exit 1; }
"${clean[@]}" uv run --no-sync python -c 'import pytest' 2>/dev/null \
  && { echo "dev dependencies still installed; the walk would not test a runtime-only install"; exit 1; }

# A process already on one of these ports could answer for the one under test.
for port in 11434 8710 8711; do
  if curl --silent --output /dev/null --max-time 2 "http://127.0.0.1:$port/"; then
    echo "port $port is already in use"; exit 1
  fi
done

python tests/text_smoke_backend.py llama3.1 &
backend=$!
server=
trap 'kill $backend ${server:-} 2>/dev/null || true; rm -rf "$tmp"' EXIT
# The stub answers any request (even a refused one) once it is bound.
for _ in $(seq 1 40); do
  code=$(curl --silent --output /dev/null --write-out '%{http_code}' --max-time 2 \
    -X POST http://127.0.0.1:11434/v1/chat/completions -d '{}' || true)
  [[ $code != 000 ]] && break
  sleep 0.25
done
[[ $code != 000 ]] || { echo "stub backend never bound 11434"; exit 1; }

cp chord.example.yaml "$tmp/chord.yaml"
"${clean[@]}" CHORD_CONFIG="$tmp/chord.yaml" uv run --no-sync python -m chord --check-config
"${clean[@]}" CHORD_CONFIG="$tmp/chord.yaml" uv run --no-sync python -m chord > "$tmp/server.log" 2>&1 &
server=$!
for _ in $(seq 1 40); do
  curl --silent --fail http://127.0.0.1:8710/health >/dev/null && break
  sleep 1
done
kill -0 "$backend" 2>/dev/null || { echo "stub backend exited"; exit 1; }
kill -0 "$server" 2>/dev/null || { cat "$tmp/server.log"; echo "chord exited"; exit 1; }
curl --silent --show-error --fail http://127.0.0.1:8710/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"chord-1-poly","messages":[{"role":"user","content":"hello"}]}' > "$tmp/reply.json" \
  || { cat "$tmp/server.log"; exit 1; }
python - "$tmp/reply.json" <<'PY'
import json, sys
reply = json.load(open(sys.argv[1]))
assert reply["choices"][0]["message"]["content"] == "text-only smoke passed", reply
print("README quickstart: runtime-only install, chat 200 through the example config")
PY
