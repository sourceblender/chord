#!/usr/bin/env bash
# Clean public-style source export -> text-only image -> one real API request.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
backend_pid=
container=chord-text-smoke-$$
cleanup() {
  docker logs "$container" 2>/dev/null || true
  docker rm -f "$container" >/dev/null 2>&1 || true
  [[ -z $backend_pid ]] || kill "$backend_pid" 2>/dev/null || true
  rm -rf "$tmp"
}
trap cleanup EXIT

cd "$root"
if [[ -f scripts/scrub_check.py ]]; then
  python scripts/scrub_check.py --export "$tmp/export"
  checkout="$tmp/export"
else
  checkout="$root"
fi
test ! -e "$checkout/deploy"
test ! -e "$checkout/vendor"
test ! -e "$checkout/harness"
test ! -e "$checkout/qa/redteam"
test ! -e "$checkout/.github/workflows/ci.yml"
docker build -t chord:text-smoke "$checkout"
docker run --rm -i --entrypoint /app/.venv/bin/python chord:text-smoke -I -S - \
  /opt/chord-wheel /app/.venv \
  < "$root/scripts/verify_runtime_wheel.py"

python tests/text_smoke_backend.py &
backend_pid=$!
cat > "$tmp/chord.yaml" <<'YAML'
endpoints:
  chat:
    type: openai-chat
    url: http://127.0.0.1:11434/v1
    model: example-local-model
routing:
  chat: {endpoint: chat}
YAML
docker run --detach --name "$container" --network host \
  -e PUBLIC_HOST=127.0.0.1 \
  -e CHORD_API_KEY=local-smoke-key \
  -e CHORD_CONFIG=/app/chord.yaml \
  -v "$tmp/chord.yaml:/app/chord.yaml:ro" chord:text-smoke >/dev/null

for _ in $(seq 1 40); do
  if curl --silent --fail --header 'Authorization: Bearer local-smoke-key' \
    http://127.0.0.1:8710/health >/dev/null; then break; fi
  sleep 1
done
curl --silent --show-error --fail --header 'Authorization: Bearer local-smoke-key' \
  --header 'Content-Type: application/json' \
  --data '{"model":"chord-1-poly","messages":[{"role":"user","content":"hello"}]}' \
  http://127.0.0.1:8710/v1/chat/completions > "$tmp/reply.json"
python - "$tmp/reply.json" <<'PY'
import json, sys
reply = json.load(open(sys.argv[1]))
assert reply['choices'][0]['message']['content'] == 'text-only smoke passed', reply
print('text-only public-style image: chat 200, expected answer')
PY
