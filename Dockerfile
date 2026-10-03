# Runtime image. Optional media providers are configured by the operator.
FROM python:3.12-slim@sha256:dddfd7e07f9d15aeeca61529320492139d21cac7f0070c00609243e51e4e0016 AS wheel-builder
RUN apt-get update && apt-get upgrade -y \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12@sha256:04d046b13e60d6bcec73cbc5e1cad25d680dea90c8573340950a0ac2d1aef424 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv build --wheel --out-dir /dist

FROM python:3.12-slim@sha256:dddfd7e07f9d15aeeca61529320492139d21cac7f0070c00609243e51e4e0016
RUN apt-get update && apt-get upgrade -y \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12@sha256:04d046b13e60d6bcec73cbc5e1cad25d680dea90c8573340950a0ac2d1aef424 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY chord.example.yaml ./chord.example.yaml
COPY --from=wheel-builder /dist /opt/chord-wheel
RUN uv sync --frozen --no-dev --no-install-project \
 && uv pip install --python /app/.venv/bin/python --no-deps /opt/chord-wheel/chord-*.whl \
 && python -m pip uninstall -y pip \
 && useradd --system --uid 10710 chord \
 && mkdir -p /data && chown chord /data
USER chord
ENV CHORD_DATA_DIR=/data CHORD_CONFIG=/app/chord.example.yaml PUBLIC_HOST=0.0.0.0 PUBLIC_PORT=8710 PYTHONDONTWRITEBYTECODE=1 PATH=/app/.venv/bin:$PATH
EXPOSE 8710
HEALTHCHECK --interval=10s --start-period=5s --timeout=5s --retries=3 CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PUBLIC_PORT', '8710'), timeout=4).status == 200 else 1)"
CMD ["python", "-m", "chord"]
