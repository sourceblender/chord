# Runtime image. Optional media providers are configured by the operator.
FROM python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f AS wheel-builder
COPY --from=ghcr.io/astral-sh/uv:0.12@sha256:04d046b13e60d6bcec73cbc5e1cad25d680dea90c8573340950a0ac2d1aef424 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv build --wheel --out-dir /dist

FROM python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
COPY --from=ghcr.io/astral-sh/uv:0.12@sha256:04d046b13e60d6bcec73cbc5e1cad25d680dea90c8573340950a0ac2d1aef424 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY chord.example.yaml ./chord.example.yaml
COPY --from=wheel-builder /dist /opt/chord-wheel
RUN uv sync --frozen --no-dev --no-install-project \
 && uv pip install --python /app/.venv/bin/python --no-deps /opt/chord-wheel/chord-*.whl \
 && useradd --system --uid 10710 chord \
 && mkdir -p /data && chown chord /data
USER chord
ENV CHORD_DATA_DIR=/data CHORD_CONFIG=/app/chord.example.yaml PUBLIC_HOST=0.0.0.0 PUBLIC_PORT=8710 PYTHONDONTWRITEBYTECODE=1 PATH=/app/.venv/bin:$PATH
EXPOSE 8710
HEALTHCHECK --interval=10s --start-period=5s --timeout=5s --retries=3 CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PUBLIC_PORT', '8710'), timeout=4).status == 200 else 1)"
CMD ["python", "-m", "chord"]
