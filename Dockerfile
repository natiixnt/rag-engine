FROM rust:1.74-slim AS rust-builder

WORKDIR /build
COPY rust_core/ ./rust_core/

RUN apt-get update && apt-get install -y python3-dev && rm -rf /var/lib/apt/lists/*
RUN cd rust_core && cargo build --release


FROM python:3.11-slim AS python-builder

WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir build && \
    python -m build --wheel --outdir /dist


FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=python-builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

COPY --from=rust-builder /build/rust_core/target/release/librag_engine_core.so /usr/local/lib/python3.11/site-packages/rag_engine_core.so

COPY config.yaml /app/config.yaml

EXPOSE 8000

CMD ["uvicorn", "rag_engine.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
