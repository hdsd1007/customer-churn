# STAGE 1 BUILDER STAGE
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

COPY pyproject.toml uv.lock .

RUN uv venv .venv && \
    uv sync --no-dev

# STAGE 2 RUNTIME
FROM python:3.12-slim-bookworm

WORKDIR /app

COPY --from=builder /app/.venv .venv 

COPY src/ src/
COPY artifacts/ artifacts/

ENV PATH="/app/.venv/bin:$PATH"

ENV PYTHONPATH="/app"

EXPOSE 8000

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

