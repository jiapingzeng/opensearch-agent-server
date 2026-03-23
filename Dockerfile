FROM python:3.12-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
COPY run_server.py .

RUN uv sync --frozen --no-dev

CMD ["uv", "run", "python", "run_server.py"]
