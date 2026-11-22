FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY emkambo ./emkambo
COPY tests ./tests
RUN pip install --no-cache-dir ".[test]"

# Mount a working directory holding config.yaml and data/ at /work
WORKDIR /work
ENTRYPOINT ["emkambo"]
CMD ["--help"]
