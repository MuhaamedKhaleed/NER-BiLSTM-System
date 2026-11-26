FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip wheel --no-cache-dir --wheel-dir /wheels .
FROM python:3.11-slim AS runtime
RUN useradd --create-home --uid 10001 app
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir --no-index --find-links=/wheels ner-bilstm && rm -rf /wheels
USER app
WORKDIR /home/app
ENTRYPOINT ["ner-bilstm"]
CMD ["--help"]
