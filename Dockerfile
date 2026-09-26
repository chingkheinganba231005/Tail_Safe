# The full TailSafe app (simulator + API + web UI) in one container.
#
#   docker build -t tailsafe .
#   docker run -p 7860:7860 tailsafe        # then open http://localhost:7860
#
# The same image runs on hosts such as Hugging Face Spaces (Docker), which
# gives everyone a public URL; see README → "Run the full app".

FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NUMBA_CACHE_DIR=/tmp/numba-cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    TAILSAFE_CACHE_DIR=/tmp/tailsafe-cache \
    PORT=7860
WORKDIR /app
COPY pyproject.toml README.md ./
COPY config ./config
COPY schemas ./schemas
COPY tailsafe ./tailsafe
RUN pip install -e ".[surrogate]"
COPY --from=web /web/dist ./web/dist
RUN useradd --create-home --uid 1000 tailsafe && chown -R tailsafe /app
USER tailsafe
EXPOSE 7860
CMD ["sh", "-c", "exec uvicorn tailsafe.api.app:app --host 0.0.0.0 --port ${PORT}"]
