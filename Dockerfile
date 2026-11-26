# Codex CLI (OpenAI), only used by the daily digests with DIGEST_PROVIDER=codex_cli (logged in once, by OAuth).
FROM node:22-bookworm-slim AS codex
RUN npm install -g --no-audit --no-fund @openai/codex@0.160.1 && npm cache clean --force

FROM python:3.12-slim

ARG LOCAL_WHISPER=0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY --from=codex /usr/local/bin/node /usr/local/bin/node
COPY --from=codex /usr/local/lib/node_modules/@openai /usr/local/lib/node_modules/@openai
RUN ln -s /usr/local/lib/node_modules/@openai/codex/bin/codex.js /usr/local/bin/codex

WORKDIR /app
COPY requirements.txt requirements-local.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && if [ "$LOCAL_WHISPER" = "1" ]; then pip install --no-cache-dir -r requirements-local.txt; fi

COPY app ./app
RUN useradd --create-home --uid 1000 bot && mkdir -p logs data /home/bot/.codex && chmod 700 /home/bot/.codex \
 && chown -R bot:bot /app /home/bot/.codex
USER bot
ENV CODEX_HOME=/home/bot/.codex

EXPOSE 8787
CMD ["sh", "-c", "uvicorn --factory app.main:app_factory --host 0.0.0.0 --port ${PORT:-8787} --workers 1 --no-access-log"]
