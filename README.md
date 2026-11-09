# LaschukResume

WhatsApp voice-note → transcript → short Portuguese summary, via **Evolution API v2**.

When someone in a configured group sends an audio or voice note, LaschukResume:

1. receives the `messages.upsert` webhook from Evolution,
2. keeps only messages from the configured group (by `GROUP_JID` or `GROUP_NAME`), ignoring its own messages and duplicates,
3. downloads the media as base64 (`/chat/getBase64FromMediaMessage`, or inline `base64` when the webhook sends it),
4. transcribes with the **OpenAI Whisper API** or local **faster-whisper**,
5. summarizes **in Portuguese, in 2–4 short lines**, with **Claude Haiku** or OpenAI,
6. replies in the same group **quoting the original audio**.

## License notice (not OSI open source)

LaschukResume is **source-available**, not OSI open source. See [`LICENSE`](./LICENSE).

You may use and modify the code **only if**:

- the product name stays **LaschukResume** (no rebranding / white-label rename);
- replies posted in WhatsApp groups end with the footer `_transcrito por github.com/ustoppble/laschuk-resume_` (DM transcripts forwarded to the Áudios group carry no footer).

Shipping under another name is forbidden and terminates the license grant.

Public repo: https://github.com/ustoppble/laschuk-resume

## Requirements

- Python 3.11+ (with `start.sh`) **or** Docker + Docker Compose
- An Evolution API v2 instance connected to the WhatsApp number in the target group
- `ANTHROPIC_API_KEY` (or `CLAUDE_API_KEY`) for summaries, or `OPENAI_API_KEY` + `OPENAI_SUMMARY_MODEL`
- `OPENAI_API_KEY` for Whisper, or local faster-whisper

## Setup

```bash
cp .env.example .env
$EDITOR .env           # fill EVOLUTION_*, keys, GROUP_NAME or GROUP_JID, WEBHOOK_TOKEN
./start.sh            # or: docker compose up -d --build
./stop.sh
tail -f logs/laschuk-resume.log
curl -s localhost:8787/health
```

Register the Evolution webhook at `POST $EVOLUTION_URL/webhook/set/$EVOLUTION_INSTANCE` pointing to `http://<bot-host>:8787/webhook` with `MESSAGES_UPSERT` and preferably `base64: true`. Full `curl` example is in `.env.example` comments / historical hillside docs pattern:

```bash
set -a; source .env; set +a
BOT_URL="http://<bot-host>:${PORT:-8787}/webhook"
curl -X POST "$EVOLUTION_URL/webhook/set/$EVOLUTION_INSTANCE" \
  -H "apikey: $EVOLUTION_API_KEY" -H "Content-Type: application/json" \
  -d "{
    \"webhook\": {
      \"enabled\": true,
      \"url\": \"$BOT_URL\",
      \"headers\": {\"x-webhook-token\": \"$WEBHOOK_TOKEN\"},
      \"byEvents\": false,
      \"base64\": true,
      \"events\": [\"MESSAGES_UPSERT\"]
    }
  }"
```

## oc76

```bash
docker compose -f docker-compose.yml -f docker-compose.oc76.yml up -d --build
```

Joins the external `evolution-leonora_default` network and binds the port on loopback only.

## Tests

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest
```
