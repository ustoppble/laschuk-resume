# LaschukResume

WhatsApp voice-note → transcript → short Portuguese summary, via **Evolution API v2**.

When someone in a configured group sends an audio or voice note, LaschukResume:

1. receives the `messages.upsert` webhook from Evolution,
2. keeps only messages from the configured group (by `GROUP_JID` or `GROUP_NAME`), ignoring its own messages and duplicates,
3. downloads the media as base64 (`/chat/getBase64FromMediaMessage`, or inline `base64` when the webhook sends it),
4. transcribes with the **OpenAI Whisper API** or local **faster-whisper**,
5. summarizes **in Portuguese, in 2–4 short lines**, with **Claude Sonnet 5.5** (default, `ANTHROPIC_MODEL`) or OpenAI (`OPENAI_SUMMARY_MODEL`),
6. replies in the same group **quoting the original audio**.

While it works, the original group audio carries a status **reaction**: ⏳ as soon as the audio is accepted (before download), ✅ once the summary is posted, and ❌ if anything fails (download, transcription, empty transcript or summary, or sending the reply), so an audio never stays on ⏳. A failed reaction is only logged and never blocks the summary. Set `GROUP_REACTIONS=0` to turn it off. The same applies to audios in the Áudios (transcripts) group. DMs and your own chat never get a reaction: there the reaction would be visible to the sender.

## Daily digests (optional, off by default)

Two scheduled digests, each with its own switch. Both read the message history Evolution already stores
(`/chat/findMessages`) at run time, so text messages are never copied to a new store, a restart loses nothing,
and any past day can be previewed with the same code. Evolution must keep message history
(`DATABASE_SAVE_DATA_NEW_MESSAGE=true`). Audios enter through their transcripts, cached in a small SQLite
(`DIGEST_DB`, deleted after `DIGEST_RETENTION_HOURS`); the webhook pipeline fills that cache too, so an audio is
not transcribed twice.

- **Group digest** (`DIGEST_GROUP_ENABLED=1`, `DIGEST_GROUP_JID`, `DIGEST_GROUP_TIME=08:00`): only the **top 3
  highlights** of the previous day, 00:00 to 24:00, picked for the best insight for the group, not for the
  number of messages: decisions or announcements, news, launches or shared content, tips or lessons that serve
  everyone, and discussions of ideas, strategy or results. Product bug reports and individual support questions
  (tool limits, setup, install, environment errors, "how do I") are never highlighted nor mentioned, and a code
  filter drops any that slip through (the owner digest still lists bugs). Each highlight is a 1-line title that
  states the point, 1 line on what happened with a real WhatsApp mention (`@<number>` in the text + Evolution's
  `mentioned` field) of who started it, and a `Sacada:` line with a specific takeaway drawn from the conversation;
  a highlight without a real takeaway is dropped. A weak day gets fewer than 3, and a day without any highlight
  sends nothing. On a long day each part proposes candidates and one final call picks the top 3 of the whole day.
  It never says or hints that someone went unanswered: the prompt forbids it, and a draft that does is
  regenerated once, then the offending sentence is removed.
  Right after it (`DIGEST_GROUP_FEEDBACK_POLL=1`, the default) the bot posts a poll "Esse resumo foi útil?" with
  SIM / NÃO and 1 vote per person (`/message/sendPoll`); no highlight, no digest and no poll. Votes come in through
  the same `messages.upsert` webhook: Evolution v2 decrypts them itself (it needs the poll it sent, stored in its
  database) and sends the chosen option names; the last vote per person counts and an empty vote removes it. The
  next owner digest gets one line under its header: `Resumo do grupo de ontem: X SIM, Y NÃO`.
- **Owner digest** (`DIGEST_OWNER_ENABLED=1`, `DIGEST_OWNER_TIME=07:00`): sent privately (`DIGEST_OWNER_TARGET_JID`,
  default the owner's own chat). It reads every chat of the owner's number (`DIGEST_OWNER_INSTANCE`) that had
  activity the day before and lists what is pending for the owner: unanswered messages or audios, questions,
  requests, deadlines and the owner's own promises ("I'll send it tomorrow"). Grouped by conversation, most
  urgent first; each item says who, what and since when. 1x1 chats are read in full. In groups only what
  involves the owner is read (an @mention, a reply to or quote of the owner's message, the owner's first name, or
  the owner speaking) plus a few messages around it; groups where none of that happened are dropped **before any
  model call**, so hundreds of quiet groups cost nothing. Narrow it with `DIGEST_OWNER_SCOPE`,
  `DIGEST_OWNER_INCLUDE_JIDS` and `DIGEST_OWNER_EXCLUDE_JIDS`.

The scheduler runs inside the app (one worker) and records each run in SQLite, so a restart or a deploy near the
scheduled time never posts twice and still sends late, up to `DIGEST_CATCHUP_MINUTES`. Days are split into model
calls of `DIGEST_BATCH_CHARS` characters (long group days are summarized in parts and merged), which keeps each
call inside low rate-limit tiers. `DIGEST_MODEL` picks a model for the 2 digests only (empty = the
summarizer's model); on a free tier with a per-day token quota per model, a smaller model such as
`openai/gpt-oss-20b` keeps the digests from eating the quota of the per-audio summaries, which always keep
`OPENAI_SUMMARY_MODEL`. `DIGEST_PROVIDER=codex_cli` writes the 2 digests with the Codex CLI already logged in with a ChatGPT account
(OAuth, `codex login --device-auth` once) instead of an API key, with `DIGEST_CLI_MODEL`. It runs as a plain text
model: read-only sandbox, a throwaway empty directory, no user config or rules, no web search, the prompt on stdin,
and a timeout (`DIGEST_CLI_TIMEOUT_SECONDS`); the CLI must be installed and logged in where the app runs. Preview a
past day without sending anything:

```bash
python -m app.digest group --date 2026-11-24   # as it would be posted at DIGEST_GROUP_TIME that day
python -m app.digest owner --date 2026-11-24   # as it would be sent at DIGEST_OWNER_TIME the next morning
```

Same prompt-injection boundary as the audio summary: message text, names and transcripts go as JSON data in the
user message, never in the system prompt, and the model only sees short handles (`P1`, `C1`, message numbers),
never phone numbers or JIDs.

## License notice (not OSI open source)

LaschukResume is **source-available**, not OSI open source. See [`LICENSE`](./LICENSE).

You may use and modify the code **only if**:

- the product name stays **LaschukResume** (no rebranding / white-label rename);
- audios you send to yourself (your own chat) are transcribed too: the reply (full transcript + summary) always quotes the audio in that same chat, even when `DM_TRANSCRIPT_TARGET_JID` is set; audios you send to other people stay ignored.
- audios sent or forwarded inside the `DM_TRANSCRIPT_TARGET_JID` group (Áudios) are transcribed in place: quoted reply with the full transcript + summary, no footer. That group is accepted automatically; it does not need to be repeated in `GROUP_JID`. The bot's own transcripts are text, so they are never reprocessed.
- 1x1 audios from other people go to that group (or the owner chat) as full transcript + summary, headed by the sender's name and phone, e.g. `🎙️ Nome · +55 11 91234-5678 (14:54)` (name or phone is dropped when unknown).
- replies posted in WhatsApp groups end with the footer `_transcrito por github.com/ustoppble/laschuk-resume_` (DM transcripts forwarded to the Áudios group carry no footer).

Shipping under another name is forbidden and terminates the license grant.

Public repo: https://github.com/ustoppble/laschuk-resume

## Requirements

- Python 3.11+ (with `start.sh`) **or** Docker + Docker Compose
- An Evolution API v2 instance connected to the WhatsApp number in the target group
- `ANTHROPIC_API_KEY` (or `CLAUDE_API_KEY`) for summaries, or `OPENAI_API_KEY` + `OPENAI_SUMMARY_MODEL`
- `OPENAI_API_KEY` for Whisper, or local faster-whisper

## Summary model

`SUMMARIZER=auto` uses Claude when `ANTHROPIC_API_KEY` (or `CLAUDE_API_KEY`) is set, otherwise OpenAI. The choice is made once at startup; there is no per-request fallback between the two.

| Variable | Default | Notes |
| --- | --- | --- |
| `ANTHROPIC_MODEL` | `claude-sonnet-5-5` | any Claude API model id, e.g. `claude-haiku-5-5` (cheaper) or `claude-opus-5-5` (stronger) |
| `OPENAI_SUMMARY_MODEL` | *(empty)* | required only when summarizing with OpenAI |

Current Claude models think before answering by default, and that thinking counts toward `max_tokens`, so the request allows 4096 tokens even though the summary is 2–4 lines. Do not set `temperature`/`top_p`: these models reject them.

## Setup

```bash
cp .env.example .env
$EDITOR .env           # fill EVOLUTION_*, keys, GROUP_NAME or GROUP_JID, WEBHOOK_TOKEN
./start.sh            # or: docker compose up -d --build
./stop.sh
tail -f logs/laschuk-resume.log
curl -s localhost:8787/health
```

Register the Evolution webhook at `POST $EVOLUTION_URL/webhook/set/$EVOLUTION_INSTANCE` pointing to `http://<bot-host>:8787/webhook` with `MESSAGES_UPSERT` and preferably `base64: true`. Example:

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

## Running next to the Evolution API

If Evolution runs in its own Docker Compose project, the bot can join its network so Evolution reaches the webhook at `http://laschuk-resume:8787/webhook`. Copy the example override and set your Evolution network name (`docker network ls`):

```bash
cp docker-compose.override.example.yml docker-compose.override.yml
$EDITOR docker-compose.override.yml   # networks.evolution.name = your Evolution network
docker compose up -d --build          # the override file is picked up automatically
```

The override also binds the host port on loopback only. `docker-compose.override.yml` and `docker-compose.*.local.yml` are git-ignored, so each deployment keeps its real file out of the repository.

## Tests

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

## Prompt injection hardening

The audio transcript and WhatsApp speaker name are **untrusted input**, on every route that
summarizes (group reply, DM transcript, self chat and the Áudios group). The summarizer
sends them as JSON data, separately from its system instructions, and explicitly asks
the model not to follow commands embedded in the audio (including fake role markers,
requests for secrets, and instructions to send data elsewhere).

This is defense in depth, **not a guarantee** that an LLM will never produce a
manipulated summary. The model is not given execution tools, credentials, filesystem
access, or a free-form HTTP client. Keep it that way; never treat transcript content
as permission to run actions. `clean_summary` still enforces the output length/line
limits, but does not semantically verify factual accuracy or adversarial compliance.

Run the boundary regression tests with `python -m pytest tests/test_prompt_injection.py`.
They inspect API request structure using mocked responses; they are **not** proof of
resistance against every prompt injection. Real-model adversarial evaluations are
recommended before sensitive deployments.

