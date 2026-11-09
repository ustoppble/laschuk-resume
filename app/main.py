"""FastAPI webhook receiver + audio → transcript → summary → quoted reply pipeline."""
from __future__ import annotations

import asyncio
import base64
import binascii
import hmac
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request

from .config import Settings, load_settings
from .dedupe import SeenStore
from .evolution import EvolutionClient
from .parse import ROUTE_DM, ROUTE_GROUP, AudioMessage, evaluate_payload, find_group_jid, iter_messages
from .summarize import Summarizer, build_summarizer
from .transcribe import Transcriber, build_transcriber

log = logging.getLogger("laschuk-resume")

GROUP_RETRY_SECONDS = 60
# Brazil has had no DST since 2019; a fixed offset avoids depending on tzdata in the slim image.
BRT = timezone(timedelta(hours=-3), "BRT")


# Footer for group replies only (DM transcripts sent to the Áudios group carry none).
GROUP_FOOTER = "_transcrito por github.com/ustoppble/laschuk-resume_"


def setup_logging(settings: Settings) -> None:
    Path(settings.log_file).parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(settings.log_level)
    for h in list(root.handlers):
        if getattr(h, "_laschuk_resume", False):
            root.removeHandler(h)
    file_handler = RotatingFileHandler(settings.log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    stream_handler = logging.StreamHandler()
    for h in (file_handler, stream_handler):
        h.setFormatter(fmt)
        h._laschuk_resume = True  # type: ignore[attr-defined]
        root.addHandler(h)
    logging.getLogger("httpx").setLevel(logging.WARNING)


class Bot:
    def __init__(
        self,
        settings: Settings,
        evolution: EvolutionClient,
        transcriber: Transcriber,
        summarizer: Summarizer,
        seen: SeenStore,
    ):
        self.s = settings
        self.evolution = evolution
        self.transcriber = transcriber
        self.summarizer = summarizer
        self.seen = seen
        self.group_jid = settings.group_jid
        self._group_lock = asyncio.Lock()
        self._last_group_attempt = 0.0
        self.owner_jid = settings.owner_jid
        self._owner_lock = asyncio.Lock()
        self._last_owner_attempt = 0.0
        self._sem = asyncio.Semaphore(settings.max_concurrency)
        self._tasks: set[asyncio.Task] = set()

    async def ensure_group(self) -> str:
        """Resolve the group JID by name once; retry at most every GROUP_RETRY_SECONDS on failure."""
        if self.group_jid:
            return self.group_jid
        async with self._group_lock:
            if self.group_jid or time.monotonic() - self._last_group_attempt < GROUP_RETRY_SECONDS:
                return self.group_jid
            self._last_group_attempt = time.monotonic()
            try:
                groups = await self.evolution.fetch_groups()
                self.group_jid = find_group_jid(groups, self.s.group_name)
                log.info("resolved group %r -> %s", self.s.group_name, self.group_jid)
            except Exception as exc:
                log.error("could not resolve group %r: %s (set GROUP_JID to skip lookup)", self.s.group_name, exc)
        return self.group_jid

    async def ensure_owner(self) -> str:
        """Resolve the instance's own JID (ownerJid); DMs are only processed once it is known."""
        if self.owner_jid or not self.s.dm_transcribe:
            return self.owner_jid
        async with self._owner_lock:
            if self.owner_jid or time.monotonic() - self._last_owner_attempt < GROUP_RETRY_SECONDS:
                return self.owner_jid
            self._last_owner_attempt = time.monotonic()
            try:
                self.owner_jid = await self.evolution.fetch_owner_jid()
                log.info("resolved owner chat for DM transcripts")
            except Exception as exc:
                log.error("could not resolve ownerJid: %s (DM audios are skipped; set OWNER_JID)", exc)
        return self.owner_jid

    async def handle_payload(self, payload: dict[str, Any]) -> dict[str, int]:
        instance = payload.get("instance")
        if instance and instance != self.s.evolution_instance:
            log.debug("ignoring webhook for instance %r", instance)
            return {"accepted": 0, "ignored": 1}

        group_jid = await self.ensure_group()
        # DMs are only accepted when we know where to send them (the owner's own chat).
        # The owner lookup only runs when a non-group message is present, so group traffic never waits on it.
        allow_dm = False
        if self.s.dm_transcribe and any(
            not str((m.get("key") or {}).get("remoteJid") or "").endswith("@g.us") for m in iter_messages(payload)
        ):
            try:
                allow_dm = bool(self.s.dm_transcript_target_jid) or bool(
                    await asyncio.wait_for(self.ensure_owner(), timeout=15))
            except asyncio.TimeoutError:
                log.error("ownerJid lookup timed out; DM audios in this webhook skipped")
        decisions = evaluate_payload(
            payload, group_jid, max_age_seconds=self.s.max_message_age_seconds, allow_dm=allow_dm
        )
        accepted = 0
        for d in decisions:
            if not d.accepted or d.audio is None:
                log.debug("skip %s: %s", d.message_id or "-", d.reason)
                continue
            if not self.seen.claim(d.message_id):
                log.info("duplicate %s ignored", d.message_id)
                continue
            accepted += 1
            log.info("queued %s audio %s from %s (%ss)", d.audio.route, d.message_id,
                     d.audio.push_name or d.audio.participant or "-", d.audio.seconds)
            task = asyncio.create_task(self._process_guarded(d.audio))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        return {"accepted": accepted, "ignored": len(decisions) - accepted}

    async def _process_guarded(self, audio: AudioMessage) -> None:
        async with self._sem:
            started = time.monotonic()
            try:
                await self.process_audio(audio)
                log.info("done %s in %.1fs", audio.message_id, time.monotonic() - started)
            except Exception:
                log.exception("failed processing audio %s", audio.message_id)

    async def process_audio(self, audio: AudioMessage) -> str | None:
        if self.s.max_audio_seconds and audio.seconds > self.s.max_audio_seconds:
            log.warning("audio %s is %ss (> MAX_AUDIO_SECONDS=%s); skipped",
                        audio.message_id, audio.seconds, self.s.max_audio_seconds)
            return None

        b64, mimetype = audio.inline_base64, audio.mimetype
        if not b64:
            b64, fetched_mime = await self.evolution.get_media_base64(audio.key)
            mimetype = fetched_mime or mimetype
        try:
            raw = base64.b64decode(b64.split(",", 1)[-1], validate=False)
        except (binascii.Error, ValueError) as exc:
            raise RuntimeError(f"invalid base64 media for {audio.message_id}") from exc
        if not raw:
            raise RuntimeError(f"empty media for {audio.message_id}")

        transcript = await self.transcriber.transcribe(raw, mimetype)
        log.info("transcribed %s via %s (%d chars)", audio.message_id, self.transcriber.name, len(transcript))
        if not transcript.strip():
            log.warning("empty transcript for %s; no reply sent", audio.message_id)
            return None

        if audio.route == ROUTE_DM:
            return await self._deliver_dm(audio, transcript)
        if audio.route != ROUTE_GROUP:
            log.error("unknown route %r for %s; nothing sent", audio.route, audio.message_id)
            return None
        return await self._reply_group(audio, transcript)

    async def _deliver_dm(self, audio: AudioMessage, transcript: str) -> str | None:
        """1x1 audio: full transcript + short summary to the OWNER's chat only. Never the sender."""
        target = self.s.dm_transcript_target_jid or self.owner_jid
        label = "target group" if self.s.dm_transcript_target_jid else "owner chat"
        if not target or target in {audio.remote_jid, audio.contact_jid}:
            log.error("dm %s: %s unknown or equals source; nothing sent", audio.message_id, label)
            return None
        try:
            summary = await self.summarizer.summarize(transcript, audio.push_name)
        except Exception as exc:
            log.warning("dm %s: summary failed (%s); sending transcript only", audio.message_id, exc)
            summary = ""
        name = audio.push_name or audio.contact_jid.split("@")[0] or "contato"
        ts = datetime.fromtimestamp(audio.timestamp, BRT) if audio.timestamp else datetime.now(BRT)
        text = (
            f"🎙️ {name} ({ts:%H:%M})\n{transcript.strip()}\n\n"
            f"Resumo: {summary or '(indisponível)'}"
        )
        await self.evolution.send_text(target, text)
        log.info("dm %s transcript delivered to %s", audio.message_id, label)
        return text

    async def _reply_group(self, audio: AudioMessage, transcript: str) -> str | None:
        summary = await self.summarizer.summarize(transcript, audio.push_name)
        if not summary:
            log.warning("empty summary for %s; no reply sent", audio.message_id)
            return None
        log.info("summary %s via %s: %r", audio.message_id, self.summarizer.name, summary)

        body = f"{self.s.reply_prefix}\n{summary}" if self.s.reply_prefix else summary
        text = f"{body}\n\n{GROUP_FOOTER}"
        await self.evolution.send_text_reply(audio.remote_jid, text, audio.key, audio.message)
        log.info("replied to %s in %s", audio.message_id, audio.remote_jid)
        return text

    async def drain(self, timeout: float = 30) -> None:
        if self._tasks:
            await asyncio.wait(self._tasks, timeout=timeout)


def create_app(settings: Settings | None = None, bot: Bot | None = None) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        http_client = None
        seen = None
        if bot is None:
            settings.validate()
            setup_logging(settings)
            http_client = httpx.AsyncClient(timeout=settings.http_timeout_seconds)
            seen = SeenStore(settings.dedupe_db)
            app.state.bot = Bot(
                settings,
                EvolutionClient(settings, http_client),
                build_transcriber(settings, http_client),
                build_summarizer(settings, http_client),
                seen,
            )
            log.info("LaschukResume starting (instance=%s, group=%s, transcriber=%s, summarizer=%s)",
                     settings.evolution_instance, settings.group_jid or repr(settings.group_name),
                     app.state.bot.transcriber.name, app.state.bot.summarizer.name)
            await app.state.bot.ensure_group()
            await app.state.bot.ensure_owner()
            if settings.dm_transcribe and settings.dm_transcript_target_jid:
                log.info("DM transcripts: on (target=%s)", settings.dm_transcript_target_jid)
            else:
                log.info("DM transcripts: %s", "on (owner chat only)" if app.state.bot.owner_jid else
                         ("off" if not settings.dm_transcribe else "pending ownerJid"))
        else:
            app.state.bot = bot
        try:
            yield
        finally:
            await app.state.bot.drain()
            if http_client is not None:
                await http_client.aclose()
            if seen is not None:
                seen.close()
            log.info("LaschukResume stopped")

    app = FastAPI(title="LaschukResume", lifespan=lifespan)

    def _check_token(request: Request) -> None:
        if not settings.webhook_token:
            return
        given = request.query_params.get("token") or request.headers.get("x-webhook-token") or ""
        if not hmac.compare_digest(given, settings.webhook_token):
            raise HTTPException(status_code=401, detail="invalid token")

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        b: Bot = request.app.state.bot
        return {"ok": True, "group_jid": b.group_jid or None, "dm_owner_resolved": bool(b.owner_jid), "dm_target_set": bool(b.s.dm_transcript_target_jid),
                "in_flight": len(b._tasks)}

    # Evolution posts to the base URL, or to /<event-name> when "webhook by events" is enabled.
    @app.post("/webhook")
    @app.post("/webhook/{event_path:path}")
    async def webhook(request: Request, event_path: str = "") -> dict[str, Any]:
        _check_token(request)
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid JSON")
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="expected a JSON object")
        if not payload.get("event") and event_path:
            payload["event"] = event_path
        result = await request.app.state.bot.handle_payload(payload)
        return {"ok": True, **result}

    return app


def app_factory() -> FastAPI:
    """Entry point for `uvicorn --factory app.main:app_factory`."""
    return create_app()
