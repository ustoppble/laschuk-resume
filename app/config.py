"""Settings loaded from environment (.env). No secret has a default value."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


class ConfigError(RuntimeError):
    pass


def _get(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _int(name: str, default: int) -> int:
    raw = _get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    evolution_url: str
    evolution_instance: str
    evolution_api_key: str
    group_name: str = "Hillside"
    group_jid: str = ""
    webhook_token: str = ""
    # 1x1 audio → transcript sent ONLY to the owner's own chat (ownerJid of the instance).
    dm_transcribe: bool = True
    owner_jid: str = ""  # optional override; otherwise resolved via instance/fetchInstances
    # Where DM transcripts go (e.g. the "Áudios" group). Empty = owner's own chat.
    dm_transcript_target_jid: str = ""

    transcriber: str = "auto"  # auto | openai | local
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_transcribe_model: str = "whisper-1"
    openai_summary_model: str = ""
    local_whisper_model: str = "small"
    local_whisper_device: str = "cpu"
    local_whisper_compute_type: str = "int8"
    transcribe_language: str = "pt"

    summarizer: str = "auto"  # auto | anthropic | openai
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-5-5"

    reply_prefix: str = "🎧 *Resumo do áudio:*"
    max_audio_seconds: int = 1200
    max_message_age_seconds: int = 900
    max_concurrency: int = 2
    http_timeout_seconds: int = 120

    log_file: str = str(ROOT / "logs" / "laschuk-resume.log")
    log_level: str = "INFO"
    dedupe_db: str = str(ROOT / "data" / "seen.sqlite3")

    missing: tuple[str, ...] = field(default=())

    @property
    def evolution_base(self) -> str:
        return self.evolution_url.rstrip("/")

    def validate(self) -> None:
        """Raise ConfigError with every missing/invalid setting at once."""
        problems = list(self.missing)
        if self.transcriber not in {"auto", "openai", "local"}:
            problems.append("TRANSCRIBER must be auto, openai or local")
        if self.transcriber == "openai" and not self.openai_api_key:
            problems.append("TRANSCRIBER=openai requires OPENAI_API_KEY")
        if self.summarizer not in {"auto", "anthropic", "openai"}:
            problems.append("SUMMARIZER must be auto, anthropic or openai")
        if self.summarizer == "anthropic" and not self.anthropic_api_key:
            problems.append("SUMMARIZER=anthropic requires ANTHROPIC_API_KEY (or CLAUDE_API_KEY)")
        if self.summarizer == "openai" and not (self.openai_api_key and self.openai_summary_model):
            problems.append("SUMMARIZER=openai requires OPENAI_API_KEY and OPENAI_SUMMARY_MODEL")
        if self.summarizer == "auto" and not self.anthropic_api_key and not (
            self.openai_api_key and self.openai_summary_model
        ):
            problems.append(
                "no summarizer configured: set ANTHROPIC_API_KEY (or CLAUDE_API_KEY), "
                "or OPENAI_API_KEY + OPENAI_SUMMARY_MODEL"
            )
        if problems:
            raise ConfigError("; ".join(problems))


def load_settings(env_file: str | os.PathLike | None = None) -> Settings:
    load_dotenv(env_file or ROOT / ".env", override=False)

    required = {
        "EVOLUTION_URL": _get("EVOLUTION_URL"),
        "EVOLUTION_INSTANCE": _get("EVOLUTION_INSTANCE"),
        "EVOLUTION_API_KEY": _get("EVOLUTION_API_KEY"),
    }
    missing = tuple(f"{k} is not set" for k, v in required.items() if not v)

    defaults = Settings(evolution_url="", evolution_instance="", evolution_api_key="")
    return Settings(
        evolution_url=required["EVOLUTION_URL"],
        evolution_instance=required["EVOLUTION_INSTANCE"],
        evolution_api_key=required["EVOLUTION_API_KEY"],
        group_name=_get("GROUP_NAME", defaults.group_name),
        group_jid=_get("GROUP_JID"),
        webhook_token=_get("WEBHOOK_TOKEN"),
        dm_transcribe=_get("DM_TRANSCRIBE", "1").lower() not in {"0", "false", "no", "off"},
        owner_jid=_get("OWNER_JID"),
        dm_transcript_target_jid=_get("DM_TRANSCRIPT_TARGET_JID"),
        transcriber=_get("TRANSCRIBER", defaults.transcriber).lower(),
        openai_api_key=_get("OPENAI_API_KEY"),
        openai_base_url=_get("OPENAI_BASE_URL", defaults.openai_base_url).rstrip("/"),
        openai_transcribe_model=_get("OPENAI_TRANSCRIBE_MODEL", defaults.openai_transcribe_model),
        openai_summary_model=_get("OPENAI_SUMMARY_MODEL"),
        local_whisper_model=_get("LOCAL_WHISPER_MODEL", defaults.local_whisper_model),
        local_whisper_device=_get("LOCAL_WHISPER_DEVICE", defaults.local_whisper_device),
        local_whisper_compute_type=_get("LOCAL_WHISPER_COMPUTE_TYPE", defaults.local_whisper_compute_type),
        transcribe_language=_get("TRANSCRIBE_LANGUAGE", defaults.transcribe_language),
        summarizer=_get("SUMMARIZER", defaults.summarizer).lower(),
        anthropic_api_key=_get("ANTHROPIC_API_KEY") or _get("CLAUDE_API_KEY"),
        anthropic_model=_get("ANTHROPIC_MODEL", defaults.anthropic_model),
        reply_prefix=_get("REPLY_PREFIX", defaults.reply_prefix),
        max_audio_seconds=_int("MAX_AUDIO_SECONDS", defaults.max_audio_seconds),
        max_message_age_seconds=_int("MAX_MESSAGE_AGE_SECONDS", defaults.max_message_age_seconds),
        max_concurrency=max(1, _int("MAX_CONCURRENCY", defaults.max_concurrency)),
        http_timeout_seconds=_int("HTTP_TIMEOUT_SECONDS", defaults.http_timeout_seconds),
        log_file=_get("LOG_FILE", defaults.log_file),
        log_level=_get("LOG_LEVEL", defaults.log_level).upper(),
        dedupe_db=_get("DEDUPE_DB", defaults.dedupe_db),
        missing=missing,
    )
