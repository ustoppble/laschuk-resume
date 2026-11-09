"""Pure parsing/filtering of Evolution API v2 webhook payloads (no I/O)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

UPSERT_EVENTS = {"messages.upsert", "messages_upsert"}

ROUTE_GROUP = "group"
ROUTE_DM = "dm"
ROUTE_SELF = "self"  # the owner's own chat ("Você"): audio they send to themselves
ROUTE_TRANSCRIPTS = "transcripts"  # audio sent/forwarded inside the Áudios group (DM_TRANSCRIPT_TARGET_JID)
USER_SUFFIX = "@s.whatsapp.net"

# Wrappers WhatsApp uses around the real message content.
_WRAPPERS = (
    "ephemeralMessage",
    "viewOnceMessage",
    "viewOnceMessageV2",
    "viewOnceMessageV2Extension",
    "documentWithCaptionMessage",
)


@dataclass(frozen=True)
class AudioMessage:
    message_id: str
    remote_jid: str
    participant: str
    push_name: str
    key: dict[str, Any]
    message: dict[str, Any]
    mimetype: str
    seconds: int
    ptt: bool
    timestamp: int
    inline_base64: str  # filled when Evolution has WEBHOOK_BASE64 enabled
    # Explicit routing: "group" = quoted reply in the same group (original behaviour);
    # "dm" = 1x1 chat, transcript goes ONLY to the owner's own chat, never back to the sender;
    # "self" = the owner's own chat, always answered quoted in that same chat;
    # "transcripts" = the Áudios group, answered quoted in place with the full transcript (no footer).
    route: str = ROUTE_GROUP
    contact_jid: str = ""  # dm only: sender as <number>@s.whatsapp.net


@dataclass(frozen=True)
class Decision:
    """Outcome of filtering a single message: `audio` is set only when accepted."""

    accepted: bool
    reason: str
    message_id: str = ""
    audio: AudioMessage | None = None


def normalize_event(name: Any) -> str:
    return str(name or "").strip().lower().replace("-", ".").replace("_", ".")


def is_upsert_event(payload: dict[str, Any]) -> bool:
    return normalize_event(payload.get("event")) in {normalize_event(e) for e in UPSERT_EVENTS}


def iter_messages(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Evolution sends `data` as one message object; some setups send a list or {messages: [...]}."""
    data = payload.get("data")
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and isinstance(data.get("messages"), list):
        items = data["messages"]
    elif isinstance(data, dict):
        items = [data]
    else:
        items = []
    return (m for m in items if isinstance(m, dict))


def unwrap_message(message: dict[str, Any] | None) -> dict[str, Any]:
    msg = message or {}
    for _ in range(5):  # wrappers can nest; bound the loop
        for wrapper in _WRAPPERS:
            inner = msg.get(wrapper)
            if isinstance(inner, dict) and isinstance(inner.get("message"), dict):
                msg = inner["message"]
                break
        else:
            return msg
    return msg


def _int(value: Any, default: int = 0) -> int:
    # Baileys sometimes serialises longs as {"low": n, "high": 0, "unsigned": bool}
    if isinstance(value, dict) and "low" in value:
        value = value.get("low")
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def dm_contact_jid(key: dict[str, Any]) -> str:
    """Return the sender's <number>@s.whatsapp.net for a 1x1 chat, or "" if not a 1x1 chat.

    WhatsApp now often addresses 1x1 chats by LID (`...@lid`) and puts the phone JID in
    `remoteJidAlt`; both forms count as 1x1. Groups, status@broadcast, newsletters → "".
    """
    remote_jid = str(key.get("remoteJid") or "")
    if remote_jid.endswith(USER_SUFFIX):
        return remote_jid
    if remote_jid.endswith("@lid"):
        # The phone JID travels in remoteJidAlt (or senderPn on some Evolution versions).
        for field in ("remoteJidAlt", "senderPn"):
            alt = str(key.get(field) or "")
            if alt.endswith(USER_SUFFIX):
                return alt
    return ""


def format_phone(jid: str) -> str:
    """E.164 with all digits together for a <number>@s.whatsapp.net JID: "+5511912345678"; "" when no usable number.

    No spaces or hyphens, so WhatsApp renders the whole number as one tappable link.
    """
    digits, _, server = normalize_jid(jid).partition("@")
    if server != USER_SUFFIX[1:] or not digits.isdigit() or not 8 <= len(digits) <= 15:
        return ""
    return f"+{digits}"


def normalize_jid(jid: Any) -> str:
    """Drop the multi-device suffix (`<number>:<device>@s.whatsapp.net` → `<number>@s.whatsapp.net`)."""
    user, sep, server = str(jid or "").strip().partition("@")
    return f"{user.split(':', 1)[0]}{sep}{server}"


def evaluate_message(
    item: dict[str, Any],
    group_jid: str,
    *,
    now: float | None = None,
    max_age_seconds: int = 0,
    allow_dm: bool = False,
    owner_jid: str = "",
    transcript_jid: str = "",
) -> Decision:
    key = item.get("key") or {}
    message_id = str(key.get("id") or "")
    if not message_id:
        return Decision(False, "no message id")
    # fromMe allowed: the bot runs on the owner's own WhatsApp; it only replies with text, so no loop.

    remote_jid = str(key.get("remoteJid") or "")
    route, contact_jid = ROUTE_GROUP, ""
    if remote_jid.endswith("@g.us"):
        if transcript_jid and remote_jid == transcript_jid:
            route = ROUTE_TRANSCRIPTS  # the Áudios group is accepted without being listed in GROUP_JID
        elif not group_jid or remote_jid not in {j.strip() for j in group_jid.split(",") if j.strip()}:
            return Decision(False, "other group", message_id)
    else:
        if remote_jid == "status@broadcast" or remote_jid.endswith("@broadcast"):
            return Decision(False, "status/broadcast", message_id)
        contact_jid = dm_contact_jid(key) if allow_dm else ""
        if not contact_jid:
            return Decision(False, "not a group message", message_id)
        from_me = key.get("fromMe") is True or str(key.get("fromMe")).lower() == "true"
        if owner_jid and normalize_jid(contact_jid) == normalize_jid(owner_jid):
            route = ROUTE_SELF  # chat with oneself: fromMe is expected here and is accepted
        elif from_me:
            return Decision(False, "own dm message", message_id)  # owner → someone else: ignored
        else:
            route = ROUTE_DM

    raw_message = item.get("message") or {}
    content = unwrap_message(raw_message)
    audio = content.get("audioMessage")
    if not isinstance(audio, dict):
        return Decision(False, f"not audio ({item.get('messageType') or 'unknown'})", message_id)

    timestamp = _int(item.get("messageTimestamp"))
    if max_age_seconds and timestamp:
        age = (now if now is not None else time.time()) - timestamp
        if age > max_age_seconds:
            return Decision(False, f"too old ({int(age)}s)", message_id)

    inline = raw_message.get("base64") or content.get("base64") or ""
    return Decision(
        True,
        "audio",
        message_id,
        AudioMessage(
            message_id=message_id,
            remote_jid=remote_jid,
            participant=str(key.get("participant") or item.get("participant") or ""),
            push_name=str(item.get("pushName") or ""),
            key={k: key[k] for k in ("id", "remoteJid", "fromMe", "participant") if k in key},
            message=raw_message,
            mimetype=str(audio.get("mimetype") or "audio/ogg"),
            seconds=_int(audio.get("seconds")),
            ptt=bool(audio.get("ptt")),
            timestamp=timestamp,
            inline_base64=inline if isinstance(inline, str) else "",
            route=route,
            contact_jid=contact_jid,
        ),
    )


def evaluate_payload(
    payload: dict[str, Any],
    group_jid: str,
    *,
    now: float | None = None,
    max_age_seconds: int = 0,
    allow_dm: bool = False,
    owner_jid: str = "",
    transcript_jid: str = "",
) -> list[Decision]:
    if not isinstance(payload, dict) or not is_upsert_event(payload):
        return []
    return [
        evaluate_message(m, group_jid, now=now, max_age_seconds=max_age_seconds, allow_dm=allow_dm,
                         owner_jid=owner_jid, transcript_jid=transcript_jid)
        for m in iter_messages(payload)
    ]


def find_group_jid(groups: Any, name: str) -> str:
    """Pick the JID of the group whose subject equals `name` (case/space-insensitive).

    Raises LookupError when zero or several groups match, so the operator sets GROUP_JID.
    """
    if isinstance(groups, dict):
        groups = groups.get("groups") or groups.get("data") or []
    wanted = " ".join(name.split()).casefold()
    matches = sorted(
        {
            str(g.get("id"))
            for g in groups or []
            if isinstance(g, dict)
            and g.get("id")
            and " ".join(str(g.get("subject") or "").split()).casefold() == wanted
        }
    )
    if not matches:
        raise LookupError(f"no WhatsApp group named {name!r} visible to this instance")
    if len(matches) > 1:
        raise LookupError(f"several groups named {name!r}: {', '.join(matches)}; set GROUP_JID")
    return matches[0]


def mime_extension(mimetype: str) -> str:
    base = mimetype.split(";")[0].strip().lower()
    return {
        "audio/ogg": "ogg",
        "audio/opus": "ogg",
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/mp4": "m4a",
        "audio/m4a": "m4a",
        "audio/x-m4a": "m4a",
        "audio/aac": "m4a",
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/webm": "webm",
        "audio/amr": "amr",
    }.get(base, "ogg")


def clean_summary(text: str, max_lines: int = 4, max_chars: int = 600) -> str:
    """Normalise LLM output into at most `max_lines` non-empty short lines."""
    lines = []
    for raw in (text or "").replace("\r", "").split("\n"):
        line = raw.strip().strip("`").strip()
        if not line:
            continue
        if line.lower().startswith(("resumo:", "**resumo")) and len(line) < 20:
            continue
        lines.append(line)
    out = "\n".join(lines[:max_lines])
    if len(out) > max_chars:
        out = out[: max_chars - 1].rstrip() + "…"
    return out
