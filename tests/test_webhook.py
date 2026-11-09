"""Webhook + pipeline tests with fake Evolution/transcriber/summarizer (no network)."""
import asyncio
import base64
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.dedupe import SeenStore
from app.main import Bot, create_app

FIXTURE = Path(__file__).parent / "fixtures" / "messages_upsert_audio.json"
GROUP = "120363000000000001@g.us"


class FakeEvolution:
    def __init__(self, groups=None):
        self.groups = groups or []
        self.sent = []
        self.media_calls = []
        self.reactions = []
        self.events = []  # ordered ("react", emoji) / ("reply", jid) to check sequencing

    async def fetch_groups(self):
        return self.groups

    async def get_media_base64(self, key):
        self.media_calls.append(key)
        return base64.b64encode(b"OggS-fake-audio").decode(), "audio/ogg; codecs=opus"

    async def send_text_reply(self, jid, text, quoted_key, quoted_message):
        self.sent.append({"jid": jid, "text": text, "key": quoted_key, "message": quoted_message})
        self.events.append(("reply", jid))
        return {"key": {"id": "REPLY"}}

    async def send_reaction(self, remote_jid, key, emoji):
        self.reactions.append({"jid": remote_jid, "key": key, "emoji": emoji})
        self.events.append(("react", emoji))
        return {"key": {"id": "REACTION"}}


class FakeTranscriber:
    name = "fake"

    def __init__(self, text="Oi pessoal, a reunião do condomínio ficou para sexta às 19h."):
        self.text = text
        self.calls = []

    async def transcribe(self, audio, mimetype):
        self.calls.append((audio, mimetype))
        return self.text


class FakeSummarizer:
    name = "fake"

    async def summarize(self, transcript, speaker=""):
        return "Reunião do condomínio na sexta às 19h.\nMaria avisou o grupo."


def make_settings(**kw):
    base = dict(
        evolution_url="http://evolution.invalid",
        evolution_instance="test-instance",
        evolution_api_key="x",
        group_jid=GROUP,
        anthropic_api_key="x",
        max_message_age_seconds=0,
    )
    base.update(kw)
    return Settings(**base)


@pytest.fixture
def payload():
    return json.loads(FIXTURE.read_text())


def build(settings=None, groups=None, transcriber=None):
    settings = settings or make_settings()
    evo = FakeEvolution(groups)
    tr = transcriber or FakeTranscriber()
    bot = Bot(settings, evo, tr, FakeSummarizer(), SeenStore(":memory:"))
    return settings, bot, evo, tr


def post(client, payload, path="/webhook", **kw):
    return client.post(path, json=payload, **kw)


def test_audio_gets_quoted_summary_reply(payload):
    settings, bot, evo, tr = build()
    with TestClient(create_app(settings, bot)) as client:
        r = post(client, payload)
        assert r.status_code == 200 and r.json()["accepted"] == 1
    # TestClient shutdown drains in-flight tasks
    assert len(evo.sent) == 1
    sent = evo.sent[0]
    assert sent["jid"] == GROUP
    assert sent["key"]["id"] == "3EB0A1B2C3D4E5F6A7B8"
    assert sent["text"].startswith(settings.reply_prefix)
    assert sent["text"].endswith("\n\n_transcrito por github.com/ustoppble/laschuk-resume_")
    assert sent["text"].count("github.com") == 1
    assert not hasattr(settings, "reply_footer")
    assert "sexta às 19h" in sent["text"]
    assert evo.media_calls and tr.calls[0][0] == b"OggS-fake-audio"


def test_duplicate_webhook_replies_once(payload):
    settings, bot, evo, _ = build()
    with TestClient(create_app(settings, bot)) as client:
        assert post(client, payload).json()["accepted"] == 1
        assert post(client, payload).json()["accepted"] == 0
    assert len(evo.sent) == 1


def test_foreign_group_messages_not_replied(payload):
    settings, bot, evo, _ = build()
    other = json.loads(json.dumps(payload))
    other["data"]["key"]["remoteJid"] = "999@g.us"
    other["data"]["key"]["id"] = "OTHER"
    with TestClient(create_app(settings, bot)) as client:
        assert post(client, other).json()["accepted"] == 0
    assert evo.sent == []


def test_other_instance_ignored(payload):
    settings, bot, evo, _ = build()
    payload["instance"] = "someone-else"
    with TestClient(create_app(settings, bot)) as client:
        assert post(client, payload).json()["accepted"] == 0
    assert evo.sent == []


def test_group_resolved_by_name(payload):
    settings, bot, evo, _ = build(
        make_settings(group_jid=""), groups=[{"id": GROUP, "subject": "Hillside"}]
    )
    with TestClient(create_app(settings, bot)) as client:
        assert post(client, payload).json()["accepted"] == 1
        assert client.get("/health").json()["group_jid"] == GROUP
    assert len(evo.sent) == 1


def test_inline_base64_skips_download(payload):
    payload["data"]["message"]["base64"] = base64.b64encode(b"inline-audio").decode()
    settings, bot, evo, tr = build()
    with TestClient(create_app(settings, bot)) as client:
        post(client, payload)
    assert evo.media_calls == []
    assert tr.calls[0][0] == b"inline-audio"
    assert len(evo.sent) == 1


def test_event_path_variant(payload):
    del payload["event"]
    settings, bot, evo, _ = build()
    with TestClient(create_app(settings, bot)) as client:
        assert post(client, payload, path="/webhook/messages-upsert").json()["accepted"] == 1


def test_token_required_when_configured(payload):
    settings, bot, evo, _ = build(make_settings(webhook_token="s3cret"))
    with TestClient(create_app(settings, bot)) as client:
        assert post(client, payload).status_code == 401
        assert post(client, payload, path="/webhook?token=wrong").status_code == 401
        assert post(client, payload, path="/webhook?token=s3cret").status_code == 200


def test_invalid_json_rejected():
    settings, bot, _, _ = build()
    with TestClient(create_app(settings, bot)) as client:
        r = client.post("/webhook", content=b"not json", headers={"content-type": "application/json"})
        assert r.status_code == 400


def test_pipeline_error_is_logged_not_raised(payload, caplog):
    class Boom(FakeTranscriber):
        async def transcribe(self, audio, mimetype):
            raise RuntimeError("whisper down")

    settings, bot, evo, _ = build(transcriber=Boom())
    with TestClient(create_app(settings, bot)) as client:
        assert post(client, payload).status_code == 200
    assert evo.sent == []
    assert any("failed processing audio" in r.message for r in caplog.records)


def test_empty_transcript_sends_nothing(payload):
    settings, bot, evo, _ = build(transcriber=FakeTranscriber(text="  "))
    with TestClient(create_app(settings, bot)) as client:
        post(client, payload)
    assert evo.sent == []


def test_too_long_audio_skipped(payload):
    payload["data"]["message"]["audioMessage"]["seconds"] = 5000
    settings, bot, evo, tr = build(make_settings(max_audio_seconds=600))
    with TestClient(create_app(settings, bot)) as client:
        post(client, payload)
    assert evo.sent == [] and tr.calls == []


def test_seen_store_persists(tmp_path):
    db = str(tmp_path / "seen.sqlite3")
    s1 = SeenStore(db)
    assert s1.claim("A") is True
    assert s1.claim("A") is False
    s1.close()
    s2 = SeenStore(db)
    assert s2.claim("A") is False
    assert s2.claim("B") is True
    s2.close()


def test_settings_validation_lists_missing():
    from app.config import ConfigError, load_settings
    import os

    keys = ["EVOLUTION_URL", "EVOLUTION_INSTANCE", "EVOLUTION_API_KEY", "ANTHROPIC_API_KEY",
            "CLAUDE_API_KEY", "OPENAI_API_KEY", "OPENAI_SUMMARY_MODEL"]
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        s = load_settings(env_file="/nonexistent/.env")
        with pytest.raises(ConfigError) as exc:
            s.validate()
        msg = str(exc.value)
        assert "EVOLUTION_URL is not set" in msg and "no summarizer configured" in msg
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


# --- status reactions on the original group audio (⏳ → ✅, ❌ on failure) ---


def test_group_audio_reactions_wrap_the_reply(payload):
    settings, bot, evo, _ = build()
    with TestClient(create_app(settings, bot)) as client:
        assert post(client, payload).json()["accepted"] == 1
    assert evo.events == [("react", "⏳"), ("reply", GROUP), ("react", "✅")]
    # the reaction targets the original audio, participant included (required in groups)
    assert all(r["jid"] == GROUP for r in evo.reactions)
    assert evo.reactions[0]["key"] == {
        "remoteJid": GROUP, "fromMe": False, "id": "3EB0A1B2C3D4E5F6A7B8",
        "participant": "5511999990000@s.whatsapp.net",
    }


def test_hourglass_comes_before_download(payload):
    settings, bot, evo, _ = build()
    order = []
    original = evo.get_media_base64

    async def media(key):
        order.append([e[1] for e in evo.events])
        return await original(key)

    evo.get_media_base64 = media
    with TestClient(create_app(settings, bot)) as client:
        post(client, payload)
    assert order == [["⏳"]]


def test_transcription_failure_reacts_cross(payload):
    class Boom(FakeTranscriber):
        async def transcribe(self, audio, mimetype):
            raise RuntimeError("whisper down")

    settings, bot, evo, _ = build(transcriber=Boom())
    with TestClient(create_app(settings, bot)) as client:
        post(client, payload)
    assert evo.events == [("react", "⏳"), ("react", "❌")]


def test_download_failure_reacts_cross(payload):
    settings, bot, evo, _ = build()

    async def broken(key):
        raise RuntimeError("media gone")

    evo.get_media_base64 = broken
    with TestClient(create_app(settings, bot)) as client:
        post(client, payload)
    assert evo.events == [("react", "⏳"), ("react", "❌")]


def test_empty_summary_reacts_cross(payload):
    class Empty(FakeSummarizer):
        async def summarize(self, transcript, speaker=""):
            return ""

    settings = make_settings()
    evo = FakeEvolution()
    bot = Bot(settings, evo, FakeTranscriber(), Empty(), SeenStore(":memory:"))
    with TestClient(create_app(settings, bot)) as client:
        post(client, payload)
    assert evo.events == [("react", "⏳"), ("react", "❌")]


def test_empty_transcript_reacts_cross(payload):
    settings, bot, evo, _ = build(transcriber=FakeTranscriber(text="  "))
    with TestClient(create_app(settings, bot)) as client:
        post(client, payload)
    assert evo.events == [("react", "⏳"), ("react", "❌")]


def test_send_failure_reacts_cross(payload):
    settings, bot, evo, _ = build()

    async def broken(jid, text, quoted_key, quoted_message):
        raise RuntimeError("send failed")

    evo.send_text_reply = broken
    with TestClient(create_app(settings, bot)) as client:
        post(client, payload)
    assert evo.events == [("react", "⏳"), ("react", "❌")]


def test_reaction_failure_never_blocks_reply(payload, caplog):
    settings, bot, evo, _ = build()

    async def broken(remote_jid, key, emoji):
        raise RuntimeError("reaction endpoint down")

    evo.send_reaction = broken
    with TestClient(create_app(settings, bot)) as client:
        post(client, payload)
    assert len(evo.sent) == 1
    assert any("reaction" in r.message and r.levelname == "WARNING" for r in caplog.records)
    assert not any("failed processing audio" in r.message for r in caplog.records)


def test_group_reactions_can_be_disabled(payload):
    settings, bot, evo, _ = build(make_settings(group_reactions=False))
    with TestClient(create_app(settings, bot)) as client:
        post(client, payload)
    assert len(evo.sent) == 1 and evo.reactions == []


def test_too_long_audio_gets_no_reaction(payload):
    payload["data"]["message"]["audioMessage"]["seconds"] = 5000
    settings, bot, evo, _ = build(make_settings(max_audio_seconds=600))
    with TestClient(create_app(settings, bot)) as client:
        post(client, payload)
    assert evo.reactions == []


@pytest.mark.parametrize("raw, expected", [(None, True), ("1", True), ("0", False), ("off", False)])
def test_group_reactions_env(monkeypatch, raw, expected):
    from app.config import load_settings

    if raw is None:
        monkeypatch.delenv("GROUP_REACTIONS", raising=False)
    else:
        monkeypatch.setenv("GROUP_REACTIONS", raw)
    assert load_settings(env_file="/nonexistent/.env").group_reactions is expected
