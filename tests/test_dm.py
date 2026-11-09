"""1x1 (DM) audio routing: transcript goes ONLY to the owner's own chat; groups unchanged."""
import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.dedupe import SeenStore
from app.main import Bot, create_app
from app.parse import ROUTE_DM, ROUTE_GROUP, evaluate_payload

from .test_webhook import GROUP, FakeEvolution, FakeSummarizer, FakeTranscriber, make_settings

FIXTURE = Path(__file__).parent / "fixtures" / "messages_upsert_audio.json"
OWNER = "555100000000@s.whatsapp.net"
SENDER = "5511999990000@s.whatsapp.net"
SENDER_LID = "123456789012345@lid"


class OwnerEvolution(FakeEvolution):
    def __init__(self, owner=OWNER, **kw):
        super().__init__(**kw)
        self.owner = owner
        self.plain = []
        self.owner_calls = 0

    async def fetch_owner_jid(self):
        self.owner_calls += 1
        if not self.owner:
            raise RuntimeError("no owner")
        return self.owner

    async def send_text(self, jid, text):
        self.plain.append({"jid": jid, "text": text})
        return {"key": {"id": "DM"}}

    def all_targets(self):
        return [m["jid"] for m in self.sent] + [m["jid"] for m in self.plain]


@pytest.fixture
def group_payload():
    return json.loads(FIXTURE.read_text())


def dm_payload(lid=False, from_me=False, msg_id="DMAUDIO1"):
    p = json.loads(FIXTURE.read_text())
    key = {"id": msg_id, "fromMe": from_me}
    if lid:
        key.update(remoteJid=SENDER_LID, remoteJidAlt=SENDER)
    else:
        key["remoteJid"] = SENDER
    p["data"]["key"] = key
    p["data"]["pushName"] = "João"
    p["data"]["messageTimestamp"] = 1794075600  # 2026-11-07 18:20 UTC -> 15:20 BRT
    return p


def build(owner=OWNER, **settings_kw):
    settings = make_settings(**settings_kw)
    evo = OwnerEvolution(owner=owner)
    bot = Bot(settings, evo, FakeTranscriber(), FakeSummarizer(), SeenStore(":memory:"))
    return settings, bot, evo


@pytest.mark.parametrize("lid", [False, True])
def test_dm_audio_goes_only_to_owner(lid):
    settings, bot, evo = build()
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=dm_payload(lid=lid)).json()["accepted"] == 1
    assert evo.sent == []  # no quoted reply anywhere
    assert [m["jid"] for m in evo.plain] == [OWNER]
    assert SENDER not in evo.all_targets() and SENDER_LID not in evo.all_targets()
    text = evo.plain[0]["text"]
    assert text.startswith("🎙️ João (15:20)\n")
    assert FakeTranscriber().text in text  # full transcript
    assert "\n\nResumo: Reunião do condomínio" in text
    assert text.endswith("Maria avisou o grupo.")
    assert "LaschukResume" not in text and "github.com" not in text


def test_dm_never_sent_when_owner_unknown():
    settings, bot, evo = build(owner="")
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=dm_payload()).json()["accepted"] == 0
    assert evo.all_targets() == []


def test_dm_from_me_ignored():
    settings, bot, evo = build()
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=dm_payload(from_me=True)).json()["accepted"] == 0
        assert client.post("/webhook", json=dm_payload(lid=True, from_me=True, msg_id="X2")).json()["accepted"] == 0
    assert evo.all_targets() == []


def test_status_broadcast_ignored():
    p = dm_payload()
    p["data"]["key"]["remoteJid"] = "status@broadcast"
    p["data"]["key"]["participant"] = SENDER
    settings, bot, evo = build()
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=p).json()["accepted"] == 0
    assert evo.all_targets() == []


def test_dm_disabled_flag():
    settings, bot, evo = build(dm_transcribe=False)
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=dm_payload()).json()["accepted"] == 0
    assert evo.all_targets() == [] and evo.owner_calls == 0


def test_group_unchanged_and_never_goes_to_owner(group_payload):
    settings, bot, evo = build()
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=group_payload).json()["accepted"] == 1
    assert evo.plain == []
    assert len(evo.sent) == 1
    sent = evo.sent[0]
    assert sent["jid"] == GROUP and sent["key"]["id"] == "3EB0A1B2C3D4E5F6A7B8"
    assert sent["text"] == (
        f"{settings.reply_prefix}\nReunião do condomínio na sexta às 19h.\nMaria avisou o grupo.\n\n"
        "_transcrito por github.com/ustoppble/laschuk-resume_"
    )
    assert evo.owner_calls == 0  # group traffic never triggers the owner lookup


def test_group_from_me_still_accepted_like_before(group_payload):
    # Current production behaviour in groups (owner's own audios are summarised) is kept.
    group_payload["data"]["key"]["fromMe"] = True
    settings, bot, evo = build()
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=group_payload).json()["accepted"] == 1
    assert [m["jid"] for m in evo.sent] == [GROUP] and evo.plain == []


def test_parse_routes():
    [g] = evaluate_payload(json.loads(FIXTURE.read_text()), GROUP, allow_dm=True)
    assert g.audio.route == ROUTE_GROUP
    [d] = evaluate_payload(dm_payload(lid=True), GROUP, allow_dm=True)
    assert d.audio.route == ROUTE_DM and d.audio.contact_jid == SENDER
    [off] = evaluate_payload(dm_payload(), GROUP)  # default: DMs rejected
    assert not off.accepted and off.reason == "not a group message"
    p = dm_payload()
    p["data"]["key"]["remoteJid"] = "123@newsletter"
    [n] = evaluate_payload(p, GROUP, allow_dm=True)
    assert not n.accepted
    p = copy.deepcopy(dm_payload(lid=True))
    del p["data"]["key"]["remoteJidAlt"]
    [nolid] = evaluate_payload(p, GROUP, allow_dm=True)
    assert not nolid.accepted


AUDIOS = "120363000000000000@g.us"


@pytest.mark.parametrize("lid", [False, True])
def test_dm_goes_to_target_group_not_sender_nor_owner(lid):
    settings, bot, evo = build(dm_transcript_target_jid=AUDIOS)
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=dm_payload(lid=lid)).json()["accepted"] == 1
    assert evo.all_targets() == [AUDIOS]
    assert evo.sent == []  # plain send_text, no quoted reply
    assert "João" in evo.plain[0]["text"]


def test_dm_target_works_even_if_owner_unknown():
    settings, bot, evo = build(owner="", dm_transcript_target_jid=AUDIOS)
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=dm_payload()).json()["accepted"] == 1
    assert evo.all_targets() == [AUDIOS]


def test_audio_from_target_group_not_transcribed(group_payload):
    settings, bot, evo = build(dm_transcript_target_jid=AUDIOS)
    p = copy.deepcopy(group_payload)
    p["data"]["key"]["remoteJid"] = AUDIOS
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=p).json()["accepted"] == 0
    assert evo.all_targets() == []


# --- self-chat: audio the owner sends to their own number ("Você") is transcribed too ---

OTHER = "5511888880000@s.whatsapp.net"


def self_payload(lid=False, from_me=True, msg_id="SELFAUDIO1", remote=OWNER):
    p = dm_payload(from_me=from_me, msg_id=msg_id)
    if lid:
        p["data"]["key"].update(remoteJid="987654321098765@lid", remoteJidAlt=remote)
    else:
        p["data"]["key"]["remoteJid"] = remote
    p["data"]["pushName"] = "Eu"
    return p


@pytest.mark.parametrize("lid", [False, True])
def test_self_chat_audio_accepted_and_quoted_in_same_chat(lid):
    settings, bot, evo = build()
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=self_payload(lid=lid)).json()["accepted"] == 1
    assert evo.plain == []
    [sent] = evo.sent  # quoted reply to the audio, in the owner's own chat
    assert sent["jid"] == OWNER
    assert sent["key"]["id"] == "SELFAUDIO1" and sent["key"]["fromMe"] is True
    assert sent["text"].startswith("🎙️ Você (15:20)\n")
    assert FakeTranscriber().text in sent["text"]
    assert "\n\nResumo: Reunião do condomínio" in sent["text"]


def test_self_chat_audio_goes_to_target_when_set():
    settings, bot, evo = build(dm_transcript_target_jid=AUDIOS)
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=self_payload()).json()["accepted"] == 1
    assert evo.all_targets() == [AUDIOS]
    assert evo.sent == [] and "Você" in evo.plain[0]["text"]


def test_self_chat_matches_owner_with_device_suffix():
    settings, bot, evo = build(owner="555100000000:12@s.whatsapp.net")
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=self_payload()).json()["accepted"] == 1
    assert [m["jid"] for m in evo.sent] == [OWNER]


def test_self_chat_not_processed_when_owner_unknown():
    settings, bot, evo = build(owner="")
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=self_payload()).json()["accepted"] == 0
    assert evo.all_targets() == []


def test_from_me_audio_to_third_party_still_ignored():
    settings, bot, evo = build()
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=self_payload(remote=OTHER)).json()["accepted"] == 0
        assert client.post("/webhook", json=self_payload(remote=OTHER, lid=True, msg_id="X2")).json()["accepted"] == 0
    assert evo.all_targets() == []


def test_self_chat_text_reply_is_not_reprocessed():
    # The bot's own reply comes back as a fromMe *text* message in the self chat: no loop.
    p = self_payload()
    p["data"]["message"] = {"conversation": "🎙️ Você (15:20)\ntexto\n\nResumo: x"}
    p["data"]["messageType"] = "conversation"
    settings, bot, evo = build()
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=p).json()["accepted"] == 0
    assert evo.all_targets() == []


def test_parse_self_route():
    from app.parse import ROUTE_SELF

    [d] = evaluate_payload(self_payload(), GROUP, allow_dm=True, owner_jid=OWNER)
    assert d.accepted and d.audio.route == ROUTE_SELF
    [no_owner] = evaluate_payload(self_payload(), GROUP, allow_dm=True)  # owner not given → still ignored
    assert not no_owner.accepted
    [third] = evaluate_payload(self_payload(remote=OTHER), GROUP, allow_dm=True, owner_jid=OWNER)
    assert not third.accepted and third.reason == "own dm message"
