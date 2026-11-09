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
    assert text.startswith("🎙️ João · +5511999990000 (15:20)\n")
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
    assert evo.plain[0]["text"].startswith("🎙️ João · +5511999990000 (15:20)\n")  # LID: number from remoteJidAlt


def test_dm_target_works_even_if_owner_unknown():
    settings, bot, evo = build(owner="", dm_transcript_target_jid=AUDIOS)
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=dm_payload()).json()["accepted"] == 1
    assert evo.all_targets() == [AUDIOS]


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


def test_self_chat_audio_stays_in_self_chat_even_with_target_set():
    # Rule 3: the answer always comes back to the owner's own chat, never to the Áudios group.
    settings, bot, evo = build(dm_transcript_target_jid=AUDIOS)
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=self_payload()).json()["accepted"] == 1
    assert evo.plain == []
    [sent] = evo.sent  # quoted reply to the audio, in the owner's own chat
    assert sent["jid"] == OWNER and AUDIOS not in evo.all_targets()
    assert sent["key"]["id"] == "SELFAUDIO1"
    assert sent["text"].startswith("🎙️ Você (15:20)\n")
    assert FakeTranscriber().text in sent["text"]
    assert "\n\nResumo: Reunião do condomínio" in sent["text"]


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


# --- Áudios group (DM_TRANSCRIPT_TARGET_JID): audios sent/forwarded there are transcribed in place ---


def audios_payload(msg_id="AUDIOSGRP1", from_me=False, forwarded=True, timestamp=1794075600):
    p = json.loads(FIXTURE.read_text())
    p["data"]["key"].update(id=msg_id, remoteJid=AUDIOS, fromMe=from_me)
    p["data"]["messageTimestamp"] = timestamp
    p["data"]["pushName"] = "Eu"
    if forwarded:
        p["data"]["message"]["audioMessage"]["contextInfo"] = {"isForwarded": True, "forwardingScore": 1}
    return p


def text_payload(remote, text, msg_id="TXT1", from_me=True, message_type="conversation"):
    p = json.loads(FIXTURE.read_text())
    p["data"]["key"].update(id=msg_id, remoteJid=remote, fromMe=from_me)
    p["data"]["message"] = {"conversation": text} if message_type == "conversation" else {
        "extendedTextMessage": {"text": text}
    }
    p["data"]["messageType"] = message_type
    return p


@pytest.mark.parametrize("from_me", [False, True])
def test_audios_group_accepted_without_listing_it_in_group_jid(from_me):
    # Rule 4: the target group is accepted automatically; GROUP_JID only holds GROUP.
    settings, bot, evo = build(dm_transcript_target_jid=AUDIOS)
    assert AUDIOS not in settings.group_jid
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=audios_payload(from_me=from_me)).json()["accepted"] == 1
    assert evo.plain == []
    [sent] = evo.sent  # quoted reply in the Áudios group itself
    assert sent["jid"] == AUDIOS and sent["key"]["id"] == "AUDIOSGRP1"
    assert sent["text"].startswith("🎙️ Eu (15:20)\n")
    assert FakeTranscriber().text in sent["text"]  # full transcript
    assert "\n\nResumo: Reunião do condomínio" in sent["text"]
    assert "github.com" not in sent["text"] and settings.reply_prefix not in sent["text"]  # no footer


def test_audios_group_old_forwarded_audio_is_still_age_checked_like_any_other():
    settings, bot, evo = build(dm_transcript_target_jid=AUDIOS, max_message_age_seconds=900)
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=audios_payload(timestamp=1)).json()["accepted"] == 0
    assert evo.all_targets() == []


def test_audios_group_without_target_configured_is_other_group():
    settings, bot, evo = build()
    [d] = evaluate_payload(audios_payload(), settings.group_jid)
    assert not d.accepted and d.reason == "other group"
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=audios_payload()).json()["accepted"] == 0
    assert evo.all_targets() == []


def test_audios_group_also_listed_in_group_jid_still_gets_full_transcript_without_footer():
    settings, bot, evo = build(dm_transcript_target_jid=AUDIOS, group_jid=f"{GROUP},{AUDIOS}")
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=audios_payload()).json()["accepted"] == 1
    [sent] = evo.sent
    assert sent["jid"] == AUDIOS and FakeTranscriber().text in sent["text"]
    assert "github.com" not in sent["text"]


def test_selected_group_keeps_summary_reply_and_footer_when_target_is_set(group_payload):
    # Rule 1 is unchanged by the Áudios group.
    settings, bot, evo = build(dm_transcript_target_jid=AUDIOS)
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=group_payload).json()["accepted"] == 1
    [sent] = evo.sent
    assert sent["jid"] == GROUP and evo.plain == []
    assert sent["text"].startswith(settings.reply_prefix)
    assert FakeTranscriber().text not in sent["text"]  # summary only, not the full transcript
    assert sent["text"].endswith("_transcrito por github.com/ustoppble/laschuk-resume_")


def test_third_party_dm_still_goes_to_audios_group_as_plain_text():
    # Rule 2 is unchanged.
    settings, bot, evo = build(dm_transcript_target_jid=AUDIOS)
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=dm_payload()).json()["accepted"] == 1
    assert evo.sent == [] and [m["jid"] for m in evo.plain] == [AUDIOS]
    assert FakeTranscriber().text in evo.plain[0]["text"]


@pytest.mark.parametrize("message_type", ["conversation", "extendedTextMessage"])
@pytest.mark.parametrize("from_me", [True, False])
def test_text_in_audios_group_is_never_processed(message_type, from_me):
    # No loop: the bot's own transcripts are text, so they come back as "not audio" and are dropped.
    settings, bot, evo = build(dm_transcript_target_jid=AUDIOS)
    p = text_payload(AUDIOS, "🎙️ João (15:20)\ntranscrição\n\nResumo: x", from_me=from_me, message_type=message_type)
    [d] = evaluate_payload(p, settings.group_jid, transcript_jid=AUDIOS)
    assert not d.accepted and d.reason.startswith("not audio")
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=p).json()["accepted"] == 0
    assert evo.all_targets() == []


def test_bot_reply_in_audios_group_does_not_trigger_another_reply():
    settings, bot, evo = build(dm_transcript_target_jid=AUDIOS)
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=audios_payload()).json()["accepted"] == 1
        assert client.post("/webhook", json=dm_payload()).json()["accepted"] == 1
    assert len(evo.sent) == 1 and len(evo.plain) == 1
    # Echo both texts back exactly as the webhook would deliver them.
    echoes = [text_payload(AUDIOS, m["text"], msg_id=f"ECHO{i}") for i, m in enumerate(evo.sent + evo.plain)]
    with TestClient(create_app(settings, bot)) as client:
        assert [client.post("/webhook", json=e).json()["accepted"] for e in echoes] == [0, 0]
    assert len(evo.sent) == 1 and len(evo.plain) == 1  # nothing new was sent


def test_parse_transcripts_route():
    from app.parse import ROUTE_TRANSCRIPTS

    [d] = evaluate_payload(audios_payload(), GROUP, transcript_jid=AUDIOS)
    assert d.accepted and d.audio.route == ROUTE_TRANSCRIPTS and d.audio.remote_jid == AUDIOS
    [other] = evaluate_payload(audios_payload(), GROUP)  # no transcript_jid → other group
    assert not other.accepted and other.reason == "other group"
    p = audios_payload()
    p["data"]["key"]["remoteJid"] = "999@g.us"
    [elsewhere] = evaluate_payload(p, GROUP, transcript_jid=AUDIOS)
    assert not elsewhere.accepted and elsewhere.reason == "other group"


# --- sender phone in third-party DM transcripts (fictitious numbers only) ---


def test_format_phone():
    from app.parse import format_phone

    assert format_phone("5511912345678@s.whatsapp.net") == "+5511912345678"  # BR mobile
    assert format_phone("551132165498@s.whatsapp.net") == "+551132165498"  # BR landline
    assert format_phone("5511912345678:7@s.whatsapp.net") == "+5511912345678"  # device suffix
    assert format_phone("14155550123@s.whatsapp.net") == "+14155550123"  # other country
    for bad in ("", "123456789012345@lid", "abc@s.whatsapp.net", "123@s.whatsapp.net", "status@broadcast"):
        assert format_phone(bad) == ""


def test_dm_header_has_name_and_phone_for_phone_jid():
    settings, bot, evo = build(dm_transcript_target_jid=AUDIOS)
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=dm_payload()).json()["accepted"] == 1
    assert evo.plain[0]["text"].startswith("🎙️ João · +5511999990000 (15:20)\n")


def test_dm_header_without_push_name_shows_only_phone():
    p = dm_payload()
    del p["data"]["pushName"]
    settings, bot, evo = build(dm_transcript_target_jid=AUDIOS)
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=p).json()["accepted"] == 1
    assert evo.plain[0]["text"].startswith("🎙️ +5511999990000 (15:20)\n")


def test_dm_lid_uses_sender_pn_when_remote_jid_alt_is_missing():
    p = dm_payload(lid=True)
    p["data"]["key"]["senderPn"] = p["data"]["key"].pop("remoteJidAlt")
    settings, bot, evo = build(dm_transcript_target_jid=AUDIOS)
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=p).json()["accepted"] == 1
    assert evo.plain[0]["text"].startswith("🎙️ João · +5511999990000 (15:20)\n")


def test_dm_header_with_unusable_number_falls_back_to_name_only(monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "format_phone", lambda jid: "")
    settings, bot, evo = build(dm_transcript_target_jid=AUDIOS)
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=dm_payload()).json()["accepted"] == 1
    assert evo.plain[0]["text"].startswith("🎙️ João (15:20)\n")


def test_dm_header_to_owner_chat_also_has_phone():
    settings, bot, evo = build()
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=dm_payload()).json()["accepted"] == 1
    assert evo.plain[0]["text"].startswith("🎙️ João · +5511999990000 (15:20)\n")

@pytest.mark.parametrize("make", [dm_payload, self_payload])
def test_dm_and_self_audio_get_no_reaction(make):
    settings, bot, evo = build()
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=make()).json()["accepted"] == 1
    assert evo.reactions == []
    assert evo.plain or evo.sent  # still delivered


def test_audios_group_audio_gets_reactions():
    settings, bot, evo = build(dm_transcript_target_jid=AUDIOS)
    with TestClient(create_app(settings, bot)) as client:
        assert client.post("/webhook", json=audios_payload()).json()["accepted"] == 1
    assert [r["emoji"] for r in evo.reactions] == ["⏳", "✅"]
