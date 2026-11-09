import copy
import json
from pathlib import Path

import pytest

from app.parse import (
    clean_summary,
    evaluate_payload,
    find_group_jid,
    is_upsert_event,
    mime_extension,
    unwrap_message,
)

FIXTURE = Path(__file__).parent / "fixtures" / "messages_upsert_audio.json"
GROUP = "120363000000000001@g.us"
NOW = 1730900000 + 60


@pytest.fixture
def payload():
    return json.loads(FIXTURE.read_text())


def evaluate(p, group=GROUP, **kw):
    kw.setdefault("now", NOW)
    return evaluate_payload(p, group, **kw)


def test_sample_audio_is_accepted(payload):
    [d] = evaluate(payload)
    assert d.accepted and d.reason == "audio"
    a = d.audio
    assert a.message_id == "3EB0A1B2C3D4E5F6A7B8"
    assert a.remote_jid == GROUP
    assert a.participant == "5511999990000@s.whatsapp.net"
    assert a.push_name == "Maria"
    assert a.mimetype.startswith("audio/ogg")
    assert a.seconds == 14 and a.ptt is True
    assert a.key == {
        "id": "3EB0A1B2C3D4E5F6A7B8",
        "remoteJid": GROUP,
        "fromMe": False,
        "participant": "5511999990000@s.whatsapp.net",
    }
    assert a.inline_base64 == ""


@pytest.mark.parametrize("event", ["messages.upsert", "MESSAGES_UPSERT", "messages-upsert"])
def test_event_name_variants(payload, event):
    payload["event"] = event
    assert is_upsert_event(payload)
    assert evaluate(payload)[0].accepted


@pytest.mark.parametrize("event", ["messages.update", "send.message", "connection.update", None])
def test_other_events_ignored(payload, event):
    payload["event"] = event
    assert evaluate(payload) == []


def test_own_group_message_accepted(payload):
    # The bot runs on the owner's own WhatsApp: the owner's audios in the group are summarized too.
    payload["data"]["key"]["fromMe"] = True
    [d] = evaluate(payload)
    assert d.accepted


def test_other_group_ignored(payload):
    [d] = evaluate(payload, group="120363999999999999@g.us")
    assert not d.accepted and d.reason == "other group"


def test_unresolved_group_rejects_everything(payload):
    [d] = evaluate(payload, group="")
    assert not d.accepted


def test_private_chat_ignored(payload):
    payload["data"]["key"]["remoteJid"] = "5511999990000@s.whatsapp.net"
    [d] = evaluate(payload)
    assert not d.accepted and d.reason == "not a group message"


def test_text_message_ignored(payload):
    payload["data"]["message"] = {"conversation": "oi"}
    payload["data"]["messageType"] = "conversation"
    [d] = evaluate(payload)
    assert not d.accepted and d.reason.startswith("not audio")


def test_missing_id_ignored(payload):
    del payload["data"]["key"]["id"]
    [d] = evaluate(payload)
    assert not d.accepted and d.reason == "no message id"


def test_old_message_ignored(payload):
    [d] = evaluate(payload, now=1730900000 + 3600, max_age_seconds=900)
    assert not d.accepted and d.reason.startswith("too old")


def test_age_limit_disabled(payload):
    assert evaluate(payload, now=1730900000 + 10**6, max_age_seconds=0)[0].accepted


def test_long_timestamp_object(payload):
    payload["data"]["messageTimestamp"] = {"low": 1730900000, "high": 0, "unsigned": True}
    [d] = evaluate(payload, max_age_seconds=900)
    assert d.accepted and d.audio.timestamp == 1730900000


def test_ephemeral_wrapped_audio(payload):
    audio = payload["data"]["message"]["audioMessage"]
    payload["data"]["message"] = {"ephemeralMessage": {"message": {"audioMessage": audio}}}
    [d] = evaluate(payload)
    assert d.accepted and d.audio.seconds == 14


def test_inline_base64_from_webhook(payload):
    payload["data"]["message"]["base64"] = "T2dnUw=="
    [d] = evaluate(payload)
    assert d.audio.inline_base64 == "T2dnUw=="


def test_list_data_evaluates_each(payload):
    first = payload["data"]
    second = copy.deepcopy(first)
    second["key"]["id"] = "OTHER"
    second["key"]["remoteJid"] = "999@g.us"
    payload["data"] = [first, second]
    decisions = evaluate(payload)
    assert [d.accepted for d in decisions] == [True, False]


def test_garbage_payloads_do_not_crash():
    assert evaluate_payload({}, GROUP) == []
    assert evaluate_payload({"event": "messages.upsert", "data": "x"}, GROUP) == []
    assert evaluate_payload({"event": "messages.upsert", "data": [1, None]}, GROUP) == []
    [d] = evaluate_payload({"event": "messages.upsert", "data": {}}, GROUP)
    assert not d.accepted


def test_unwrap_plain_message():
    assert unwrap_message({"audioMessage": {}}) == {"audioMessage": {}}
    assert unwrap_message(None) == {}


GROUPS = [
    {"id": "111@g.us", "subject": "Família"},
    {"id": GROUP, "subject": "  Hillside "},
    {"id": "222@g.us", "subject": "Hillside 2"},
]


def test_find_group_jid_by_name():
    assert find_group_jid(GROUPS, "Hillside") == GROUP
    assert find_group_jid(GROUPS, "hillside") == GROUP
    assert find_group_jid({"groups": GROUPS}, "Hillside") == GROUP


def test_find_group_jid_missing():
    with pytest.raises(LookupError, match="no WhatsApp group"):
        find_group_jid(GROUPS, "Nope")


def test_find_group_jid_ambiguous():
    with pytest.raises(LookupError, match="several groups"):
        find_group_jid(GROUPS + [{"id": "333@g.us", "subject": "Hillside"}], "Hillside")


@pytest.mark.parametrize(
    "mime,ext",
    [("audio/ogg; codecs=opus", "ogg"), ("audio/mpeg", "mp3"), ("audio/mp4", "m4a"), ("weird", "ogg")],
)
def test_mime_extension(mime, ext):
    assert mime_extension(mime) == ext


def test_clean_summary_limits_lines():
    text = "Resumo:\n\n- linha 1\n- linha 2\n\n- linha 3\n- linha 4\n- linha 5\n"
    assert clean_summary(text) == "- linha 1\n- linha 2\n- linha 3\n- linha 4"


def test_clean_summary_truncates():
    out = clean_summary("a" * 1000, max_chars=100)
    assert len(out) == 100 and out.endswith("…")
