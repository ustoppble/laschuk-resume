"""HTTP shape tests for the Evolution / OpenAI / Anthropic clients using httpx.MockTransport."""
import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.evolution import EvolutionClient, EvolutionError
from app.summarize import AnthropicSummarizer, OpenAISummarizer
from app.transcribe import OpenAITranscriber, TranscriptionError


def settings(**kw):
    base = dict(evolution_url="http://evo.invalid/", evolution_instance="inst", evolution_api_key="KEY",
                openai_api_key="OK", openai_summary_model="some-model", anthropic_api_key="AK")
    base.update(kw)
    return Settings(**base)


def run(coro):
    return asyncio.run(coro)


def client_with(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_evolution_endpoints_and_headers():
    seen = []

    def handler(req: httpx.Request):
        seen.append(req)
        if req.url.path == "/group/fetchAllGroups/inst":
            return httpx.Response(200, json=[{"id": "1@g.us", "subject": "Meu Grupo"}])
        if req.url.path == "/chat/getBase64FromMediaMessage/inst":
            return httpx.Response(200, json={"base64": "QQ==", "mimetype": "audio/ogg"})
        if req.url.path == "/message/sendText/inst":
            return httpx.Response(201, json={"key": {"id": "R"}})
        return httpx.Response(404)

    async def go():
        async with client_with(handler) as http:
            evo = EvolutionClient(settings(), http)
            groups = await evo.fetch_groups()
            media = await evo.get_media_base64({"id": "M", "remoteJid": "1@g.us"})
            await evo.send_text_reply("1@g.us", "oi", {"id": "M"}, {"audioMessage": {}, "base64": "BIG"})
            return groups, media

    groups, media = run(go())
    assert groups[0]["subject"] == "Meu Grupo"
    assert media == ("QQ==", "audio/ogg")
    assert all(r.headers["apikey"] == "KEY" for r in seen)
    assert seen[0].url.params["getParticipants"] == "false"
    assert json.loads(seen[1].content) == {"message": {"key": {"id": "M", "remoteJid": "1@g.us"}}, "convertToMp4": False}
    body = json.loads(seen[2].content)
    assert body == {"number": "1@g.us", "text": "oi", "linkPreview": False, "quoted": {"key": {"id": "M"}, "message": {"audioMessage": {}}}}


def test_evolution_send_reaction_request():
    seen = []

    def handler(req: httpx.Request):
        seen.append(req)
        return httpx.Response(201, json={"key": {"id": "R"}})

    key = {"remoteJid": "1@g.us", "fromMe": False, "id": "M", "participant": "5500000000000@s.whatsapp.net"}

    async def go():
        async with client_with(handler) as http:
            await EvolutionClient(settings(), http).send_reaction("1@g.us", key, "⏳")

    run(go())
    assert seen[0].method == "POST"
    assert seen[0].url.path == "/message/sendReaction/inst"
    assert seen[0].headers["apikey"] == "KEY"
    assert json.loads(seen[0].content) == {"key": key, "reaction": "⏳"}


def test_evolution_http_error_raises():
    async def go():
        async with client_with(lambda r: httpx.Response(401, text="Unauthorized")) as http:
            await EvolutionClient(settings(), http).fetch_groups()

    with pytest.raises(EvolutionError, match="HTTP 401"):
        run(go())


def test_evolution_missing_base64_raises():
    async def go():
        async with client_with(lambda r: httpx.Response(200, json={})) as http:
            await EvolutionClient(settings(), http).get_media_base64({"id": "M"})

    with pytest.raises(EvolutionError, match="no base64"):
        run(go())


def test_openai_transcriber_request():
    captured = {}

    def handler(req):
        captured["url"] = str(req.url)
        captured["auth"] = req.headers["authorization"]
        captured["body"] = req.content
        return httpx.Response(200, json={"text": " olá mundo "})

    async def go():
        async with client_with(handler) as http:
            return await OpenAITranscriber(settings(), http).transcribe(b"OggS", "audio/ogg; codecs=opus")

    assert run(go()) == "olá mundo"
    assert captured["url"] == "https://api.openai.com/v1/audio/transcriptions"
    assert captured["auth"] == "Bearer OK"
    assert b'name="model"' in captured["body"] and b"whisper-1" in captured["body"]
    assert b'filename="audio.ogg"' in captured["body"]


def test_openai_transcriber_rejects_oversize():
    async def go():
        async with client_with(lambda r: httpx.Response(200)) as http:
            await OpenAITranscriber(settings(), http).transcribe(b"0" * (26 * 1024 * 1024), "audio/ogg")

    with pytest.raises(TranscriptionError, match="25 MB"):
        run(go())


def test_anthropic_summarizer_request():
    captured = {}

    def handler(req):
        captured["headers"] = req.headers
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "Linha 1\nLinha 2\nLinha 3\nLinha 4\nLinha 5"}]})

    async def go():
        async with client_with(handler) as http:
            return await AnthropicSummarizer(settings(), http).summarize("texto", "Maria")

    out = run(go())
    assert out.splitlines() == ["Linha 1", "Linha 2", "Linha 3", "Linha 4"]
    assert captured["headers"]["x-api-key"] == "AK"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["body"]["model"] == "claude-sonnet-5-5"
    # adaptive thinking is on by default and counts toward max_tokens: leave room for it
    assert captured["body"]["max_tokens"] >= 4096
    assert not {"temperature", "top_p", "top_k"} & captured["body"].keys()
    assert "português" in captured["body"]["system"]
    assert "Maria" in captured["body"]["messages"][0]["content"]


def test_anthropic_summarizer_ignores_thinking_blocks_and_honours_model_override():
    captured = {}

    def handler(req):
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={
            "stop_reason": "end_turn",
            "content": [{"type": "thinking", "thinking": "", "signature": "sig"},
                        {"type": "text", "text": "Linha A\nLinha B"}],
        })

    async def go():
        async with client_with(handler) as http:
            return await AnthropicSummarizer(settings(anthropic_model="claude-opus-5-5"), http).summarize("texto")

    assert run(go()) == "Linha A\nLinha B"
    assert captured["body"]["model"] == "claude-opus-5-5"


def test_anthropic_model_comes_from_env(monkeypatch):
    from app.config import load_settings

    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
    assert load_settings(env_file="/nonexistent/.env").anthropic_model == "claude-haiku-5-5"
    monkeypatch.delenv("ANTHROPIC_MODEL")
    assert load_settings(env_file="/nonexistent/.env").anthropic_model == "claude-sonnet-5-5"


def test_openai_summarizer_request():
    def handler(req):
        body = json.loads(req.content)
        assert body["model"] == "some-model"
        return httpx.Response(200, json={"choices": [{"message": {"content": "A\nB"}}]})

    async def go():
        async with client_with(handler) as http:
            return await OpenAISummarizer(settings(), http).summarize("texto")

    assert run(go()) == "A\nB"
