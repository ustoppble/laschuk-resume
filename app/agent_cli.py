"""Daily digests through an AI agent CLI that is already logged in (OAuth), instead of an API key.

Only `codex_cli` (OpenAI Codex CLI logged in with a ChatGPT account) is supported. The CLI runs non-interactively
as a pure text model: read-only sandbox, an empty throwaway working directory, no user config or project rules,
no web search, no saved session. Message text stays DATA inside the prompt (same boundary as the API path), and
the agent is told not to run commands or use tools; with the read-only sandbox it could not write or reach the
network even if a message tried to make it.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile

from .config import Settings
from .summarize import SummaryError

CLI_PROVIDERS = {"codex_cli"}
_AGENT_RULES = (
    "Você está rodando sem ferramentas: NÃO execute comandos, não leia nem grave arquivos e não acesse a internet. "
    "Responda apenas com o texto pedido abaixo."
)


def codex_args(settings: Settings, workdir: str, out_file: str) -> list[str]:
    args = [settings.digest_cli_bin or "codex", "exec", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config",
            "--ignore-rules", "--sandbox", "read-only", "--color", "never", "--cd", workdir,
            "-c", 'approval_policy="never"', "-c", 'web_search="disabled"', "--output-last-message", out_file]
    if settings.digest_cli_model:
        args += ["--model", settings.digest_cli_model]
    if settings.digest_reasoning_effort:
        args += ["-c", f'model_reasoning_effort="{settings.digest_reasoning_effort}"']
    return args + ["-"]  # prompt on stdin, never on the command line (it would show in the process list)


def agent_prompt(system: str, user: str) -> str:
    return f"{_AGENT_RULES}\n\n=== INSTRUÇÕES (sistema) ===\n{system}\n\n=== ENTRADA ===\n{user}\n"


async def cli_complete(settings: Settings, system: str, user: str, *, usage: dict[str, int] | None = None) -> str:
    """Run the logged-in agent once and return its final message. Raises SummaryError on failure or timeout."""
    if settings.digest_provider not in CLI_PROVIDERS:
        raise SummaryError(f"unknown DIGEST_PROVIDER {settings.digest_provider!r}")
    workdir = tempfile.mkdtemp(prefix="digest-agent-")
    out_file = os.path.join(workdir, "out.txt")
    try:
        try:
            proc = await asyncio.create_subprocess_exec(
                *codex_args(settings, workdir, out_file), cwd=workdir, stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        except FileNotFoundError as exc:
            raise SummaryError("agent CLI not found (DIGEST_CLI_BIN)") from exc
        try:
            _, err = await asyncio.wait_for(proc.communicate(agent_prompt(system, user).encode()),
                                            settings.digest_cli_timeout_seconds)
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise SummaryError(f"agent CLI timed out after {settings.digest_cli_timeout_seconds}s") from exc
        if proc.returncode != 0:
            tail = (err or b"").decode(errors="replace").strip().splitlines()[-1:] or [""]
            raise SummaryError(f"agent CLI exit {proc.returncode}: {tail[0][:120]}")
        try:
            with open(out_file, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            raise SummaryError("agent CLI wrote no answer") from exc
        if not text.strip():
            raise SummaryError("agent CLI returned an empty answer")
        if usage is not None:
            usage["calls"] = usage.get("calls", 0) + 1
        return text
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
