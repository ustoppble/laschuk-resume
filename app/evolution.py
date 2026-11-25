"""Minimal Evolution API v2 client: list groups, fetch media as base64, send a quoted text, react, read history."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from .config import Settings

log = logging.getLogger("laschuk_resume.evolution")


class EvolutionError(RuntimeError):
    pass


class EvolutionClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.s = settings
        self.http = client

    @property
    def _headers(self) -> dict[str, str]:
        return {"apikey": self.s.evolution_api_key, "Content-Type": "application/json"}

    def _url(self, path: str) -> str:
        return f"{self.s.evolution_base}/{path.strip('/')}/{self.s.evolution_instance}"

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = self._url(path)
        try:
            resp = await self.http.request(method, url, headers=self._headers, **kwargs)
        except httpx.HTTPError as exc:
            raise EvolutionError(f"{method} {path}: {exc.__class__.__name__}: {exc}") from exc
        if resp.status_code >= 400:
            raise EvolutionError(f"{method} {path}: HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()
        except ValueError as exc:
            raise EvolutionError(f"{method} {path}: non-JSON response") from exc

    async def fetch_groups(self) -> list[dict[str, Any]]:
        data = await self._request("GET", "group/fetchAllGroups", params={"getParticipants": "false"})
        if isinstance(data, dict):
            data = data.get("groups") or data.get("data") or []
        return data if isinstance(data, list) else []

    async def fetch_profile_name(self) -> str:
        """Display name of the instance's own WhatsApp profile ("" when unknown)."""
        try:
            resp = await self.http.get(f"{self.s.evolution_base}/instance/fetchInstances", headers=self._headers,
                                       params={"instanceName": self.s.evolution_instance}, timeout=10)
            data = resp.json() if resp.status_code < 400 else []
        except (httpx.HTTPError, ValueError):
            return ""
        for item in data if isinstance(data, list) else [data]:
            inst = item.get("instance") if isinstance(item, dict) and isinstance(item.get("instance"), dict) else item
            if isinstance(inst, dict) and (inst.get("name") or inst.get("instanceName")) == self.s.evolution_instance:
                return str(inst.get("profileName") or "").strip()
        return ""

    async def find_contact_name(self, jid: str) -> str:
        """Saved/push name of one contact via /chat/findContacts ("" when unknown)."""
        try:
            data = await self._request("POST", "chat/findContacts", json={"where": {"remoteJid": jid}})
        except EvolutionError:
            return ""
        for c in data if isinstance(data, list) else []:
            if isinstance(c, dict) and str(c.get("pushName") or "").strip():
                return str(c["pushName"]).strip()
        return ""

    async def fetch_owner_jid(self) -> str:
        """ownerJid (the instance's own WhatsApp number) via /instance/fetchInstances."""
        url = f"{self.s.evolution_base}/instance/fetchInstances"
        try:
            resp = await self.http.get(
                url, headers=self._headers, params={"instanceName": self.s.evolution_instance}, timeout=10
            )
        except httpx.HTTPError as exc:
            raise EvolutionError(f"GET instance/fetchInstances: {exc.__class__.__name__}: {exc}") from exc
        if resp.status_code >= 400:
            raise EvolutionError(f"GET instance/fetchInstances: HTTP {resp.status_code}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise EvolutionError("GET instance/fetchInstances: non-JSON response") from exc
        for item in data if isinstance(data, list) else [data]:
            if not isinstance(item, dict):
                continue
            inst = item.get("instance") if isinstance(item.get("instance"), dict) else item
            name = inst.get("name") or inst.get("instanceName")
            if name != self.s.evolution_instance:
                continue
            owner = str(inst.get("ownerJid") or inst.get("owner") or "")
            if owner.endswith("@s.whatsapp.net"):
                return owner
        raise EvolutionError(f"no ownerJid for instance {self.s.evolution_instance!r}")

    async def send_text(self, jid: str, text: str, mentioned: list[str] | None = None) -> dict[str, Any]:
        """Plain (unquoted) text message. `mentioned` = JIDs to turn the matching `@<number>` in `text`
        into real WhatsApp mentions (Evolution v2 field `mentioned`)."""
        body: dict[str, Any] = {"number": jid, "text": text, "linkPreview": False}
        if mentioned:
            body["mentioned"] = list(mentioned)
        return await self._request("POST", "message/sendText", json=body)

    async def find_group_info(self, group_jid: str) -> dict[str, Any]:
        """Subject + participants of one group via /group/findGroupInfos."""
        data = await self._request("GET", "group/findGroupInfos", params={"groupJid": group_jid})
        return data if isinstance(data, dict) else {}

    async def find_messages(
        self, remote_jid: str | None, start_ts: float, end_ts: float, *, page_size: int = 250, max_pages: int = 80
    ) -> list[dict[str, Any]]:
        """Stored messages between two epoch timestamps (inclusive), via /chat/findMessages; one chat, or every
        chat of the instance when `remote_jid` is None.

        Needs Evolution to keep message history (DATABASE_SAVE_DATA_NEW_MESSAGE). Records come newest first
        and are paged; the time filter is re-applied here because servers differ in how they apply it.
        """
        def iso(ts: float) -> str:
            return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")

        where: dict[str, Any] = {"messageTimestamp": {"gte": iso(start_ts), "lte": iso(end_ts)}}
        if remote_jid:
            where["key"] = {"remoteJid": remote_jid}
        records: list[dict[str, Any]] = []
        for page in range(1, max_pages + 1):
            data = await self._request(
                "POST", "chat/findMessages", json={"where": where, "page": page, "offset": page_size}
            )
            box = (data or {}).get("messages") if isinstance(data, dict) else None
            batch = (box or {}).get("records") if isinstance(box, dict) else None
            if not isinstance(batch, list) or not batch:
                break
            records.extend(r for r in batch if isinstance(r, dict))
            pages = box.get("pages") if isinstance(box, dict) else None
            if not isinstance(pages, int) or page >= pages:
                break
        else:
            log.warning("find_messages stopped at %s pages; older messages of the window were skipped", max_pages)
        return records

    async def get_media_base64(self, key: dict[str, Any]) -> tuple[str, str]:
        """Return (base64, mimetype) for a media message via /chat/getBase64FromMediaMessage."""
        data = await self._request(
            "POST",
            "chat/getBase64FromMediaMessage",
            json={"message": {"key": key}, "convertToMp4": False},
        )
        b64 = (data or {}).get("base64") if isinstance(data, dict) else None
        if not b64:
            raise EvolutionError("getBase64FromMediaMessage returned no base64")
        return b64, str(data.get("mimetype") or "")

    async def send_text_reply(
        self, jid: str, text: str, quoted_key: dict[str, Any], quoted_message: dict[str, Any]
    ) -> dict[str, Any]:
        # Strip inline media from the quote: Evolution only needs key + message shape.
        message = {k: v for k, v in (quoted_message or {}).items() if k != "base64"}
        # linkPreview=false: no link-preview card cluttering the summary.
        return await self._request(
            "POST",
            "message/sendText",
            json={
                "number": jid,
                "text": text,
                "linkPreview": False,
                "quoted": {"key": quoted_key, "message": message},
            },
        )

    async def send_reaction(self, remote_jid: str, key: dict[str, Any], emoji: str) -> dict[str, Any]:
        """React to a message via /message/sendReaction. In groups the key must keep `participant`."""
        return await self._request(
            "POST",
            "message/sendReaction",
            json={"key": {**key, "remoteJid": remote_jid}, "reaction": emoji},
        )
