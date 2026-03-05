import logging
from typing import Any, Dict

from discord.gateway import DiscordVoiceWebSocket

from .voice_client import VoiceClient


__all__ = ("hook",)


_log = logging.getLogger(__name__)


async def hook(self: DiscordVoiceWebSocket, msg: Dict[str, Any]):
    # TODO: implement other voice events
    op: int = msg["op"]
    data: Dict[str, Any] = msg.get("d", {})
    vc: VoiceClient = self._connection

    if not isinstance(vc, VoiceClient):
        return

    _log.debug("Voice gateway hook op=%s payload_keys=%s", op, tuple(data.keys()))

    if op == DiscordVoiceWebSocket.SPEAKING:
        vc.update_ssrc(data)
    elif op == DiscordVoiceWebSocket.CLIENT_CONNECT:
        vc.on_client_connect(data)
    elif op == DiscordVoiceWebSocket.CLIENT_DISCONNECT:
        vc.on_client_disconnect(data)
