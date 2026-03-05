import asyncio
import logging
import queue
import select
import threading
from concurrent.futures import Future
from typing import Any, Awaitable, Callable, Dict, Optional, Union

try:
    import davey  # type: ignore
except ImportError:
    davey = None

from discord.errors import ClientException
from discord.member import Member
from discord.object import Object
from discord.voice_client import VoiceClient as BaseVoiceClient
from discord.gateway import DiscordVoiceWebSocket

from . import opus
from .enums import RTCPMessageType
from .processing import AudioProcessPool
from .sink import SILENT_FRAME, AudioFrame, AudioSink

__all__ = ("VoiceClient",)


_log = logging.getLogger(__name__)


class AsyncEventWrapper:
    def __init__(self, event: Optional[threading.Event] = None):
        self.event: threading.Event = event or threading.Event()
        self._waiters: queue.Queue = queue.Queue()

    def __getattr__(self, item):
        return getattr(self.event, item)

    def set(self) -> None:
        self.event.set()
        # Queue.empty() is not reliable, so instead we just catch when the queue throws an Empty error
        try:
            while True:
                future = self._waiters.get_nowait()
                future._loop.call_soon_threadsafe(future.set_result, True)
        except queue.Empty:
            pass

    async def async_wait(self, loop) -> None:
        if self.is_set():
            return
        future = loop.create_future()
        self._waiters.put(future)
        await future


class AudioReceiver(threading.Thread):
    def __init__(
        self,
        client: 'VoiceClient',
    ) -> None:
        super().__init__()
        self.sink: Optional[AudioSink] = None
        self.process_pool: Optional[AudioProcessPool] = None
        self.client: VoiceClient = client
        self.loop = self.client.client.loop

        self.decode: bool = True
        self.decoders: Dict[int, opus.Decoder] = {}
        self.after: Optional[Callable[..., Awaitable[Any]]] = None
        self.after_kwargs: Optional[dict] = None

        self._end: AsyncEventWrapper = AsyncEventWrapper()
        self._on_standby: AsyncEventWrapper = AsyncEventWrapper()
        self._on_standby.set()
        self._resumed: AsyncEventWrapper = AsyncEventWrapper()
        self._clean: AsyncEventWrapper = AsyncEventWrapper()
        self._clean.set()
        self._connected: threading.Event = client._connected

    def _do_run(self) -> None:
        while not self._end.is_set():
            if not self._connected.is_set():
                self._connected.wait()

            data = self.client.recv_audio(dump=not self._resumed.is_set())
            if data is None:
                continue

            dave_active = self.client.should_decrypt_dave()
            _log.debug(
                "Received raw UDP packet: size=%d mode=%s decode=%s dave_active=%s",
                len(data),
                self.client.mode,
                self.decode,
                dave_active,
            )

            future = self.process_pool.submit(  # type: ignore
                data,
                self.client.guild.id % self.process_pool.max_processes,  # type: ignore
                self.decode and not dave_active,
                self.client.mode,
                self.client.secret_key,
            )
            future.add_done_callback(self._audio_processing_callback)

    def _audio_processing_callback(self, future: Future) -> None:
        try:
            packet = future.result()
        except BaseException as exc:
            _log.exception("Exception occurred in audio process", exc_info=exc)
            return
        if self.sink is None:
            return
        if isinstance(packet, AudioFrame):
            sink_callback = self.sink.on_audio
            packet.user = self.client.get_member_from_ssrc(packet.ssrc)
            _log.debug(
                "Processed RTP frame: ssrc=%s seq=%s ts=%s opus_len=%d user_resolved=%s",
                packet.ssrc,
                packet.sequence,
                packet.timestamp,
                len(packet.audio),
                packet.user.id if packet.user is not None else None,
            )

            if self.client.should_decrypt_dave() and packet.audio:
                packet = self._dave_decrypt_packet(packet)
                if packet is None:
                    _log.debug("Dropping RTP frame after DAVE decrypt attempt")
                    return

            if self.decode and packet.audio != SILENT_FRAME:
                packet = self._decode_packet(packet)
                if packet is None:
                    _log.debug("Dropping RTP frame after Opus decode attempt")
                    return
        else:
            sink_callback = self.sink.on_rtcp
            packet.pt = RTCPMessageType(packet.pt)
            _log.debug("Processed RTCP packet: type=%s", packet.pt)

        _log.debug("Delivering packet to sink callback: %s", sink_callback.__name__)
        sink_callback(packet)  # type: ignore

    def _dave_decrypt_packet(self, packet: AudioFrame) -> Optional[AudioFrame]:
        if davey is None:
            _log.debug("DAVE decryption requested but davey is unavailable")
            return packet

        state = self.client._connection
        dave_session = getattr(state, "dave_session", None)
        if dave_session is None:
            _log.debug("DAVE decryption requested but dave_session is unavailable")
            return packet

        if packet.user is None:
            _log.debug("Cannot DAVE decrypt packet for ssrc=%s: unresolved user", packet.ssrc)
            return None

        try:
            original_len = len(packet.audio)
            packet.audio = dave_session.decrypt(packet.user.id, davey.MediaType.audio, packet.audio)
            _log.debug(
                "DAVE decrypt ok: ssrc=%s user_id=%s encrypted_len=%d decrypted_len=%d",
                packet.ssrc,
                packet.user.id,
                original_len,
                len(packet.audio),
            )
            return packet
        except Exception as exc:
            _log.debug("Failed to decrypt DAVE packet for ssrc %s", packet.ssrc, exc_info=exc)
            return None

    def _decode_packet(self, packet: AudioFrame) -> Optional[AudioFrame]:
        try:
            if packet.ssrc not in self.decoders:
                _log.debug("Creating opus decoder for ssrc=%s", packet.ssrc)
                self.decoders[packet.ssrc] = opus.Decoder()
            encoded_len = len(packet.audio)
            packet.audio = self.decoders[packet.ssrc].decode(packet.audio)
            _log.debug(
                "Opus decode ok: ssrc=%s encoded_len=%d pcm_len=%d",
                packet.ssrc,
                encoded_len,
                len(packet.audio),
            )
            return packet
        except Exception as exc:
            _log.debug("Failed to decode opus packet for ssrc %s", packet.ssrc, exc_info=exc)
            return None

    def run(self) -> None:
        try:
            self._do_run()
        except Exception as exc:
            self.stop()
            _log.exception("Exception occurred in voice receiver", exc_info=exc)

    def _call_after(self) -> None:
        if self.after is not None:
            try:
                kwargs = self.after_kwargs if self.after_kwargs is not None else {}
                asyncio.run_coroutine_threadsafe(self.after(self.sink, **kwargs), self.loop)
            except Exception as exc:
                _log.exception('Calling the after function failed.', exc_info=exc)

    def _cleanup_listen(self) -> None:
        if self.sink is not None:
            threading.Thread(target=self.sink.cleanup).start()
            self._call_after()
            self.sink = None
        else:
            _log.warning("Could not call cleanup on sink because the sink attribute is None")
        self._clean.set()

    def start_listening(
        self,
        sink: AudioSink,
        processing_pool: AudioProcessPool,
        *,
        decode: bool = True,
        after: Optional[Callable[..., Awaitable[Any]]] = None,
        after_kwargs: Optional[dict] = None,
    ) -> None:
        self.sink = sink
        self.process_pool = processing_pool
        self.decode = decode
        self.after = after
        self.after_kwargs = after_kwargs
        self._on_standby.clear()
        self._clean.clear()
        self._resumed.set()

    def stop(self) -> None:
        self._end.set()

    def stop_listening(self) -> None:
        self._resumed.clear()
        self._on_standby.set()
        self._cleanup_listen()

    def pause(self) -> None:
        self._resumed.clear()

    def resume(self) -> None:
        self._resumed.set()

    def is_done(self) -> bool:
        return self._end.is_set()

    def is_listening(self) -> bool:
        return self._resumed.is_set() and not self._on_standby.is_set()

    def is_paused(self) -> bool:
        return not self._resumed.is_set() and not self._on_standby.is_set()

    def is_on_standby(self) -> bool:
        return self._on_standby.is_set()

    def is_cleaning(self) -> bool:
        return self._on_standby.is_set() and not self._clean.is_set()

    async def wait_for_resumed(self, *, loop=None) -> None:
        await self._resumed.async_wait(self.loop if loop is None else loop)

    async def wait_for_standby(self, *, loop=None) -> None:
        await self._on_standby.async_wait(self.loop if loop is None else loop)

    async def wait_for_clean(self, *, loop=None) -> None:
        await self._clean.async_wait(self.loop if loop is None else loop)


class VoiceClient(BaseVoiceClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._receiver: Optional[AudioReceiver] = None
        self._ssrc_map: Dict[int, Dict[str, Union[Member, Object]]] = {}

    async def on_voice_server_update(self, data) -> None:
        await super().on_voice_server_update(data)

        self._receiver = AudioReceiver(self)
        self._receiver.start()

    def should_decrypt_dave(self) -> bool:
        state = self._connection
        active = getattr(state, "dave_protocol_version", 0) > 0 and getattr(state, "dave_session", None) is not None
        _log.debug(
            "DAVE status: protocol_version=%s session_present=%s can_encrypt=%s active=%s",
            getattr(state, "dave_protocol_version", None),
            getattr(state, "dave_session", None) is not None,
            getattr(state, "can_encrypt", None),
            active,
        )
        return active

    async def disconnect(self, *, force=False):
        if not force and not self.is_connected():
            return

        if self._receiver is not None:
            self._receiver.stop()
        await super().disconnect(force=force)

    async def connect_websocket(self) -> DiscordVoiceWebSocket:
        from .gateway import hook

        ws = await DiscordVoiceWebSocket.from_client(self, hook=hook)
        self._connected.clear()
        while ws.secret_key is None:
            await ws.poll_event()
        self._connected.set()
        return ws

    def update_ssrc(self, data):
        ssrc = data["ssrc"]
        user_id = int(data["user_id"])
        speaking = data["speaking"]
        if ssrc in self._ssrc_map:
            self._ssrc_map[ssrc]["speaking"] = speaking
        else:
            user = self.guild.get_member(user_id)
            self._ssrc_map[ssrc] = {
                "user": user if user is not None else Object(id=user_id, type=Member),
                "speaking": speaking,
            }

    def on_client_connect(self, data):
        user_id = int(data["user_id"])
        audio_ssrc = data.get("audio_ssrc")
        if audio_ssrc is None:
            _log.debug("CLIENT_CONNECT missing audio_ssrc for user_id=%s", user_id)
            return

        user = self.guild.get_member(user_id)
        self._ssrc_map[audio_ssrc] = {
            "user": user if user is not None else Object(id=user_id, type=Member),
            "speaking": self._ssrc_map.get(audio_ssrc, {}).get("speaking", 0),
        }
        _log.debug("Mapped audio_ssrc=%s to user_id=%s via CLIENT_CONNECT", audio_ssrc, user_id)

    def on_client_disconnect(self, data):
        user_id = int(data["user_id"])
        removed = 0
        for ssrc, info in list(self._ssrc_map.items()):
            user = info.get("user")
            if user is not None and user.id == user_id:
                self._ssrc_map.pop(ssrc, None)
                removed += 1
        _log.debug("CLIENT_DISCONNECT user_id=%s removed_mappings=%d", user_id, removed)

    def get_member_from_ssrc(self, ssrc) -> Optional[Union[Member, Object]]:
        if ssrc in self._ssrc_map:
            user = self._ssrc_map[ssrc]["user"]
            if isinstance(user, Object) and (member := self.guild.get_member(user.id)) is not None:
                self._ssrc_map[ssrc]["user"] = member
                return member
            return user

    def listen(
        self,
        sink: AudioSink,
        processing_pool: AudioProcessPool,
        *,
        decode: bool = True,
        supress_warning: bool = False,
        after: Optional[Callable[..., Awaitable[Any]]] = None,
        **kwargs,
    ) -> None:
        """Receives audio into an :class:`AudioSink`

        IMPORTANT: If you call this function, the running section of your code should be
        contained within an `if __name__ == "__main__"` statement to avoid conflicts with
        multiprocessing that result in the asyncio event loop dying.

        The finalizer, ``after`` is called after listening has stopped or
        an error has occurred.

        If an error happens while the audio receiver is running, the exception is
        caught and the audio receiver is then stopped.  If no after callback is
        passed, any caught exception will be logged using the library logger.

        If this function is called multiple times on the same voice client,
        it is recommended to use  wait_for_listen_ready before making the
        next call to avoid errors.

        Parameters
        -----------
        sink: :class:`AudioSink`
            The audio sink we're passing audio to.
        processing_pool: :class:`AudioProcessPool`
            A process pool where received audio packets will be submitted for processing.
        decode: :class:`bool`
            Whether to decode data received from discord.
        supress_warning: :class:`bool`
            Whether to supress the warning raised when listen is run unsafely.
        after: Callable[..., Awaitable[Any]]
            The finalizer that is called after the receiver stops. This function
            must be a coroutine function. This function must have at least two
            parameters, ``sink`` and ``error``, that denote, respectfully, the
            sink passed to this function and an optional exception that was
            raised during playing. The function can have additional arguments
            that match the keyword arguments passed to this function.

        Raises
        -------
        ClientException
            Already listening or not connected.
        TypeError
            sink is not an :class:`AudioSink` or after is not a callable.
        OpusNotLoaded
            Opus, required to decode audio, is not loaded.
        """
        if not self.is_connected():
            raise ClientException('Not connected to voice.')

        if self.is_listen_receiving():
            raise ClientException('Listening is already active.')

        if not isinstance(sink, AudioSink):
            raise TypeError(f"sink must be an AudioSink not {sink.__class__.__name__}")

        if not supress_warning and self.is_listen_cleaning():
            _log.warning(
                "Cleanup is still in progress for the last call to listen and so errors may occur. "
                "It is recommended to use wait_for_listen_ready before calling listen unless you "
                "know what you're doing."
            )

        if decode:
            # Check that opus is loaded and throw error else
            opus.Decoder.get_opus_version()

        self._receiver.start_listening(sink, processing_pool, decode=decode, after=after, after_kwargs=kwargs)  # type: ignore

    def is_listening(self) -> bool:
        """Indicates if the client is currently listening and processing audio."""
        return self._receiver is not None and self._receiver.is_listening()

    def is_listening_paused(self) -> bool:
        """Indicate if the client is currently listening, but paused (not processing audio)."""
        return self._receiver is not None and self._receiver.is_paused()

    def is_listen_receiving(self) -> bool:
        """Indicates whether listening is active, regardless of the pause state."""
        return self._receiver is not None and not self._receiver.is_on_standby()

    def is_listen_cleaning(self) -> bool:
        """Check if the receiver is cleaning up."""
        return self._receiver is not None and self._receiver.is_cleaning()

    def stop_listening(self) -> None:
        """Stops listening"""
        if self._receiver:
            self._receiver.stop_listening()

    def pause_listening(self) -> None:
        """Pauses listening"""
        if self._receiver:
            self._receiver.pause()

    def resume_listening(self) -> None:
        """Resumes listening"""
        if self._receiver:
            self._receiver.resume()

    async def wait_for_listen_ready(self) -> None:
        """|coro|

        Wait till it's safe to make a call to listen.
        Basically waits for is_listen_receiving and is_listen_cleaning to be false.
        """
        if self._receiver is None:
            return
        await self._receiver.wait_for_standby()
        await self._receiver.wait_for_clean()

    def recv_audio(self, *, dump: bool = False) -> Optional[bytes]:
        """Attempts to receive raw audio and returns it, otherwise nothing.

        You must be connected to receive audio.

        Logs any error thrown by the connection socket.

        Parameters
        ----------
        dump: :class:`bool`
            Will not return data if true

        Returns
        -------
        Optional[bytes]
            If audio was received then it's returned.
        """
        ready, _, err = select.select([self.socket], [], [self.socket], 0.01)
        if err:
            _log.error(f"Socket error: {err[0]}")
            return
        if not ready or not self.is_connected():
            return

        data = self.socket.recv(4096)
        if dump:
            return
        return data
