from __future__ import annotations

import pickle
import socket
import struct
from typing import Any

_HEADER = struct.Struct('!Q')
_MAX_MESSAGE_BYTES = 128 * 1024 * 1024


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError('Policy RPC connection closed while receiving data.')
        chunks.extend(chunk)
    return bytes(chunks)


def send_message(connection: socket.socket, payload: Any) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    if len(data) > _MAX_MESSAGE_BYTES:
        raise ValueError(f'Policy RPC message is too large: {len(data)} bytes.')
    connection.sendall(_HEADER.pack(len(data)))
    connection.sendall(data)


def receive_message(connection: socket.socket) -> Any:
    (size,) = _HEADER.unpack(_recv_exact(connection, _HEADER.size))
    if size <= 0 or size > _MAX_MESSAGE_BYTES:
        raise ValueError(f'Invalid policy RPC message size: {size}.')
    return pickle.loads(_recv_exact(connection, size))  # nosec: trusted localhost process boundary


class PolicyRPCClient:
    def __init__(self, *, host: str = '127.0.0.1', port: int = 8765, timeout: float = 120.0):
        self.host = str(host)
        self.port = int(port)
        self.timeout = float(timeout)
        self._connection: socket.socket | None = None

    def connect(self) -> None:
        if self._connection is not None:
            return
        self._connection = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._connection.settimeout(self.timeout)

    def close(self) -> None:
        if self._connection is None:
            return
        try:
            self._connection.close()
        finally:
            self._connection = None

    def request(self, command: str, **payload) -> dict[str, Any]:
        self.connect()
        assert self._connection is not None
        send_message(self._connection, {'command': str(command), **payload})
        response = receive_message(self._connection)
        if not isinstance(response, dict):
            raise RuntimeError(f'Invalid policy RPC response: {type(response)!r}.')
        if not bool(response.get('ok', False)):
            raise RuntimeError(str(response.get('error', 'Policy RPC request failed.')))
        return response

    def ping(self) -> dict[str, Any]:
        return self.request('ping')

    def reset(self) -> None:
        self.request('reset')

    def predict(self, observation: dict[str, Any], *, task: str) -> Any:
        return self.request('predict', observation=observation, task=str(task))['action']

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
