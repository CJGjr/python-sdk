"""Vendor-defined (non-spec) methods and raw handler registration on the low-level Server.

The vendor round trips play the client's side of the wire by hand over memory streams: the typed
client cannot send a request or notification for a method outside its closed unions.
"""

from typing import Any

import anyio
import pytest
from inline_snapshot import snapshot
from mcp_types import (
    DiscoverResult,
    Implementation,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    NotificationParams,
    RequestParams,
    ServerCapabilities,
)
from mcp_types.version import MODERN_PROTOCOL_VERSIONS

from mcp.client.client import Client
from mcp.server import Server, ServerRequestContext
from mcp.shared.memory import create_client_server_memory_streams
from mcp.shared.message import SessionMessage
from tests.interaction._requirements import requirement

pytestmark = pytest.mark.anyio


@requirement("custom-methods:server-handler:roundtrip")
async def test_vendor_method_request_is_dispatched_to_its_handler_and_returns_its_result() -> None:
    """A request for a vendor-defined method registered via `add_request_handler` reaches the
    handler with validated params, and the handler's return value comes back as the JSON-RPC
    result.

    SDK-defined. The typed client cannot send a vendor-method request (`ClientRequest` is a
    closed union), so the test plays the client's side of the wire by hand against a real
    Server. Reserve this pattern for behaviour the typed API cannot produce.
    """

    class GreetParams(RequestParams):
        name: str

    async def greet(ctx: ServerRequestContext, params: GreetParams) -> dict[str, Any]:
        assert params.name == "interaction-suite"
        return {"greeting": f"hello {params.name}"}

    server = Server("vendor")
    server.add_request_handler("vendor/greet", GreetParams, greet)
    result: dict[str, Any] | None = None

    async with create_client_server_memory_streams() as (client_streams, server_streams):
        client_read, client_write = client_streams
        server_read, server_write = server_streams

        async with anyio.create_task_group() as server_task_group:
            server_task_group.start_soon(server.run, server_read, server_write, server.create_initialization_options())

            with anyio.fail_after(5):
                await client_write.send(
                    SessionMessage(
                        JSONRPCRequest(
                            jsonrpc="2.0",
                            id=0,
                            method="initialize",
                            params={
                                "protocolVersion": "2025-11-25",
                                "capabilities": {},
                                "clientInfo": {"name": "raw", "version": "0.0.1"},
                            },
                        )
                    )
                )
                init_response = await client_read.receive()
                assert isinstance(init_response, SessionMessage)
                assert isinstance(init_response.message, JSONRPCResponse)
                await client_write.send(
                    SessionMessage(JSONRPCNotification(jsonrpc="2.0", method="notifications/initialized"))
                )

                await client_write.send(
                    SessionMessage(
                        JSONRPCRequest(jsonrpc="2.0", id=1, method="vendor/greet", params={"name": "interaction-suite"})
                    )
                )
                greet_response = await client_read.receive()
                assert isinstance(greet_response, SessionMessage)
                assert isinstance(greet_response.message, JSONRPCResponse)  # a result, not METHOD_NOT_FOUND
                result = greet_response.message.result

            server_task_group.cancel_scope.cancel()

    assert result == snapshot({"greeting": "hello interaction-suite"})


@requirement("custom-methods:notification-handler")
async def test_vendor_notification_is_delivered_to_its_registered_handler() -> None:
    """A client-sent notification for a vendor-defined method reaches the handler registered via
    `add_notification_handler`, with params validated against the registered model.

    SDK-defined. The typed client cannot send a vendor-method notification (`ClientNotification`
    is a closed union), so the test plays the client's side of the wire by hand against a real
    Server. The notification is not tied to an awaited request, so delivery is synchronised with
    an event.
    """

    class EventParams(NotificationParams):
        detail: str

    received: list[EventParams] = []
    delivered = anyio.Event()

    async def on_event(ctx: ServerRequestContext, params: EventParams) -> None:
        received.append(params)
        delivered.set()

    server = Server("vendor")
    server.add_notification_handler("vendor/event", EventParams, on_event)

    async with create_client_server_memory_streams() as (client_streams, server_streams):
        client_read, client_write = client_streams
        server_read, server_write = server_streams

        async with anyio.create_task_group() as server_task_group:
            server_task_group.start_soon(server.run, server_read, server_write, server.create_initialization_options())

            with anyio.fail_after(5):
                await client_write.send(
                    SessionMessage(
                        JSONRPCRequest(
                            jsonrpc="2.0",
                            id=0,
                            method="initialize",
                            params={
                                "protocolVersion": "2025-11-25",
                                "capabilities": {},
                                "clientInfo": {"name": "raw", "version": "0.0.1"},
                            },
                        )
                    )
                )
                init_response = await client_read.receive()
                assert isinstance(init_response, SessionMessage)
                assert isinstance(init_response.message, JSONRPCResponse)
                await client_write.send(
                    SessionMessage(JSONRPCNotification(jsonrpc="2.0", method="notifications/initialized"))
                )

                await client_write.send(
                    SessionMessage(
                        JSONRPCNotification(jsonrpc="2.0", method="vendor/event", params={"detail": "reindex"})
                    )
                )
                await delivered.wait()

            server_task_group.cancel_scope.cancel()

    assert received == [EventParams(detail="reindex")]


@requirement("protocol:request-handler:override-builtin")
async def test_add_request_handler_replaces_the_built_in_discover_handler() -> None:
    """`add_request_handler` for a spec method with a built-in handler replaces it wholesale:
    the client adopts the user-supplied `server/discover` result, not the auto-derived one.

    SDK-defined. The reserved-method arm (`initialize` raises ValueError at registration) is
    pinned by tests/server/test_runner.py.
    """

    async def discover(ctx: ServerRequestContext, params: RequestParams | None) -> DiscoverResult:
        assert ctx.method == "server/discover"
        return DiscoverResult(
            supported_versions=list(MODERN_PROTOCOL_VERSIONS),
            capabilities=ServerCapabilities(),
            server_info=Implementation(name="overridden", version="9.9.9"),
        )

    server = Server("real-name", version="0.0.1")
    server.add_request_handler("server/discover", RequestParams, discover)

    async with Client(server) as client:
        assert client.server_info == snapshot(Implementation(name="overridden", version="9.9.9"))
