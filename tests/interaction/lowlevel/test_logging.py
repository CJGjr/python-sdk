"""Logging interactions against the low-level Server, driven through the public Client API.

Notification ordering: await-free callbacks finish in arrival order, and passing
``related_request_id`` keeps each notification on the originating request's POST stream over
streamable HTTP, so plain-list collection is deterministic on every transport leg.
"""

import mcp_types as types
import pytest
from inline_snapshot import snapshot
from mcp_types import (
    INVALID_PARAMS,
    LOG_LEVEL_META_KEY,
    CallToolResult,
    EmptyResult,
    ErrorData,
    LoggingMessageNotificationParams,
    TextContent,
)

from mcp import MCPError
from mcp.server import Server, ServerRequestContext
from tests.interaction._connect import Connect
from tests.interaction._requirements import requirement

pytestmark = pytest.mark.anyio

ALL_LEVELS: tuple[types.LoggingLevel, ...] = (
    "debug",
    "info",
    "notice",
    "warning",
    "error",
    "critical",
    "alert",
    "emergency",
)


@requirement("logging:set-level")
async def test_set_logging_level_reaches_handler(connect: Connect) -> None:
    """The level requested by the client is delivered to the server's handler verbatim."""

    async def set_logging_level(ctx: ServerRequestContext, params: types.SetLevelRequestParams) -> EmptyResult:
        assert params.level == "warning"
        return EmptyResult()

    server = Server("logger", on_set_logging_level=set_logging_level)  # pyright: ignore[reportDeprecated]

    async with connect(server) as client:
        result = await client.set_logging_level("warning")  # pyright: ignore[reportDeprecated]

    assert result == snapshot(EmptyResult())


@requirement("logging:message:fields")
@requirement("tools:call:logging-mid-execution")
async def test_log_messages_reach_logging_callback_in_order(connect: Connect) -> None:
    """Log messages sent during a tool call arrive at the logging callback, in order, before the call returns.

    The two messages pin the full notification shape: severity, optional logger name, and both
    string and structured data payloads.
    """
    received: list[LoggingMessageNotificationParams] = []

    async def collect(params: LoggingMessageNotificationParams) -> None:
        received.append(params)

    async def list_tools(
        ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[types.Tool(name="chatty", input_schema={"type": "object"})])

    async def call_tool(ctx: ServerRequestContext, params: types.CallToolRequestParams) -> CallToolResult:
        assert params.name == "chatty"
        await ctx.session.send_log_message(  # pyright: ignore[reportDeprecated]
            level="info", data="starting up", logger="app.lifecycle", related_request_id=ctx.request_id
        )
        await ctx.session.send_log_message(  # pyright: ignore[reportDeprecated]
            level="error", data={"code": 502, "retryable": True}, related_request_id=ctx.request_id
        )
        return CallToolResult(content=[TextContent(text="done")])

    async def set_logging_level(ctx: ServerRequestContext, params: types.SetLevelRequestParams) -> EmptyResult:
        """Registered so the logging capability is advertised; the client never sets a level."""
        raise NotImplementedError

    server = Server(  # pyright: ignore[reportDeprecated]
        "logger", on_list_tools=list_tools, on_call_tool=call_tool, on_set_logging_level=set_logging_level
    )

    async with connect(server, logging_callback=collect) as client:
        result = await client.call_tool("chatty", {})
        # Captured at return time: both messages were delivered before call_tool returned.
        received_at_return = list(received)

    assert result == snapshot(CallToolResult(content=[TextContent(text="done")]))
    assert received_at_return == snapshot(
        [
            LoggingMessageNotificationParams(level="info", logger="app.lifecycle", data="starting up"),
            LoggingMessageNotificationParams(level="error", data={"code": 502, "retryable": True}),
        ]
    )
    assert received == received_at_return  # nothing arrived after the call returned


@requirement("logging:message:all-levels")
async def test_log_messages_at_every_severity_level(connect: Connect) -> None:
    """Each of the eight RFC 5424 severity levels is deliverable as a log message notification."""
    received: list[LoggingMessageNotificationParams] = []

    async def collect(params: LoggingMessageNotificationParams) -> None:
        received.append(params)

    async def list_tools(
        ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[types.Tool(name="siren", input_schema={"type": "object"})])

    async def call_tool(ctx: ServerRequestContext, params: types.CallToolRequestParams) -> CallToolResult:
        assert params.name == "siren"
        for level in ALL_LEVELS:
            await ctx.session.send_log_message(  # pyright: ignore[reportDeprecated]
                level=level, data=f"a {level} message", related_request_id=ctx.request_id
            )
        return CallToolResult(content=[TextContent(text="logged")])

    async def set_logging_level(ctx: ServerRequestContext, params: types.SetLevelRequestParams) -> EmptyResult:
        """Registered so the logging capability is advertised; the client never sets a level."""
        raise NotImplementedError

    server = Server(  # pyright: ignore[reportDeprecated]
        "logger", on_list_tools=list_tools, on_call_tool=call_tool, on_set_logging_level=set_logging_level
    )

    async with connect(server, logging_callback=collect) as client:
        await client.call_tool("siren", {})

    assert [params.level for params in received] == list(ALL_LEVELS)


@requirement("logging:message:filtered")
async def test_lowlevel_server_delivers_messages_below_the_requested_level(connect: Connect) -> None:
    """After logging/setLevel("error"), a debug message still reaches the client's callback.

    Pinned divergence (the low-level half of the note on `logging:message:filtered`): the
    low-level Server leaves filtering entirely to the author's setLevel handler, so an
    acknowledge-only handler filters nothing and every severity is delivered.
    """
    received: list[LoggingMessageNotificationParams] = []
    levels: list[types.LoggingLevel] = []

    async def collect(params: LoggingMessageNotificationParams) -> None:
        received.append(params)

    async def set_logging_level(ctx: ServerRequestContext, params: types.SetLevelRequestParams) -> EmptyResult:
        levels.append(params.level)
        return EmptyResult()

    async def list_tools(
        ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[types.Tool(name="chatty", input_schema={"type": "object"})])

    async def call_tool(ctx: ServerRequestContext, params: types.CallToolRequestParams) -> CallToolResult:
        assert params.name == "chatty"
        await ctx.session.send_log_message(  # pyright: ignore[reportDeprecated]
            level="debug", data="below the level", related_request_id=ctx.request_id
        )
        await ctx.session.send_log_message(  # pyright: ignore[reportDeprecated]
            level="error", data="at the level", related_request_id=ctx.request_id
        )
        return CallToolResult(content=[TextContent(text="done")])

    server = Server(  # pyright: ignore[reportDeprecated]
        "logger", on_list_tools=list_tools, on_call_tool=call_tool, on_set_logging_level=set_logging_level
    )

    async with connect(server, logging_callback=collect) as client:
        await client.set_logging_level("error")  # pyright: ignore[reportDeprecated]
        await client.call_tool("chatty", {})

    # Dispatch identity: the level reached the author's handler before the tool ran.
    assert levels == ["error"]
    assert received == snapshot(
        [
            LoggingMessageNotificationParams(level="debug", data="below the level"),
            LoggingMessageNotificationParams(level="error", data="at the level"),
        ]
    )


@requirement("logging:per-request-level:invalid-level")
async def test_unrecognized_per_request_log_level_is_rejected_with_invalid_params(connect: Connect) -> None:
    """A request whose _meta carries an unrecognized io.modelcontextprotocol/logLevel fails with -32602.

    Spec-mandated (2026-07-28 logging error handling, a SHOULD the SDK honours). The rejection is
    the per-version surface validation and fires before the tool handler; the same call with a
    recognized level succeeds, proving the level value is what was rejected. The error message is
    the SDK's own surface-validation output.
    """

    async def list_tools(
        ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[types.Tool(name="echo", input_schema={"type": "object"})])

    async def call_tool(ctx: ServerRequestContext, params: types.CallToolRequestParams) -> CallToolResult:
        assert params.name == "echo"
        return CallToolResult(content=[TextContent(text="ok")])

    server = Server("strict", on_list_tools=list_tools, on_call_tool=call_tool)

    async with connect(server) as client:
        accepted = await client.call_tool("echo", {}, meta={LOG_LEVEL_META_KEY: "warning"})
        with pytest.raises(MCPError) as exc_info:
            await client.call_tool("echo", {}, meta={LOG_LEVEL_META_KEY: "verbose"})

    assert accepted == snapshot(CallToolResult(content=[TextContent(text="ok")]))
    assert exc_info.value.error == snapshot(
        ErrorData(code=INVALID_PARAMS, message="Invalid request parameters", data="")
    )
