from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI

from ..config import resolve_llm_runtime_config
from ..schemas import RunMode
from ..security.tool_permissions import (
    DISALLOWED_SDK_BUILTIN_TOOLS,
    ToolPermissionDeniedError,
    ToolPermissionGate,
    redact_secrets,
)

ScienceToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass
class AgentTurnResult:
    ok: bool
    provider: str
    public_summary: str
    tool_traces: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


class SDKAdapter:
    """
    Controlled Agent SDK Adapter (L3 §12 / ADR-V2-06).
    - Primary LIVE engine: `claude-agent-sdk==0.2.189` (`ClaudeSDKClient`) with:
      * Custom in-process MCP server (`create_sdk_mcp_server("science", ...)`)
      * `tools=[]` and `disallowed_tools=DISALLOWED_SDK_BUILTIN_TOOLS` so built-in Bash/Write/Read/Task/Agent tools cannot bypass `ResourceScheduler` or `ToolPermissionGate`
      * Explicit `can_use_tool` permission callback backed by `ToolPermissionGate`
    - Secondary LIVE engine: OpenAI-compatible function-calling loop (`AsyncOpenAI`, e.g. user-configured DeepSeek/OpenAI key) enforcing the exact same `ToolPermissionGate` & science handlers when Claude CLI/Anthropic auth is unavailable on a reviewer's machine.
    - Simulation engine: deterministic offline turn (ONLY allowed when `mode == "simulation"`).
    Never exposes private `ThinkingBlock` / reasoning text.
    """

    def __init__(self, workspace_root: Path) -> None:
        self.gate = ToolPermissionGate(workspace_root)
        self.workspace_root = self.gate.allowed_root

    def detect_live_capability(self) -> tuple[bool, str]:
        cfg = resolve_llm_runtime_config()
        forced = cfg["forced_provider"]
        if forced == "simulation":
            return False, "simulation"
        if forced == "openai" and cfg["has_openai_compatible"]:
            return True, f"openai_tool_harness:{cfg['openai_model']}"
        if forced == "claude" and (cfg["has_claude_cli"] or cfg["has_anthropic_key"]):
            return True, "claude_agent_sdk:0.2.189"
        if cfg["has_claude_cli"] or cfg["has_anthropic_key"]:
            return True, "claude_agent_sdk:0.2.189"
        if cfg["has_openai_compatible"]:
            return True, f"openai_tool_harness:{cfg['openai_model']}"
        return False, "none"

    async def run_controlled_science_turn(
        self,
        *,
        mode: RunMode,
        role: str,
        system_prompt: str,
        user_prompt: str,
        tools_spec: list[dict[str, Any]],
        handlers: dict[str, ScienceToolHandler],
        max_turns: int = 4,
    ) -> AgentTurnResult:
        if mode == "simulation":
            return await self._run_simulation_turn(role=role, handlers=handlers, tools_spec=tools_spec)

        cfg = resolve_llm_runtime_config()
        forced = cfg["forced_provider"]
        errors: list[str] = []

        try_claude = forced in ("", "claude") and (cfg["has_claude_cli"] or cfg["has_anthropic_key"])
        if try_claude:
            try:
                res = await asyncio.wait_for(
                    self._run_claude_sdk_turn(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        tools_spec=tools_spec,
                        handlers=handlers,
                        max_turns=max_turns,
                    ),
                    timeout=75.0,
                )
                if res.ok:
                    return res
                errors.append(res.error or "ClaudeSDKClient turn returned non-ok")
            except Exception as exc:
                errors.append(f"ClaudeSDKClient error: {type(exc).__name__}: {exc}")

        if forced in ("", "openai") and cfg["has_openai_compatible"]:
            try:
                res_oa = await asyncio.wait_for(
                    self._run_openai_tool_turn(
                        cfg=cfg,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        tools_spec=tools_spec,
                        handlers=handlers,
                        max_turns=max_turns,
                    ),
                    timeout=75.0,
                )
                if res_oa.ok:
                    return res_oa
                errors.append(res_oa.error or "OpenAI tool turn returned non-ok")
            except Exception as exc:
                errors.append(f"OpenAI tool harness error: {type(exc).__name__}: {exc}")

        return AgentTurnResult(
            ok=False,
            provider="blocked",
            public_summary="",
            tool_traces=[],
            error=str(redact_secrets(" | ".join(errors) or "No live LLM/SDK credentials available.")),
        )

    async def _run_claude_sdk_turn(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tools_spec: list[dict[str, Any]],
        handlers: dict[str, ScienceToolHandler],
        max_turns: int,
    ) -> AgentTurnResult:
        import claude_agent_sdk as sdk

        traces: list[dict[str, Any]] = []
        sdk_tools = []
        allowed_mcp_names: list[str] = []

        for spec in tools_spec:
            t_name = str(spec["name"])
            t_desc = str(spec["description"])
            t_schema = dict(spec.get("parameters") or {"type": "object", "properties": {}})
            mcp_full_name = f"mcp__science__{t_name}"
            allowed_mcp_names.append(mcp_full_name)
            handler_fn = handlers[t_name]

            def _make_tool(bound_name: str, bound_desc: str, bound_schema: dict[str, Any], bound_fn: ScienceToolHandler):
                @sdk.tool(bound_name, bound_desc, bound_schema)
                async def _wrapped(args: dict[str, Any]) -> dict[str, Any]:
                    self.gate.authorize_tool_call(bound_name, args)
                    result = await bound_fn(args)
                    safe_res = redact_secrets(result)
                    traces.append(
                        {
                            "tool": bound_name,
                            "provider": "claude-agent-sdk==0.2.189",
                            "args": redact_secrets(args),
                            "status": "ok",
                            "result_keys": sorted(result.keys()) if isinstance(result, dict) else [],
                        }
                    )
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(safe_res, ensure_ascii=False),
                            }
                        ]
                    }

                return _wrapped

            sdk_tools.append(_make_tool(t_name, t_desc, t_schema, handler_fn))

        async def _can_use_tool(
            tool_name: str,
            input_data: dict[str, Any],
            _ctx: Any,
        ) -> Any:
            try:
                self.gate.authorize_tool_call(tool_name, input_data)
                return sdk.PermissionResultAllow(updated_input=input_data)
            except ToolPermissionDeniedError as exc:
                return sdk.PermissionResultDeny(message=str(exc), interrupt=True)

        mcp_server = sdk.create_sdk_mcp_server(name="science", version="1.0.0", tools=sdk_tools)
        sdk_cwd = self.workspace_root / "sdk_cwd"
        sdk_cwd.mkdir(parents=True, exist_ok=True)

        opts = sdk.ClaudeAgentOptions(
            tools=[],
            mcp_servers={"science": mcp_server},
            allowed_tools=allowed_mcp_names,
            disallowed_tools=list(DISALLOWED_SDK_BUILTIN_TOOLS),
            can_use_tool=_can_use_tool,
            max_turns=max_turns,
            cwd=str(sdk_cwd),
            system_prompt=(
                f"{system_prompt}\n"
                "Provide concise, factual scientific summaries suitable for public research logs. "
                "Never include private chain-of-thought."
            ),
        )

        text_chunks: list[str] = []
        async with sdk.ClaudeSDKClient(options=opts) as client:
            await client.query(user_prompt)
            async for msg in client.receive_response():
                if isinstance(msg, sdk.AssistantMessage):
                    for block in getattr(msg, "content", []) or []:
                        # Extract TextBlock only; explicitly ignore ThinkingBlock (no private CoT!)
                        if isinstance(block, sdk.TextBlock) and getattr(block, "text", ""):
                            text_chunks.append(str(block.text).strip())

        summary_txt = "\n".join(c for c in text_chunks if c).strip()
        if not summary_txt and traces:
            summary_txt = f"Executed {len(traces)} controlled scientific tool call(s) via ClaudeSDKClient."
        return AgentTurnResult(
            ok=True,
            provider="claude-agent-sdk==0.2.189",
            public_summary=str(redact_secrets(summary_txt))[:1600],
            tool_traces=traces,
        )

    async def _run_openai_tool_turn(
        self,
        *,
        cfg: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        tools_spec: list[dict[str, Any]],
        handlers: dict[str, ScienceToolHandler],
        max_turns: int,
    ) -> AgentTurnResult:
        client = AsyncOpenAI(
            api_key=cfg["openai_api_key"],
            base_url=cfg["openai_base_url"],
            timeout=60.0,
        )
        oa_tools = [
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s["description"],
                    "parameters": s.get("parameters") or {"type": "object", "properties": {}},
                },
            }
            for s in tools_spec
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        traces: list[dict[str, Any]] = []
        final_text = ""

        for _ in range(max_turns):
            resp = await client.chat.completions.create(
                model=cfg["openai_model"],
                messages=messages,
                tools=oa_tools if oa_tools else None,
                max_tokens=1024,
            )
            choice = resp.choices[0].message
            tool_calls = getattr(choice, "tool_calls", None) or []
            if not tool_calls:
                final_text = (choice.content or "").strip()
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": choice.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )
            for tc in tool_calls:
                fn_name = tc.function.name
                try:
                    parsed_args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    parsed_args = {}
                self.gate.authorize_tool_call(fn_name, parsed_args)
                handler = handlers[fn_name]
                result = await handler(parsed_args)
                safe_res = redact_secrets(result)
                traces.append(
                    {
                        "tool": fn_name,
                        "provider": f"openai_compatible:{cfg['openai_model']}",
                        "args": redact_secrets(parsed_args),
                        "status": "ok",
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(safe_res, ensure_ascii=False),
                    }
                )

        if not final_text and traces:
            final_text = f"Executed {len(traces)} controlled scientific tool call(s) via {cfg['openai_model']}."
        return AgentTurnResult(
            ok=True,
            provider=f"openai_compatible:{cfg['openai_model']}",
            public_summary=str(redact_secrets(final_text))[:1600],
            tool_traces=traces,
        )

    async def _run_simulation_turn(
        self,
        *,
        role: str,
        handlers: dict[str, ScienceToolHandler],
        tools_spec: list[dict[str, Any]],
    ) -> AgentTurnResult:
        traces: list[dict[str, Any]] = []
        for spec in tools_spec[:1]:
            t_name = str(spec["name"])
            if t_name in handlers:
                self.gate.authorize_tool_call(t_name, {"simulation": True})
                await handlers[t_name]({"simulation": True})
                traces.append({"tool": t_name, "provider": "simulation_mock", "status": "ok"})
        return AgentTurnResult(
            ok=True,
            provider="simulation_mock",
            public_summary=f"[SIMULATION] {role.capitalize()} completed protocol check.",
            tool_traces=traces,
        )
