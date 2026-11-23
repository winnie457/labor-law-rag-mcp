"""Minimal MCP client to test query_knowledge_hub over stdio.

Usage:
    python scripts/test_mcp_query.py "竞业限制的期限最长是多久？"
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _start_server() -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PROJECT_ROOT),
        env=env,
    )


def _send_jsonrpc(
    proc: subprocess.Popen,
    messages: List[Dict[str, Any]],
    expected_responses: int,
    timeout: float = 120.0,
) -> List[Dict[str, Any]]:
    assert proc.stdin is not None
    assert proc.stdout is not None

    for msg in messages:
        proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()

    responses: List[Dict[str, Any]] = []
    stop_event = threading.Event()

    def _reader() -> None:
        while not stop_event.is_set():
            line = proc.stdout.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if "id" in data and ("result" in data or "error" in data):
                responses.append(data)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    deadline = time.time() + timeout
    while len(responses) < expected_responses and time.time() < deadline:
        time.sleep(0.1)

    stop_event.set()
    return responses


def _find(responses: List[Dict[str, Any]], req_id: int) -> Dict[str, Any] | None:
    for r in responses:
        if r.get("id") == req_id:
            return r
    return None


def main() -> int:
    query = sys.argv[1] if len(sys.argv) > 1 else "竞业限制的期限最长是多久？"
    proc = _start_server()
    try:
        messages = [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "clientInfo": {"name": "test-client", "version": "1.0"},
                    "capabilities": {},
                },
            },
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "query_knowledge_hub",
                    "arguments": {"query": query, "top_k": 5},
                },
            },
        ]

        responses = _send_jsonrpc(proc, messages, expected_responses=2, timeout=120.0)

        init_resp = _find(responses, 1)
        if init_resp is None:
            print("[FAIL] No initialize response")
            return 1

        tool_resp = _find(responses, 2)
        if tool_resp is None:
            print("[FAIL] No tool call response")
            return 1

        result = tool_resp.get("result", {})
        if result.get("isError"):
            print(f"[FAIL] Tool error: {result.get('content', [{}])[0].get('text', '')}")
            return 1

        content = result.get("content", [])
        for block in content:
            if block.get("type") == "text":
                print(block["text"])
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    sys.exit(main())
