from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from streamlit.testing.v1 import AppTest


def test_streamlit_coding_approval_workflow(monkeypatch) -> None:
    requests_seen: list[tuple[str, str]] = []
    completed = False

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def _send_json(self, value: dict) -> None:
            payload = json.dumps(value).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            if self.path.startswith("/conversations?"):
                self._send_json(
                    {
                        "conversations": [
                            {
                                "id": "session-1",
                                "updated_at": "2026-11-11T09:00:00",
                            }
                        ]
                    }
                )
                return
            if self.path == "/conversations/session-1":
                messages = [{"role": "user", "content": "Create file"}]
                if completed:
                    messages.append(
                        {"role": "assistant", "content": "The file was written."}
                    )
                self._send_json(
                    {
                        "conversation": {"id": "session-1"},
                        "messages": messages,
                    }
                )
                return
            self.send_error(404)

        def do_POST(self) -> None:
            nonlocal completed
            requests_seen.append(("POST", self.path))
            if self.path == "/coding/turn":
                self._send_json(
                    {
                        "status": "approval_required",
                        "transcript": "Create file",
                        "response": "Review this exact change.",
                        "conversation_id": "session-1",
                        "pending_tool": {
                            "pending_id": "pending-1",
                            "tool_name": "write_file",
                            "arguments": {
                                "path": "notes.txt",
                                "content": "local note",
                            },
                        },
                    }
                )
                return
            if self.path == "/coding/approval/prepare":
                self._send_json(
                    {
                        "request_id": "request-1",
                        "tool_name": "write_file",
                        "arguments": {
                            "path": "notes.txt",
                            "content": "local note",
                            "overwrite": False,
                        },
                        "expires_in_seconds": 120,
                    }
                )
                return
            if self.path == "/coding/approval/confirm":
                self._send_json({"approval_token": "token-1"})
                return
            if self.path == "/coding/approval/resume":
                completed = True
                self._send_json(
                    {
                        "status": "completed",
                        "response": "The file was written.",
                        "pending_tool": None,
                        "tool_calls_executed": 1,
                    }
                )
                return
            self.send_error(404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(
        "APP_API_URL", f"http://127.0.0.1:{server.server_address[1]}"
    )
    try:
        page = AppTest.from_file("app/ui/streamlit_app.py").run(timeout=15)
        page.radio[0].set_value("Coding assistant").run(timeout=15)
        page.button[0].click().run(timeout=15)
        assert page.session_state["pending_id"] == "pending-1"

        page.button[0].click().run(timeout=15)
        assert page.session_state["approval_request"]["request_id"] == "request-1"
        assert any("local note" in str(item.value) for item in page.json)

        page.checkbox[0].set_value(True).run(timeout=15)
        page.button[0].click().run(timeout=15)

        assert page.session_state["pending_id"] is None
        assert requests_seen == [
            ("POST", "/coding/turn"),
            ("POST", "/coding/approval/prepare"),
            ("POST", "/coding/approval/confirm"),
            ("POST", "/coding/approval/resume"),
        ]
        assert not page.exception
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
