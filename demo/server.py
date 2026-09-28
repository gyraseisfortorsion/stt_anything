"""Run with uv run python -m demo.server; binds only to localhost."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

from .service import MAX_UPLOAD, STATIC, APIError, DemoApp

ASSETS = {
    "/": ("index.html", "text/html"),
    "/index.html": ("index.html", "text/html"),
    "/styles.css": ("styles.css", "text/css"),
    "/app.js": ("app.js", "text/javascript"),
    "/model.mjs": ("model.mjs", "text/javascript"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
}


class DemoServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], app: DemoApp) -> None:
        self.app = app
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    @property
    def app(self) -> DemoApp:
        return cast(DemoServer, self.server).app

    def send(
        self,
        status: int,
        value: dict[str, Any] | list[Any] | bytes,
        content_type: str = "application/json",
    ) -> None:
        body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header(
            "Content-Type",
            content_type + ("; charset=utf-8" if content_type != "audio/wav" else ""),
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def handle_request(self, method: str) -> None:
        port = cast(DemoServer, self.server).server_port
        allowed = {f"localhost:{port}", f"127.0.0.1:{port}"}
        try:
            if self.headers.get("Host") not in allowed:
                raise APIError("Use the localhost address printed by the demo server.", 403)
            origin = self.headers.get("Origin")
            if origin and (
                urlsplit(origin).netloc not in allowed or urlsplit(origin).scheme != "http"
            ):
                raise APIError("Requests must come from this local demo.", 403)
            route = urlsplit(self.path)
            path = route.path
            if method == "GET":
                if path in ASSETS:
                    file, mime = ASSETS[path]
                    self.send(200, (STATIC / file).read_bytes(), mime)
                elif path == "/api/config":
                    self.send(200, self.app.config())
                elif path == "/api/indexes":
                    self.send(200, self.app.indexes())
                elif path.startswith("/api/jobs/"):
                    self.send(200, self.app.job(path.removeprefix("/api/jobs/")))
                elif path.startswith("/api/audio/"):
                    self.send(200, self.app.audio(path.removeprefix("/api/audio/")), "audio/wav")
                else:
                    raise APIError("Not found", 404)
            else:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_UPLOAD:
                    raise APIError("Request must contain at most 32 MB.", 413)
                body = self.rfile.read(length)
                if path == "/api/indexes":
                    payload = json.loads(body)
                    if not isinstance(payload, dict):
                        raise APIError("Expected a JSON object.")
                    self.send(202, self.app.create_index(payload))
                elif path == "/api/runs":
                    query = parse_qs(route.query)
                    self.send(202, self.app.analyze(query.get("index_id", [""])[0], body))
                else:
                    raise APIError("Not found", 404)
        except (ValueError, TypeError, KeyError, OSError) as error:
            self.send(error.status if isinstance(error, APIError) else 400, {"error": str(error)})

    def do_GET(self) -> None:
        self.handle_request("GET")

    def do_POST(self) -> None:
        self.handle_request("POST")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local keyword spotting web demo")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    app = DemoApp(args.data_dir)
    try:
        with DemoServer(("127.0.0.1", args.port), app) as server:
            print(f"Keyword lab → http://localhost:{server.server_port}", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    finally:
        app.close()


if __name__ == "__main__":
    main()
