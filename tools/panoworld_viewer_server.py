#!/usr/bin/env python3
"""Serve a lightweight PanoWorld panorama viewer for a viewpoints directory."""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import posixpath
import socket
import subprocess
import sys
from typing import Any
from urllib.parse import quote, unquote, urlparse


DEFAULT_IMAGE_CANDIDATES = (
    "panoImage_2048.png",
    "panoImage_2048_franch.png",
    "panoImage_2048_lrm.png",
)
ASSET_DIR = Path(__file__).resolve().parent / "viewer_assets"


def sort_view_key(path: Path) -> tuple[int, Any]:
    name = path.name
    if name.isdigit():
        return (0, int(name))
    return (1, name)


def media_url(view_name: str, filename: str) -> str:
    return "/media/" + quote(view_name) + "/" + quote(filename)


def read_extrinsics(view_dir: Path) -> dict[str, Any] | None:
    extrinsics_path = view_dir / "extrinsics.txt"
    if not extrinsics_path.is_file():
        return None

    rows = []
    with extrinsics_path.open("r", encoding="utf-8") as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            if values:
                rows.append(values)

    if len(rows) < 3 or any(len(row) < 4 for row in rows[:3]):
        return None

    return {
        "position": [rows[0][3], rows[1][3], rows[2][3]],
        "rotation": [[rows[r][c] for c in range(3)] for r in range(3)],
    }


def first_existing_image(view_dir: Path, image_name: str, start_image_name: str) -> Path | None:
    candidates = [
        image_name,
        start_image_name,
        "panoImage_2048_8k_flux.png",
        "panoImage_2048_lrm.png",
        "panoImage_2048_lrm_mask.png",
    ]
    seen = set()
    for name in candidates:
        if not name or name in seen:
            continue
        seen.add(name)
        path = view_dir / name
        if path.is_file():
            return path

    for name in DEFAULT_IMAGE_CANDIDATES:
        path = view_dir / name
        if path.is_file():
            return path
    return None


def build_manifest(viewpoints_dir: Path, image_name: str, start_image_name: str) -> dict[str, Any]:
    viewpoints_dir = viewpoints_dir.resolve()
    if not viewpoints_dir.is_dir():
        raise FileNotFoundError(f"Viewpoints directory not found: {viewpoints_dir}")

    views = []
    for view_dir in sorted((p for p in viewpoints_dir.iterdir() if p.is_dir()), key=sort_view_key):
        image_path = first_existing_image(view_dir, image_name, start_image_name)
        if image_path is None:
            continue

        view = {
            "id": view_dir.name,
            "image": media_url(view_dir.name, image_path.name),
            "imageName": image_path.name,
        }
        pose = read_extrinsics(view_dir)
        if pose is not None:
            view["pose"] = pose
        views.append(view)

    if not views:
        raise FileNotFoundError(
            f"No panorama images found under {viewpoints_dir}. "
            f"Tried {image_name!r}, {start_image_name!r}, and common PanoWorld names."
        )

    return {
        "title": "PanoWorld Interactive Panorama Tour",
        "viewpointsDir": str(viewpoints_dir),
        "views": views,
    }


def html_page(manifest: dict[str, Any]) -> bytes:
    manifest_json = json.dumps(manifest, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(str(manifest["title"]))
    first_image = html.escape(str(manifest["views"][0]["image"]))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="PanoWorld panorama tour viewer.">
  <title>{title}</title>
  <link rel="preload" as="image" href="{first_image}">
  <link rel="modulepreload" href="/static/vendor/three.module.min.js">
  <link rel="stylesheet" href="/static/panoworld_viewer.css">
  <script>
    window.PANOWORLD_MANIFEST = {manifest_json};
  </script>
  <script type="module" src="/static/panoworld_viewer.js"></script>
</head>
<body>
  <main id="pano-tour-stage" class="pano-tour-stage">
    <div id="pano-tour-loading" class="pano-tour-loading">
      <span class="pano-tour-spinner" aria-hidden="true"></span>
      <span>Loading panorama tour...</span>
    </div>
    <div id="pano-tour-hotspots" class="pano-tour-hotspots" aria-hidden="false"></div>
    <div class="pano-tour-hud">
      <h1 class="pano-tour-title">{title}</h1>
      <span id="pano-tour-current" class="pano-tour-badge">Viewpoint --</span>
    </div>
    <div class="pano-tour-stage-actions">
      <button id="pano-tour-prev" class="pano-tour-control" type="button" title="Previous viewpoint" aria-label="Previous viewpoint">‹</button>
      <button id="pano-tour-reset" class="pano-tour-control" type="button" title="Reset view" aria-label="Reset view">⟲</button>
      <button id="pano-tour-zoom-out" class="pano-tour-control" type="button" title="Wider FOV" aria-label="Wider FOV">−</button>
      <span id="pano-tour-fov-label" class="pano-tour-fov-label">FOV --°</span>
      <button id="pano-tour-zoom-in" class="pano-tour-control" type="button" title="Narrower FOV" aria-label="Narrower FOV">+</button>
      <button id="pano-tour-fullscreen" class="pano-tour-control" type="button" title="Fullscreen" aria-label="Fullscreen">⛶</button>
      <button id="pano-tour-next" class="pano-tour-control" type="button" title="Next viewpoint" aria-label="Next viewpoint">›</button>
    </div>
    <div id="pano-tour-strip" class="pano-tour-strip"></div>
    <div class="pano-tour-hint">Drag to look around · Scroll or use FOV controls to zoom · Click a hotspot or viewpoint to move</div>
    <div id="pano-tour-tooltip" class="pano-tour-tooltip" hidden></div>
  </main>
</body>
</html>
""".encode("utf-8")


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "PanoWorldViewer/1.0"

    def do_HEAD(self) -> None:  # noqa: N802
        self.route_request(head_only=True)

    def do_GET(self) -> None:  # noqa: N802
        self.route_request(head_only=False)

    def route_request(self, head_only: bool) -> None:
        parsed = urlparse(self.path)
        route = posixpath.normpath(unquote(parsed.path))
        if route in ("", "/"):
            self.send_bytes(html_page(self.server.manifest), "text/html; charset=utf-8", head_only=head_only)
            return
        if route == "/manifest.json":
            payload = json.dumps(self.server.manifest, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_bytes(payload, "application/json; charset=utf-8", head_only=head_only)
            return
        if route.startswith("/static/"):
            self.serve_static(route[len("/static/") :], head_only=head_only)
            return
        if route.startswith("/media/"):
            self.serve_media(route[len("/media/") :], head_only=head_only)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def serve_static(self, static_path: str, head_only: bool) -> None:
        parts = [part for part in static_path.split("/") if part]
        if not parts or any(part in ("", ".", "..") for part in parts):
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid static path")
            return

        path = (ASSET_DIR / Path(*parts)).resolve()
        try:
            path.relative_to(ASSET_DIR)
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN, "Path escapes static directory")
            return
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Static asset not found")
            return
        self.send_file(path, head_only=head_only, cache_control="no-store")

    def serve_media(self, media_path: str, head_only: bool) -> None:
        parts = [part for part in media_path.split("/") if part]
        if len(parts) != 2:
            self.send_error(HTTPStatus.BAD_REQUEST, "Expected /media/<view>/<filename>")
            return

        view_name, filename = parts
        if "/" in filename or filename in ("", ".", ".."):
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid filename")
            return

        base = self.server.viewpoints_dir
        path = (base / view_name / filename).resolve()
        try:
            path.relative_to(base)
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN, "Path escapes viewpoints directory")
            return
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Media not found")
            return
        self.send_file(path, head_only=head_only, cache_control="no-store")

    def send_file(self, path: Path, head_only: bool, cache_control: str) -> None:
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix == ".js":
            content_type = "text/javascript"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", cache_control)
        self.end_headers()
        if head_only:
            return
        with path.open("rb") as f:
            self.wfile.write(f.read())

    def send_bytes(self, payload: bytes, content_type: str, head_only: bool = False) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head_only:
            self.wfile.write(payload)

    def log_message(self, fmt: str, *args: Any) -> None:
        try:
            sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))
        except BrokenPipeError:
            pass


def add_host_candidate(candidates: list[str], host: str | None) -> None:
    if not host:
        return
    host = host.strip()
    if not host:
        return
    if "://" in host:
        parsed = urlparse(host)
        host = parsed.hostname or host
    host = host.strip("[]")
    if not host or host in ("0.0.0.0", "::"):
        return
    if host not in candidates:
        candidates.append(host)


def guess_route_ip() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return None


def hostname_ips() -> list[str]:
    hosts: list[str] = []
    for name in (socket.gethostname(), socket.getfqdn()):
        add_host_candidate(hosts, name)
        try:
            for addr in socket.getaddrinfo(name, None, family=socket.AF_INET):
                add_host_candidate(hosts, addr[4][0])
        except OSError:
            pass

    try:
        output = subprocess.check_output(
            ["hostname", "-I"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
        for addr in output.split():
            add_host_candidate(hosts, addr)
    except (OSError, subprocess.SubprocessError):
        pass
    return hosts


def access_hosts(bound_host: str, public_host: str | None) -> list[str]:
    hosts: list[str] = []
    add_host_candidate(hosts, public_host)

    if bound_host in ("0.0.0.0", "::", ""):
        add_host_candidate(hosts, guess_route_ip())
        for host in hostname_ips():
            add_host_candidate(hosts, host)
    else:
        add_host_candidate(hosts, bound_host)

    if not hosts:
        add_host_candidate(hosts, "127.0.0.1")
    return hosts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a PanoWorld panorama viewer.")
    parser.add_argument("--viewpoints", required=True, help="Path to a viewpoints directory.")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8003")))
    parser.add_argument("--public-host", default=os.environ.get("PUBLIC_HOST", ""))
    parser.add_argument("--image-name", default=os.environ.get("IMAGE_NAME", "panoImage_2048.png"))
    parser.add_argument("--start-image-name", default=os.environ.get("START_IMAGE_NAME", "panoImage_2048_franch.png"))
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    viewpoints_dir = Path(args.viewpoints).expanduser().resolve()
    manifest = build_manifest(viewpoints_dir, args.image_name, args.start_image_name)

    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    server.viewpoints_dir = viewpoints_dir
    server.manifest = manifest

    print(f"PanoWorld viewer serving {len(manifest['views'])} panoramas from:")
    print(f"  {viewpoints_dir}")
    print("Open one of these URLs from your local browser:")
    for host in access_hosts(args.host, args.public_host):
        print(f"  http://{host}:{args.port}/")
    print("If none of these URLs opens, the server network is blocking inbound access.")
    print(f"In that case, expose port {args.port} through your SSH/cloud platform and open the forwarded URL.")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping viewer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
