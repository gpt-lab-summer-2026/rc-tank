#!/usr/bin/env python3
"""
Live MJPEG view of whatever a control loop is looking at.

    from stream import MJPEGStreamer

    view = MJPEGStreamer(port=8080)
    ...
    view.update(annotated_frame)      # as often as you like

Open http://<pi>:8080/ in a browser. If the network will not route
that, tunnel it instead and use localhost:

    ssh -N -L 8080:localhost:8080 pi@rover

-------------------------------------------------------------
IT MUST NEVER BLOCK THE CALLER

roam.py stops refreshing the bridge if its loop stalls for more than
command_timeout, and the firmware watchdog then releases the relays.
A viewer on a slow phone over patchy WiFi must not be able to cause
that, so nothing here waits on a socket while holding anything the
control loop needs.

update() drops the frame into a single slot and returns. Clients read
whatever is in the slot when they get there. A slow client misses
frames; it cannot slow the tank down. That is the whole design.

Encoding is done on the CALLER's thread, deliberately — it is a few
milliseconds and doing it here would mean either a queue that can
back up or a lock the loop can wait on.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

_PAGE = b"""<!doctype html><meta charset=utf-8>
<title>roam</title>
<style>
 html,body{margin:0;background:#111;color:#ccc;font:14px system-ui;height:100%}
 body{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px}
 img{max-width:100vw;max-height:90vh;image-rendering:pixelated}
</style>
<img src="/stream.mjpg" alt="live view">
<div>green = floor &middot; red = where each column stops &middot; cyan = reference patch</div>
"""


class _Handler(BaseHTTPRequestHandler):
    streamer: "MJPEGStreamer" = None      # set by the server

    def log_message(self, *a):
        pass                              # never scribble on the status line

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(_PAGE)))
            self.end_headers()
            self.wfile.write(_PAGE)
            return

        if self.path != "/stream.mjpg":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()

        last = -1
        try:
            while True:
                jpeg, seq = self.streamer.wait_for_frame(last)
                if jpeg is None:          # streamer closing
                    return
                last = seq
                self.wfile.write(b"--FRAME\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass                          # viewer closed the tab, fine


class MJPEGStreamer:
    """One latest-frame slot, served to any number of browsers."""

    def __init__(self, port: int = 8080, quality: int = 70):
        self.quality = quality
        self._jpeg = None
        self._seq = 0
        self._closing = False
        self._cv = threading.Condition()

        handler = type("Handler", (_Handler,), {"streamer": self})
        self._server = ThreadingHTTPServer(("", port), handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="mjpeg", daemon=True)
        self._thread.start()
        self.port = self._server.server_address[1]

    def update(self, bgr) -> None:
        """Publish a frame. Returns as soon as it is encoded."""
        ok, buf = cv2.imencode(".jpg", bgr,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            return
        with self._cv:
            self._jpeg = buf.tobytes()
            self._seq += 1
            self._cv.notify_all()

    def wait_for_frame(self, since: int):
        """Block THIS client until a frame newer than `since` exists."""
        with self._cv:
            while self._seq <= since and not self._closing:
                self._cv.wait(timeout=1.0)
            if self._closing:
                return None, since
            return self._jpeg, self._seq

    def close(self) -> None:
        with self._cv:
            self._closing = True
            self._cv.notify_all()
        self._server.shutdown()
        self._server.server_close()
