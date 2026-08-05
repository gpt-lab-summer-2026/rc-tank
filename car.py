"""
Client for the ESP32 relay bridge.

The ESP32 sets relay states and nothing else. Everything about how
the car drives is decided here.

    from car import Car

    with Car() as car:
        car.forward()
        time.sleep(1)
        car.spin_left()
        time.sleep(0.5)
        car.stop()

Needs pyserial:  pip install pyserial
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import serial
from serial.tools import list_ports

__all__ = ["Car", "BridgeError", "find_port"]


class BridgeError(RuntimeError):
    """The bridge replied with an error, or did not reply at all."""


# USB-serial chips commonly found on ESP32 dev boards.
_LIKELY = ("CP210", "CH340", "CH910", "FTDI", "FT232", "ESP32", "USB Serial")


def find_port() -> Optional[str]:
    """Best guess at which serial device is the ESP32."""
    ports = list(list_ports.comports())
    for p in ports:
        blob = f"{p.description} {p.manufacturer} {p.product}"
        if any(k.lower() in blob.lower() for k in _LIKELY):
            return p.device
    for p in ports:                      # fall back to the usual suspects
        if "ttyUSB" in p.device or "ttyACM" in p.device:
            return p.device
    return None


# Per motor: +1 forward, 0 stop, -1 reverse -> (relay A, relay B)
_DIR = {1: (1, 0), 0: (0, 0), -1: (0, 1)}


class Car:
    """Serial link to the relay bridge.

    Opening the port resets most ESP32 boards, so the constructor
    waits for it to come back before talking to it.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baud: int = 115200,
        timeout: float = 0.3,
        keepalive: bool = True,
        boot_delay: float = 2.0,
    ):
        port = port or find_port()
        if port is None:
            raise BridgeError("no serial port found — pass port= explicitly")

        self.port = port
        self._lock = threading.Lock()
        self._ser = serial.Serial(port, baud, timeout=timeout)

        time.sleep(boot_delay)            # board resets when the port opens
        self._ser.reset_input_buffer()

        self._last_tx = time.monotonic()
        self._watchdog_s = 0.4
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

        info = self.info()
        if "bridge" not in info:
            raise BridgeError(f"unexpected greeting: {info!r}")

        if keepalive:
            self._thread = threading.Thread(
                target=self._keepalive, name="car-keepalive", daemon=True
            )
            self._thread.start()

    # ---------------------------------------------------------- io

    def _send(self, line: str) -> str:
        with self._lock:
            if self._ser is None:
                raise BridgeError("port is closed")
            self._ser.reset_input_buffer()
            self._ser.write((line + "\n").encode())
            self._ser.flush()
            reply = self._ser.readline().decode(errors="replace").strip()
            self._last_tx = time.monotonic()

        if not reply:
            raise BridgeError(f"no reply to {line!r}")
        if reply.startswith("err"):
            raise BridgeError(f"{line!r} -> {reply}")
        return reply

    def _keepalive(self) -> None:
        """Ping when idle, so the firmware watchdog never fires.

        Without this, a slow inference step between commands looks
        to the ESP32 like the Pi has gone away.
        """
        while not self._stop_evt.is_set():
            if self._watchdog_s > 0:
                idle = time.monotonic() - self._last_tx
                if idle > self._watchdog_s / 3:
                    try:
                        self._send("P")
                    except Exception:
                        pass              # close() will surface real failures
            self._stop_evt.wait(0.05)

    # ----------------------------------------------------- commands

    def relays(self, a: int, b: int, c: int, d: int) -> str:
        """Set IN1..IN4 directly. Returns the applied state line."""
        return self._send(f"R {int(a)} {int(b)} {int(c)} {int(d)}")

    def drive(self, left: int, right: int) -> str:
        """Set each motor to -1 (reverse), 0 (stop) or +1 (forward)."""
        if left not in _DIR or right not in _DIR:
            raise ValueError("left and right must be -1, 0 or 1")
        a, b = _DIR[left]
        c, d = _DIR[right]
        return self.relays(a, b, c, d)

    def forward(self) -> str:
        return self.drive(1, 1)

    def backward(self) -> str:
        return self.drive(-1, -1)

    def spin_left(self) -> str:
        """Both wheels opposed — turns on the spot, hardest on the contacts."""
        return self.drive(-1, 1)

    def spin_right(self) -> str:
        return self.drive(1, -1)

    def arc_left(self) -> str:
        """Inner wheel stopped — a wider, gentler turn."""
        return self.drive(0, 1)

    def arc_right(self) -> str:
        return self.drive(1, 0)

    def stop(self) -> str:
        return self._send("S")

    def ping(self) -> str:
        """Report applied state without changing anything."""
        return self._send("P")

    def applied(self) -> tuple[int, int, int, int]:
        """Relay states as actually set right now.

        During a dead-time pause this lags what you asked for.
        """
        parts = self.ping().split()
        return tuple(int(x) for x in parts[1:5])  # type: ignore[return-value]

    # ------------------------------------------------------- config

    def info(self) -> str:
        return self._send("?")

    def config(
        self,
        deadtime_ms: Optional[int] = None,
        watchdog_ms: Optional[int] = None,
        active_low: Optional[bool] = None,
    ) -> str:
        """Change firmware behaviour at runtime — no reflashing.

        deadtime_ms=0 lets reversals happen instantly, so sequence
        the stop yourself if you still want one.
        watchdog_ms=0 means a crash here leaves the motors running.
        """
        reply = self.info()
        if deadtime_ms is not None:
            reply = self._send(f"C D {int(deadtime_ms)}")
        if watchdog_ms is not None:
            reply = self._send(f"C W {int(watchdog_ms)}")
            self._watchdog_s = watchdog_ms / 1000.0
        if active_low is not None:
            reply = self._send(f"C A {1 if active_low else 0}")

        for field in reply.split():
            if field.startswith("watchdog="):
                self._watchdog_s = int(field.split("=")[1]) / 1000.0
        return reply

    # ----------------------------------------------------- lifecycle

    def close(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.stop()
        except Exception:
            pass
        with self._lock:
            if self._ser is not None:
                self._ser.close()
                self._ser = None

    def __enter__(self) -> "Car":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<Car {self.port}>"
