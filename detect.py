#!/usr/bin/env python3
"""
Object detection that runs beside the driving loop instead of in it.

    from detect import Detector, Reporter

    det = Detector("yolo11n.onnx")
    det.submit(frame, "floor")        # returns immediately
    for found in det.poll():          # whatever finished since last time
        ...

Wants a YOLOv8/v11-family ONNX export, which is what Ultralytics
produces by default:

    yolo export model=yolo11n.onnx imgsz=480

-------------------------------------------------------------
WHY A THREAD AND A SINGLE SLOT

roam.py stops refreshing the bridge if its loop stalls longer than
command_timeout, and the firmware watchdog then releases the relays.
Inference on a Pi 5 CPU takes far longer than one 10 Hz tick, so
running it inline would stop the tank every time it looked at
something.

So submit() drops the frame into a one-deep slot and returns. The
worker picks up whatever is in the slot when it is free, and a frame
that arrives while it is busy REPLACES the waiting one rather than
queueing behind it. Detection then runs as fast as the Pi manages,
the loop keeps its timing, and what gets skipped is old frames —
which is the right thing to skip, since the tank has already moved
past them.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]      # x0, y0, x1, y1 in source pixels
    area_frac: float                    # share of the frame it covers
    context: str                        # "floor" while driving, "survey" when stopped

    def __str__(self) -> str:
        return f"{self.label} ({self.confidence:.0%})"


def _letterbox(bgr, size):
    """Fit into a square without distorting it, padding the rest grey."""
    h, w = bgr.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 114, np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    out[dy:dy + nh, dx:dx + nw] = resized
    return out, scale, dx, dy


class Detector:
    """A YOLO ONNX model, driven from a worker thread."""

    def __init__(self, model_path: str, labels=COCO_CLASSES,
                 conf: float = 0.40, iou: float = 0.45, threads: int = 2):
        import onnxruntime as ort       # deferred: only detection runs need it

        self.labels = tuple(labels)
        self.conf = conf
        self.iou = iou

        # Leave cores for the control loop and the camera. All four on
        # inference makes the 10 Hz tick miss, which is the one thing
        # this whole design exists to avoid.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, threads)
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, opts,
                                            providers=["CPUExecutionProvider"])

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # Exports are usually square and static; fall back to 640 when
        # the axis is symbolic.
        self.size = inp.shape[2] if isinstance(inp.shape[2], int) else 640

        self._pending = None            # (frame, context) waiting to be run
        self._results = deque(maxlen=8)
        self._infer_lock = threading.Lock()
        self._cv = threading.Condition()
        self._closing = False
        self.busy = False
        self.last_ms = 0.0

        self._thread = threading.Thread(target=self._work, name="detect",
                                        daemon=True)
        self._thread.start()

    # ------------------------------------------------------ producer

    def submit(self, bgr, context: str = "floor") -> bool:
        """Offer a frame. Never blocks. Replaces any frame still waiting."""
        with self._cv:
            if self._closing:
                return False
            dropped = self._pending is not None
            self._pending = (bgr, context)
            self._cv.notify()
        return not dropped

    def poll(self) -> list[list[Detection]]:
        """Everything that finished since the last call."""
        with self._cv:
            out = list(self._results)
            self._results.clear()
        return out

    # ------------------------------------------------------ consumer

    def _work(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._closing:
                    self._cv.wait(timeout=0.5)
                if self._closing:
                    return
                frame, context = self._pending
                self._pending = None
                self.busy = True
            try:
                found = self.run_sync(frame, context)
                with self._cv:
                    self._results.append(found)
            except Exception:
                pass                    # a bad frame must not kill the thread
            finally:
                self.busy = False

    def run_sync(self, bgr, context: str = "survey") -> list[Detection]:
        """Run on the CALLING thread. Only for when the tank is stopped."""
        started = time.monotonic()
        square, scale, dx, dy = _letterbox(bgr, self.size)
        blob = square[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        with self._infer_lock:          # one inference at a time, either caller
            raw = self.session.run(None, {self.input_name: blob})[0]

        self.last_ms = (time.monotonic() - started) * 1000.0
        return self._decode(raw, bgr.shape, scale, dx, dy, context)

    # ----------------------------------------------------- postprocess

    def _decode(self, raw, shape, scale, dx, dy, context):
        """YOLOv8/v11 head: (1, 4 + classes, anchors), boxes as cxcywh."""
        pred = np.squeeze(raw)
        if pred.ndim != 2:
            return []
        if pred.shape[0] < pred.shape[1]:       # (84, 8400) -> (8400, 84)
            pred = pred.T

        boxes_cxcywh = pred[:, :4]
        scores_all = pred[:, 4:]
        if scores_all.size == 0:
            return []

        class_ids = scores_all.argmax(axis=1)
        scores = scores_all[np.arange(len(class_ids)), class_ids]
        keep = scores >= self.conf
        if not keep.any():
            return []

        boxes_cxcywh = boxes_cxcywh[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # Undo the letterbox, back into source-frame pixels.
        cx, cy, bw, bh = boxes_cxcywh.T
        x0 = (cx - bw / 2 - dx) / scale
        y0 = (cy - bh / 2 - dy) / scale
        w = bw / scale
        h = bh / scale

        rects = np.stack([x0, y0, w, h], axis=1).tolist()
        idx = cv2.dnn.NMSBoxes(rects, scores.tolist(), self.conf, self.iou)
        if len(idx) == 0:
            return []

        H, W = shape[:2]
        frame_area = float(H * W)
        out = []
        for i in np.array(idx).flatten():
            bx, by, bw_, bh_ = rects[int(i)]
            cid = int(class_ids[int(i)])
            out.append(Detection(
                label=self.labels[cid] if cid < len(self.labels) else str(cid),
                confidence=float(scores[int(i)]),
                box=(int(max(0, bx)), int(max(0, by)),
                     int(min(W, bx + bw_)), int(min(H, by + bh_))),
                area_frac=float(bw_ * bh_) / frame_area,
                context=context,
            ))
        out.sort(key=lambda d: -d.confidence)
        return out

    def close(self) -> None:
        with self._cv:
            self._closing = True
            self._cv.notify_all()
        self._thread.join(timeout=2.0)


class Reporter:
    """Turns a stream of detections into lines worth reading.

    A detector run at 1 Hz sees the same chair a hundred times. Printing
    each one buries the one line that mattered, so a label is announced
    when it appears and then held quiet until it has been gone long
    enough to count as arriving again.
    """

    def __init__(self, cooldown: float = 25.0):
        self.cooldown = cooldown
        self._last_seen: dict[str, float] = {}

    def fresh(self, found: list[Detection], now: float) -> list[Detection]:
        out = []
        for d in found:
            key = f"{d.context}:{d.label}"
            if now - self._last_seen.get(key, -1e9) >= self.cooldown:
                out.append(d)
            self._last_seen[key] = now
        return out
