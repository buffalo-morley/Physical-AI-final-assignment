"""Manual video recording with a text caption overlay (showing the natural-language
instruction currently being executed), since Genesis's built-in camera.start_recording()
captures raw frames with no hook for drawing on them before they are encoded."""

import cv2
import numpy as np

DEFAULT_FPS = 30


class CaptionedRecorder:
    """Wire `on_step` in as a controller `on_step` callback (or call it manually once per
    sim step). Every `frame_interval` steps it grabs a frame from `cam`, burns in the current
    caption (set via `set_caption`), and buffers it for `save()`."""

    def __init__(self, cam, dt, fps=DEFAULT_FPS):
        self.cam = cam
        self.fps = fps
        self.frame_interval = max(1, round(1.0 / (dt * fps)))
        self.counter = 0
        self.caption = ""
        self.frames = []

    def set_caption(self, text):
        self.caption = text

    def _draw_caption(self, rgb):
        if not self.caption:
            return rgb
        frame = rgb.copy()
        h, w = frame.shape[:2]
        band_h = 44
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - band_h), (w, h), (15, 15, 15), -1)
        frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
        cv2.putText(
            frame,
            self.caption,
            (16, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return frame

    def on_step(self, action=None):
        if self.counter % self.frame_interval == 0:
            rgb, _, _, _ = self.cam.render()
            self.frames.append(self._draw_caption(rgb))
        self.counter += 1

    def save(self, path):
        """Uses cv2.VideoWriter (in-process encoding) rather than imageio-ffmpeg, which spawns
        an external ffmpeg subprocess -- on Windows that subprocess launch can fail with
        'WinError 6: the handle is invalid' depending on how stdout/stderr were redirected by
        whatever launched this script."""
        h, w = self.frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
        for frame in self.frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
