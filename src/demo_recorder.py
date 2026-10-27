"""Records (images, proprioception, goal, action-chunk) samples from the scripted "expert"
pick-and-place controller, for training the ACT imitation-learning policy.

Both the static overhead camera and the wrist (eye-in-hand) camera are captured for every
sample, so a single data-collection pass can be used to train and compare policies that use
only the overhead view against policies that also get the wrist view."""

import cv2
import numpy as np

from src.config import CUBE_COLORS, ZONE_COLORS

IMG_SIZE = 84
CHUNK_SIZE = 50
STRIDE = 25  # how often (in sim steps) a new decision-point sample is opened
ACTION_DIM = 9
# proprioception + one-hot(object_color over CUBE_COLORS) + one-hot(target_color over ZONE_COLORS)
STATE_DIM = 9 + len(CUBE_COLORS) + len(ZONE_COLORS)


def color_onehot(color, vocab):
    v = np.zeros(len(vocab), dtype=np.float32)
    v[vocab.index(color)] = 1.0
    return v


def goal_vector(object_color, target_color):
    return np.concatenate([color_onehot(object_color, CUBE_COLORS), color_onehot(target_color, ZONE_COLORS)])


class DemoRecorder:
    """Call `on_step(action)` once per simulation step (wire it in as the controller's
    `on_step` callback). Every `STRIDE` steps it snapshots both cameras + proprioception, then
    collects the following `CHUNK_SIZE` expert actions to pair with that snapshot."""

    def __init__(self, franka, top_cam, wrist_cam, object_color, target_color, stride=STRIDE, chunk_size=CHUNK_SIZE):
        self.franka = franka
        self.top_cam = top_cam
        self.wrist_cam = wrist_cam
        self.goal = goal_vector(object_color, target_color)
        self.stride = stride
        self.chunk_size = chunk_size
        self.counter = 0
        self.samples = []
        self._pending = None

    def _snapshot(self):
        self.wrist_cam.move_to_attach()

        top_rgb, _, _, _ = self.top_cam.render()
        top_img = cv2.resize(top_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        wrist_rgb, _, _, _ = self.wrist_cam.render()
        wrist_img = cv2.resize(wrist_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        proprio = self.franka.get_dofs_position(dofs_idx_local=np.arange(9)).cpu().numpy().astype(np.float32)
        state = np.concatenate([proprio, self.goal]).astype(np.float32)
        return {"top_image": top_img, "wrist_image": wrist_img, "state": state, "actions": []}

    def on_step(self, action):
        if self._pending is not None:
            self._pending["actions"].append(np.asarray(action, dtype=np.float32))
            if len(self._pending["actions"]) >= self.chunk_size:
                self._pending["actions"] = np.stack(self._pending["actions"])
                self.samples.append(self._pending)
                self._pending = None

        if self._pending is None and self.counter % self.stride == 0:
            self._pending = self._snapshot()

        self.counter += 1

    def flush(self):
        """Pad and close any in-progress sample at the end of an episode (repeats the last
        action to fill out the chunk, so the tail of an episode is not discarded)."""
        if self._pending is not None and self._pending["actions"]:
            actions = self._pending["actions"]
            while len(actions) < self.chunk_size:
                actions.append(actions[-1])
            self._pending["actions"] = np.stack(actions)
            self.samples.append(self._pending)
            self._pending = None
