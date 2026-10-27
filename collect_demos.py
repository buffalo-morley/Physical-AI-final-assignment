"""Generates ACT training data by running the scripted IK pick-and-place "expert" controller
over randomized (object_color, target_color) pairs, recording (image, proprioception, goal,
action-chunk) samples via DemoRecorder, and saving the resulting dataset to disk."""

import argparse
import itertools
import random

import numpy as np
import torch
import genesis as gs

from src.config import CUBE_COLORS, CUBE_START_POS, ZONE_COLORS
from src.demo_recorder import DemoRecorder
from src.pick_and_place_controller import go_ready, init_robot, pick, place
from src.scene_builder import build_scene


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-pair", type=int, default=15, help="3*3=9 pairs x this = total episodes")
    parser.add_argument("--output", type=str, default="outputs/act_dataset.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    episodes_per_pair = args.episodes_per_pair
    output_path = args.output
    gs.init(backend=gs.gpu, logging_level="warning")
    scene, franka, cubes, zones, cam, top_cam, wrist_cam = build_scene()
    init_robot(franka)
    scene.step()

    rng = random.Random(0)

    pairs = list(itertools.product(CUBE_COLORS, ZONE_COLORS))
    episodes = pairs * episodes_per_pair
    rng.shuffle(episodes)

    all_samples = []
    for i, (object_color, target_color) in enumerate(episodes, start=1):
        print(f"[{i}/{len(episodes)}] episode: {object_color} -> {target_color}")

        recorder = DemoRecorder(franka, top_cam, wrist_cam, object_color, target_color)

        cube_pos = cubes[object_color].get_pos().cpu().numpy()
        zone_pos = zones[target_color].get_pos().cpu().numpy()

        pick(franka, scene, cube_pos, on_step=recorder.on_step)
        place(franka, scene, zone_pos, on_step=recorder.on_step)
        go_ready(franka, scene, on_step=recorder.on_step)
        recorder.flush()

        all_samples.extend(recorder.samples)

        # Reset this cube back to its own canonical start position (ground truth, no jitter)
        # so the next randomly-chosen episode starts from a clean, known layout.
        current_pos = cubes[object_color].get_pos().cpu().numpy()
        pick(franka, scene, current_pos)
        place(franka, scene, CUBE_START_POS[object_color], surface_half_height=0.0)
        go_ready(franka, scene)

    print(f"Collected {len(all_samples)} samples. Saving to {output_path}")
    torch.save(all_samples, output_path)


if __name__ == "__main__":
    main()
