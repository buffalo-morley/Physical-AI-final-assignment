"""Quantitative evaluation of a trained ACT policy: runs all 9 possible (cube, zone) pairs
(exhaustive, not a random sample -- the task only has 9 unique combinations, and the sim is
deterministic, so repeating the same pair would just repeat the same result) and reports how
far the target cube ended up from its intended zone (no video, just numbers).

Usage: eval_act.py --checkpoint outputs/act_policy_135_wrist.pt --wrist
"""

import argparse
import itertools

import cv2
import numpy as np
import torch
import genesis as gs

from src.act_model import ACTPolicy
from src.config import CUBE_COLORS, CUBE_START_POS, ZONE_COLORS
from src.demo_recorder import CHUNK_SIZE, IMG_SIZE, goal_vector
from src.pick_and_place_controller import go_ready, init_robot, pick, place
from src.scene_builder import build_scene

DECISIONS_PER_EPISODE = 27
ALL_DOFS = np.arange(9)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--wrist", action="store_true")
    parser.add_argument("--label", type=str, default=None)
    return parser.parse_args()


def get_image(cam):
    rgb, _, _, _ = cam.render()
    return cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def main():
    args = parse_args()
    label = args.label or args.checkpoint

    gs.init(backend=gs.gpu, logging_level="warning")
    scene, franka, cubes, zones, cam, top_cam, wrist_cam = build_scene()
    init_robot(franka)
    scene.step()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device)
    policy = ACTPolicy(chunk_size=CHUNK_SIZE).to(device)
    policy.load_state_dict(checkpoint["model_state"])
    policy.eval()
    action_mean = checkpoint["action_mean"].to(device)
    action_std = checkpoint["action_std"].to(device)
    proprio_mean = checkpoint["proprio_mean"].to(device)
    proprio_std = checkpoint["proprio_std"].to(device)

    pairs = list(itertools.product(CUBE_COLORS, ZONE_COLORS))

    print(f"\n===== {label} ({len(pairs)} trials, all cube/zone pairs) =====")
    n_success = 0
    results = []
    for object_color, target_color in pairs:
        start_pos = cubes[object_color].get_pos().cpu().numpy().copy()
        zone_pos = zones[target_color].get_pos().cpu().numpy().copy()

        goal = goal_vector(object_color, target_color)
        for _ in range(DECISIONS_PER_EPISODE):
            wrist_cam.move_to_attach()
            top_img = get_image(top_cam)
            wrist_img = get_image(wrist_cam)
            proprio = franka.get_dofs_position(dofs_idx_local=ALL_DOFS).cpu().numpy().astype(np.float32)
            state = np.concatenate([proprio, goal]).astype(np.float32)
            with torch.no_grad():
                top_t = torch.from_numpy(top_img).unsqueeze(0).to(device)
                wrist_t = torch.from_numpy(wrist_img).unsqueeze(0).to(device)
                state_t = torch.from_numpy(state).unsqueeze(0).to(device)
                state_t[:, :9] = (state_t[:, :9] - proprio_mean) / proprio_std
                images = [top_t, wrist_t] if args.wrist else [top_t]
                action_chunk_norm = policy(images, state_t)[0]
                action_chunk = (action_chunk_norm * action_std + action_mean).cpu().numpy()
            for a in action_chunk:
                franka.control_dofs_position(a, dofs_idx_local=ALL_DOFS)
                scene.step()

        end_pos = cubes[object_color].get_pos().cpu().numpy().copy()
        moved_dist = np.linalg.norm(end_pos[:2] - start_pos[:2])
        dist_to_zone = np.linalg.norm(end_pos[:2] - zone_pos[:2])
        success = dist_to_zone < 0.06 and end_pos[2] > 0.025
        n_success += int(success)
        results.append((object_color, target_color, moved_dist, dist_to_zone, end_pos[2], success))
        print(
            f"{object_color} -> {target_color}: moved {moved_dist*100:.1f}cm from start, "
            f"{dist_to_zone*100:.1f}cm from zone center, final_z={end_pos[2]*100:.1f}cm"
            f"  SUCCESS={success}"
        )

        # Reset for the next trial: put the cube back wherever it actually ended up picked up
        # from isn't necessary -- just pick it up from its current position and place it back
        # at its canonical start, via the known-reliable scripted controller.
        current_pos = cubes[object_color].get_pos().cpu().numpy()
        pick(franka, scene, current_pos)
        place(franka, scene, CUBE_START_POS[object_color], surface_half_height=0.0)
        go_ready(franka, scene)

    print(f"{label}: {n_success}/{len(pairs)} succeeded")


if __name__ == "__main__":
    main()
