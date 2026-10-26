"""Entry point: a Franka Panda arm in Genesis follows natural-language pick-and-place
instructions (parsed by Claude via AWS Bedrock) and the run is recorded to an MP4 video."""

import json
import os

import genesis as gs

from src.config import CUBE_START_POS
from src.llm_instruction_parser import parse_instruction
from src.pick_and_place_controller import go_ready, init_robot, pick, place
from src.scene_builder import build_scene

OUTPUT_VIDEO = "outputs/pick_and_place_demo.mp4"
FPS = 30
IDLE_STEPS = 60  # ~0.6s of static shot, used as intro/outro padding


def idle(scene, steps):
    for _ in range(steps):
        scene.step()


def run_instruction(franka, scene, cubes, zones, instruction: str):
    print(f"instruction: {instruction}")
    action = parse_instruction(instruction)
    print(f"  -> parsed action: {action}")

    object_color = action["object_color"]
    target_color = action["target_color"]

    cube_pos = cubes[object_color].get_pos().cpu().numpy()
    zone_pos = zones[target_color].get_pos().cpu().numpy()

    pick(franka, scene, cube_pos)
    place(franka, scene, zone_pos)
    go_ready(franka, scene)


def main():
    gs.init(backend=gs.gpu, logging_level="warning")

    scene, franka, cubes, zones, cam = build_scene()
    init_robot(franka)
    scene.step()

    with open("commands.json", encoding="utf-8") as f:
        commands = json.load(f)

    os.makedirs("outputs", exist_ok=True)
    cam.start_recording(save_to_filename=OUTPUT_VIDEO, fps=FPS)

    idle(scene, IDLE_STEPS)  # show the initial layout before the first move

    for instruction in commands:
        run_instruction(franka, scene, cubes, zones, instruction)

    idle(scene, IDLE_STEPS)

    print("Returning all blocks to their starting positions...")
    for color, start_pos in CUBE_START_POS.items():
        current_pos = cubes[color].get_pos().cpu().numpy()
        pick(franka, scene, current_pos)
        place(franka, scene, start_pos, surface_half_height=0.0)
        go_ready(franka, scene)

    idle(scene, IDLE_STEPS)

    cam.stop_recording()
    print(f"Saved video to {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
