"""Scene layout, color, and AI (Bedrock) configuration for the pick-and-place demo."""

# --- AWS Bedrock (Claude) settings ---
BEDROCK_REGION = "us-east-1"
BEDROCK_MODEL_ID = "us.anthropic.claude-haiku-5-5"

# Cubes and zones intentionally use two disjoint color vocabularies (instead of both being
# red/green/blue) so that a cube never "should" visually match its target zone by color --
# avoiding the false expectation that e.g. a red cube belongs on a red zone.
CUBE_COLORS = ["red", "green", "blue"]
ZONE_COLORS = ["yellow", "purple", "orange"]

# --- Colors (RGBA, 0-1) ---
COLORS = {
    "red": (0.85, 0.15, 0.15, 1.0),
    "green": (0.15, 0.75, 0.25, 1.0),
    "blue": (0.15, 0.35, 0.9, 1.0),
    "yellow": (0.9, 0.8, 0.1, 1.0),
    "purple": (0.55, 0.15, 0.75, 1.0),
    "orange": (0.95, 0.5, 0.1, 1.0),
}

# --- Cube layout (near row, within easy reach of the Franka base at origin) ---
CUBE_SIZE = 0.04
CUBE_START_POS = {
    "red": (0.35, -0.2, CUBE_SIZE / 2),
    "green": (0.35, 0.0, CUBE_SIZE / 2),
    "blue": (0.35, 0.2, CUBE_SIZE / 2),
}

# --- Target zone layout (far row, flat colored pads) ---
ZONE_SIZE = (0.12, 0.12, 0.01)
ZONE_POS = {
    "yellow": (0.55, -0.2, ZONE_SIZE[2] / 2),
    "purple": (0.55, 0.0, ZONE_SIZE[2] / 2),
    "orange": (0.55, 0.2, ZONE_SIZE[2] / 2),
}

# --- Scene layout description, used to let the LLM resolve spatial references like
# "the rightmost block" in addition to direct color names. ---
LAYOUT_HINT = (
    "The cubes are in the near row; from the camera's left to right they are: "
    "red, green, blue. The target zones are in the far row; from left to right they are: "
    "yellow, purple, orange."
)

# --- Motion parameters ---
GRASP_APPROACH_HEIGHT = 0.18  # height above object/target for pre-grasp waypoint
GRIPPER_OPEN = 0.04
GRIPPER_CLOSE = 0.0
MOVE_STEPS = 160  # simulation substeps for a full point-to-point arm transition
APPROACH_STEPS = 110  # substeps for the slower, more deliberate vertical approach/descent
SETTLE_STEPS = 45  # extra steps to let the controller/physics settle at a waypoint
GRASP_STEPS = 80  # substeps to open/close the gripper
