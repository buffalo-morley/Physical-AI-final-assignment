"""Parses a free-form natural language instruction into a structured pick-and-place action
using Claude via AWS Bedrock. This is where the task's "AI" requirement is satisfied:
the robot does not follow hardcoded commands, but interprets arbitrary sentences at
inference time."""

import json

import boto3

from src.config import BEDROCK_MODEL_ID, BEDROCK_REGION, CUBE_COLORS, LAYOUT_HINT, ZONE_COLORS

SYSTEM_PROMPT = f"""You are the instruction parser for a pick-and-place robot.

Read the user's natural-language instruction and extract which cube to pick up and which
target zone to place it in.

The cubes and the target zones use two DIFFERENT color vocabularies:
- cube color must be one of: {", ".join(CUBE_COLORS)}
- target zone color must be one of: {", ".join(ZONE_COLORS)}

{LAYOUT_HINT}
If the instruction uses a spatial reference instead of a color name (e.g. "the rightmost
block", "the middle zone"), resolve it to a color using the layout above.

Respond with ONLY this JSON (no prose, no code fences):
{{"object_color": "<{"|".join(CUBE_COLORS)}>", "target_color": "<{"|".join(ZONE_COLORS)}>"}}

If the instruction is ambiguous, or refers to a color outside the lists above, set the
corresponding field to null."""


class InstructionParseError(Exception):
    pass


def parse_instruction(instruction: str) -> dict:
    """Calls Claude (via Bedrock) to turn a natural language instruction into
    {"object_color": ..., "target_color": ...}."""
    client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

    response = client.converse(
        modelId=BEDROCK_MODEL_ID,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": instruction}]}],
        inferenceConfig={"maxTokens": 512},
        additionalModelRequestFields={"thinking": {"type": "disabled"}},
    )

    text = None
    for block in response["output"]["message"]["content"]:
        if "text" in block:
            text = block["text"].strip()
            break
    if text is None:
        raise InstructionParseError(f"No text content block in LLM response: {response!r}")

    # Be lenient in case the model wraps the JSON in a code block.
    if "```" in text:
        text = text.split("```")[1].replace("json", "", 1).strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        raise InstructionParseError(f"Could not parse LLM output as JSON: {text!r}") from e

    object_color = result.get("object_color")
    target_color = result.get("target_color")

    if object_color not in CUBE_COLORS or target_color not in ZONE_COLORS:
        raise InstructionParseError(
            f"Instruction '{instruction}' resolved to invalid colors: {result!r}"
        )

    return {"object_color": object_color, "target_color": target_color}
