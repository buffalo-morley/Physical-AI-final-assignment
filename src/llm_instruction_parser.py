"""Parses a free-form natural language instruction into a structured pick-and-place action
using Claude via AWS Bedrock. This is where the task's "AI" requirement is satisfied:
the robot does not follow hardcoded commands, but interprets arbitrary Japanese/English
sentences at inference time."""

import json

import boto3

from src.config import BEDROCK_MODEL_ID, BEDROCK_REGION, COLORS, LAYOUT_HINT

VALID_COLORS = list(COLORS.keys())

SYSTEM_PROMPT = f"""あなたはピック&プレース・ロボットのための指示解析器です。
ユーザーの自然言語の指示を読み取り、「どの色のブロックを」「どの色のエリアに」置くかを
抽出してください。色は必ず次のいずれかに正規化してください: {", ".join(VALID_COLORS)}。

{LAYOUT_HINT}
色名が直接書かれていない場合も、上記の配置情報から「一番右」「真ん中」などの
位置表現を色に変換してください。

必ず次のJSON形式のみで回答してください（説明文や前置きは一切不要）:
{{"object_color": "<red|green|blue>", "target_color": "<red|green|blue>"}}

指示が不明瞭、または上記の色に当てはまらない場合は object_color か target_color に
null を入れてください。"""


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

    if object_color not in VALID_COLORS or target_color not in VALID_COLORS:
        raise InstructionParseError(
            f"Instruction '{instruction}' resolved to invalid colors: {result!r}"
        )

    return {"object_color": object_color, "target_color": target_color}
