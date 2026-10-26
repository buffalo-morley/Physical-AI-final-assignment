# PAI最終課題: 自然言語指示によるPick & Place (Genesis + Franka Panda + Claude)

Physical AI応用1講座 最終課題の提出物です。[Genesis](https://genesis-embodied-ai.github.io/) 物理シミュレータ上で
Franka Panda 7軸アームを動かし、**日本語・英語の自由な自然言語指示**を AWS Bedrock 経由の
Claude (Haiku 5.5) で解析して、色付きブロックを指定されたエリアへ Pick & Place させます。

## 構成

```
main.py                         デモ実行のエントリポイント（動画を録画）
commands.json                   ロボットに与える自然言語指示のリスト
src/
  config.py                     シーン配置・色・AI(Bedrock)設定・モーションパラメータ
  scene_builder.py               Genesisシーン構築（Franka Panda, ブロック, エリア, カメラ)
  llm_instruction_parser.py      自然言語 → {object_color, target_color} をClaudeで解析
  pick_and_place_controller.py   IKベースの滑らかな動作生成・物理的な把持(摩擦グリップ)制御
outputs/
  pick_and_place_demo.mp4        生成された実行動画
```

## タスク概要

1. 手前の列に赤・緑・青のブロックが3つ、奥の列に赤・緑・青の目標エリア（パッド）が3つ配置されています。
2. `commands.json` に書かれた自然言語の指示（例:「赤いブロックを一番右のエリアに置いて」）を
   1件ずつ Claude に渡し、`{"object_color": "red", "target_color": "blue"}` のような構造化
   アクションに変換します。
3. Franka Panda がIKと物理ベースの把持（指の摩擦による実把持。テレポートではない）で
   対象のブロックを掴み、指定されたエリアまで運んで置きます。
4. 全指示の実行後、ロボットは全ブロックを初期位置へ戻します（スクリプトによる後片付け）。

## 実行方法

```bash
# Python 3.11 の venv を用意し、依存関係をインストール
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt

# AWS Bedrock (Claude) を呼び出せる認証情報を設定（例: AWS SSO / 環境変数）
# region は src/config.py の BEDROCK_REGION と合わせる

python main.py
```

実行すると `outputs/pick_and_place_demo.mp4` に動画が保存されます。

## 工夫した点

- **AIの活用**: ハードコードされたコマンドではなく、自由形式の自然言語（日本語・英語、
  色名の直接指定だけでなく「一番右の」「真ん中の」といった空間的な表現も含む）を
  Claude (AWS Bedrock) にリアルタイムで解析させ、ロボットの行動を決定しています。
  レイアウト情報をプロンプトに含めることで、色名が書かれていない曖昧な指示も解決できます。
- **物理ベースの把持**: オブジェクトを「テレポート」させるのではなく、グリッパーの指の
  摩擦接触によって物理的に把持・運搬しています。大きな横移動中に把持が外れる問題があったため、
  (1) 直線経路を複数のウェイポイントに分割してそれぞれIKを解く（手首の急な姿勢反転を防止）、
  (2) 指のPDゲインとブロックの摩擦係数を調整する、という2点で安定した把持を実現しました。
- **IKベースの滑らかな動作生成**: `genesis`のinverse_kinematics APIを用い、関節空間での
  単純な線形補間ではなく、直交座標空間でウェイポイントを刻んでIKを再計算することで、
  グリッパーの姿勢を常に下向きに保ったまま移動します。

## 使用ツール

- シミュレータ: [Genesis](https://genesis-embodied-ai.github.io/) 1.4.5 (GPU/CUDAバックエンド)
- ロボット: Franka Emika Panda (Genesis同梱のMJCFモデル)
- AI: AWS Bedrock経由の Claude Haiku 5.5 (`us.anthropic.claude-haiku-5-5`、推論のみ)
