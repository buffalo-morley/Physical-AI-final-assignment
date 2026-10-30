# PAI最終課題: 自然言語指示によるPick & Place (Genesis + Franka Panda + ACT + Claude)

Physical AI応用1講座 最終課題の提出物です。[Genesis](https://genesis-embodied-ai.github.io/) 物理シミュレータ上で
Franka Panda 7軸アームを動かし、**自由形式の自然言語指示**を AWS Bedrock 経由の
Claude (Haiku 5.5) で解析して、色付きブロックを指定されたエリアへ Pick & Place させます。

核となるのは **ACT（Action Chunking Transformer）による模倣学習ポリシー**です。
カメラ画像と関節角度から次の行動を直接予測するニューラルネットワークにより、知覚から
行動までを一貫してAIが担います。このACTの学習データは、**IK（逆運動学）制御の
エキスパート**を自動的に走らせて収集しています（人間のテレオペレーションは使用して
いません）。データ量・カメラ構成による性能の違いの比較や、失敗パターンの分析も
行っています（詳細は `report/report.md` 参照）。

## 構成

```
collect_demos.py                 IKエキスパートを走らせてACT学習データを収集
train_act.py                     ACTポリシーの学習（天井カメラのみ / 天井+手首カメラ）
run_act_demo.py                  学習済みACTポリシーをクローズドループで実行し動画化
eval_act.py                      9パターン（3色×3エリア）全数評価による定量比較

main.py                          IKエキスパート単体のデモ実行（動画にキャプション付与）
commands.json                    ロボットに与える自然言語指示のリスト（英語）
src/
  config.py                     シーン配置・色・AI(Bedrock)設定・モーションパラメータ
  scene_builder.py               Genesisシーン構築（Franka Panda, ブロック, エリア, 3台のカメラ）
  llm_instruction_parser.py      自然言語 → {object_color, target_color} をClaudeで解析
  pick_and_place_controller.py   IKベースの滑らかな動作生成・物理的な把持(摩擦グリップ)制御
  demo_recorder.py               ACT学習データの記録ロジック
  act_model.py                   ACT(Action Chunking Transformer)ポリシーの実装
  video_recorder.py              キャプション付き動画記録（cv2.VideoWriterベース）

outputs/
  act_pick_and_place_demo_135_wrist.mp4   提出動画（ACT, 135エピソード+天井+手首カメラ）
  pick_and_place_demo.mp4                 IKエキスパート単体のデモ動画（確実に動作）
```

## タスク概要

1. 手前の列に**赤・緑・青**のブロックが3つ、奥の列に**黄・紫・オレンジ**の目標エリア（パッド）が
   3つ配置されています。
2. `commands.json` に書かれた自然言語の指示（例: "Put the red cube in the rightmost zone."）を
   1件ずつ Claude に渡し、`{"object_color": "red", "target_color": "orange"}` のような構造化
   アクションに変換します。色名を直接指定する指示だけでなく、"the rightmost zone" のような
   空間的な表現も、シーン配置のヒントをプロンプトに含めることで解決します。
3. ACTポリシーが、カメラ画像と関節角度から対象ブロックを把持し、指定されたエリアまで
   運んで置くための行動チャンクを予測・実行します（物理ベースの摩擦把持。テレポートではない）。
4. このACTの学習データは、IKエキスパートを様々な(ブロック色, エリア色)の組み合わせで
   自動的に走らせて収集したものです。IKエキスパート単体でも確実にPick & Placeを実行できます
   （`main.py`）。

## 実行方法

```bash
# Python 3.11 の venv を用意し、依存関係をインストール
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt

# AWS Bedrock (Claude) を呼び出せる認証情報を設定（例: AWS SSO / 環境変数）
# region は src/config.py の BEDROCK_REGION と合わせる

# 1. IKエキスパートでACT学習データを収集
python collect_demos.py --episodes-per-pair 15 --output outputs/act_dataset_135.pt

# 2. ACTポリシーを学習（天井+手首カメラ）
python train_act.py --dataset outputs/act_dataset_135.pt --checkpoint outputs/act_policy_135_wrist.pt --wrist

# 3. 学習済みACTポリシーで動画生成
python run_act_demo.py --checkpoint outputs/act_policy_135_wrist.pt --wrist --video outputs/act_demo.mp4

# 4. 9パターン全数評価
python eval_act.py --checkpoint outputs/act_policy_135_wrist.pt --wrist --label "135ep / top+wrist"

# IKエキスパート単体のデモ（確実に動作する比較対象）
python main.py
```

## 工夫した点

- **AIの活用（2層構造）**: 自然言語の解析はClaude (AWS Bedrock)が、物理的な操作
  （把持・搬送・設置）はACT（模倣学習ポリシー）が担う構成です。色名の直接指定だけでなく
  「一番右の」のような空間的な表現もClaudeが解決し、どのブロックをどのエリアへ運ぶかを
  ACTポリシーへの条件として渡します。
- **ACT（模倣学習）によるPick & Place**: IKエキスパートを「お手本」として自動的に
  実演データを収集し、画像+関節角度→行動チャンクを予測するACT方式のポリシーを学習しました。
  天井カメラのみ／天井+手首カメラ、45／135エピソードの2×2で比較したところ、
  データ不足の状態ではカメラを増やすと逆に性能が落ちる一方、データを十分に増やすと
  手首カメラが大きく効くという相互作用が見られました（135エピソード+両カメラで9パターン中8パターン成功）。


## 使用ツール

- シミュレータ: [Genesis](https://genesis-embodied-ai.github.io/) 1.4.5 (GPU/CUDAバックエンド)
- ロボット: Franka Emika Panda (Genesis同梱のMJCFモデル)
- AI（言語理解）: AWS Bedrock経由の Claude Haiku 5.5 (`us.anthropic.claude-haiku-5-5`、推論のみ)
- AI（物理操作）: 自作のACT(Action Chunking Transformer)実装（PyTorch、学習は自前データで実施）
