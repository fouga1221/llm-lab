# スクリプトREADME（検証用CLI/ツールの使い方）

本READMEでは、本リポジトリに含まれる検証スクリプトの使い方をまとめます。目的別に最小構成で試せるよう、各コマンドの主なオプション・実行例・出力を記載しています。

## 前提条件
- Python 3.10+
- 依存導入: `pip install -r requirements.txt`
- モデルはHugging Faceから直接読み込みます（事前に`huggingface-cli login`等は不要ですが、権限が必要なモデルは別途対応）。
- APIサーバ機能を使う場合は、既定の設定ファイルが必要です。
  - `configs/model.yaml`（同梱のプレースホルダで可）
  - `configs/security.yaml`（同梱）
  - `configs/functions.json`（同梱）

---

## 1) 対話CLI: `scripts/chat_cli.py`
ローカルでモデルを直接ロードし、1回のプロンプトまたは複数ターンの対話で応答を生成します。APIサーバ不要・最短でモデルの手触りを確認できます。

- 主なオプション
  - `--model-id`（必須）: 例 `Qwen/Qwen2.5-7B-Instruct`
  - `--revision`: モデルのタグ/リビジョン
  - `--dtype`: `float16|bfloat16|float32`（既定 `bfloat16`）
  - `--device`: `auto|cpu|cuda`（既定 `auto`）
  - `--max-new-tokens`（既定 256）, `--temperature`（既定 0.6）, `--top-p`（既定 0.9）, `--repetition-penalty`（既定 1.0）
  - `--system`: システムプロンプトファイル（例 `app/prompts/system_prompt.md`）
  - `--json-schema`: 生成文中のJSONブロックを検証するスキーマ（任意）
  - `--save`: 生成結果をJSONLで保存（例 `runs/chat.jsonl`）
  - `--input`: ユーザ入力（省略時は標準入力から1回分を読み取り）
  - `--interactive`: 対話REPLを開始（複数ターンの履歴を維持）
  - `--keep-last`: REPL時に保持する直近ターン数（ユーザ+アシスタントのペア数, 既定 5）

- 実行例
  - 最小: `python scripts/chat_cli.py --model-id Qwen/Qwen2.5-7B-Instruct --input "宿屋はどこ？"`
  - スキーマ検証と保存: `python scripts/chat_cli.py --model-id Qwen/Qwen2.5-7B-Instruct --system app/prompts/system_prompt.md --json-schema configs/actions.schema.json --input "ギルドまで案内して" --save runs/chat.jsonl`
  - 対話REPL: `python scripts/chat_cli.py --model-id Qwen/Qwen2.5-7B-Instruct --interactive --save runs/chat_session.jsonl`
    - REPLコマンド: `:exit`/`:quit` 終了, `:reset` 履歴クリア

- 出力
  - 標準出力に応答テキスト
  - 1回実行時: `{"model_id", "prompt", "reply", "latency_ms", "tokens_per_sec", "json_valid"}` を1行1JSONで追記
  - 対話REPL時: ターンごとに `{"session","turn","model_id","user","assistant","latency_ms","tokens_per_sec","json_valid"}` を追記

注意: Windows PowerShellでJSONや引用符を含む引数を渡す場合は、ダブルクォートのエスケープに留意してください。

---

## 2) モデル比較: `scripts/compare_models.py`
複数モデル×複数プロンプトを一括で実行し、レイテンシやトークン毎秒、（任意で）JSON妥当性をCSVに集計します。

- 主なオプション
  - `--models`（必須, 複数）: 例 `--models Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct`
  - `--prompts`（必須）: テキスト（1行1プロンプト）またはJSONL（`input`/`prompt`フィールド）
  - `--out`（必須）: 出力CSVパス
  - `--max-new-tokens` `--temperature` `--top-p` `--revision` `--dtype` `--device`
  - `--json-schema`: 検証スキーマ（任意）

- 実行例
  - `python scripts/compare_models.py --models Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct --prompts data/scenarios/sample.jsonl --out reports/compare.csv`

- 出力
  - CSV（列: `model_id,prompt_idx,latency_ms,tokens_per_sec,json_valid`）

---

## 2.5) ベンチ実行: `scripts/bench_infer.py`
モデル×量子化×ランタイム（現状transformersのみ）×生成設定でベンチマークを実行し、CSVにログします。`NVML`があれば平均GPU利用率も記録します。

- 主なオプション
  - `--sweep`（必須）: スイープ設定YAML（例 `configs/sweep.example.yaml`）
  - `--out`（既定 `results/runs.csv`）: 出力CSV
  - `--timeout`（既定 60）: ケースごとのタイムアウト秒

- 記録列
  - `model,quant,runtime,attn,kv,kv_dtype,batch_size,load_ms,first_token_ms,tokens_per_s,peak_vram_alloc_mb,peak_vram_reserved_mb,avg_gpu_util,oom,notes`

- 備考
  - 1ウォームアップ+2計測の3行を出力し、併せて`median`行も追記します。
  - `runtime != transformers` はスキップ行（stub）を出力。将来拡張予定。

---

## 3) デコード探索: `scripts/sweep_decode.py`
単一モデルに対し、`temperature`/`top_p`/`max_new_tokens` のグリッドを走査し、性能の当たりを探ります。

- 主なオプション
  - `--model-id`（必須）, `--prompt`（必須）, `--grid`（必須: YAML）
  - `--revision` `--dtype` `--device`

- グリッドYAML例（`configs/decode.sample.yaml`）
```
max_new_tokens: [128, 256]
temperature: [0.5, 0.6, 0.7]
top_p: [0.85, 0.9]
```

- 実行例
  - `python scripts/sweep_decode.py --model-id Qwen/Qwen2.5-7B-Instruct --grid configs/decode.sample.yaml --prompt "宿屋はどこ？" --out reports/sweep.csv`

- 出力
  - CSV（列: `temperature,top_p,max_new_tokens,latency_ms,tokens_per_sec`）

---

## 4) APIチャット: `scripts/api_chat.py`
起動中のFastAPIサーバ（`scripts/run_server.py`）へ `/chat` を投げる簡易クライアントです。

- 主なオプション
  - `--endpoint`（既定 `http://localhost:8000/chat`）
  - `--session`（既定 `s1`）
  - `--input`（必須）

- 実行例
  - `python scripts/api_chat.py --input "市場はどこ？"`

- 出力
  - `/chat` のJSONレスポンスを整形表示

---

## 4.5) 単発推論: `scripts/run_infer.py`
最小限の一発推論ヘルパ。`chat_cli.py` より軽量なワンライナー用途に。

- 実行例
  - `python scripts/run_infer.py --model-id Qwen/Qwen2-7B-Instruct --input "宿屋はどこ？"`

---

## 5) structured_actions検証: `scripts/validate_actions.py`
JSONLログの `structured_actions` をルートスキーマ＋関数定義（`functions.json`）で検証します。

- 主なオプション
  - `--in`（必須）: JSONL（各行に `structured_actions` または `actions` を含む）
  - `--functions`（必須）: `configs/functions.json`
  - `--schema`: ルートスキーマ（省略時は緩い既定を使用）
  - `--report`: テキストレポートの出力先（省略時は標準出力）

- 実行例
  - `python scripts/validate_actions.py --in runs/chat.jsonl --functions configs/functions.json --report reports/validate.txt`

- 出力
  - 各行ごとの妥当性（OK/エラー詳細）と集計（Valid N/M）

---

## 6) ログ集計: `scripts/log_summary.py`
`chat_cli.py` などで保存したJSONLから、レイテンシのP50/P95、トークン毎秒平均、JSON妥当率を集計します。

- 主なオプション
  - `--logs`（必須）: JSONLのglob（例 `runs/*.jsonl`）

- 実行例
  - `python scripts/log_summary.py --logs runs/*.jsonl`

- 出力
  - 標準出力にメトリクス要約

---

## 7) APIサーバ起動: `scripts/run_server.py`
FastAPIサーバを起動します（`/chat` `/propose_action` `/schema/functions`）。

- 実行例
  - `python scripts/run_server.py --reload`

- 備考
  - 現状、設定パス（`configs/model.yaml` `configs/security.yaml` `configs/functions.json`）はアプリ内で既定参照しています。将来的にCLIオプション化予定です。

---

## 8) 量子化ユーティリティ

- AWQ 4bit: `scripts/quantize_awq.py`
  - 例: `python scripts/quantize_awq.py --model-id Qwen/Qwen2-7B-Instruct --output-dir ./models/qwen2-7b-awq`
- GPTQ 4bit: `scripts/quantize_gptq.py`
  - 例: `python scripts/quantize_gptq.py --model-id Qwen/Qwen2-7B-Instruct --output-dir ./models/qwen2-7b-gptq`

備考: いずれも対応パッケージ（`autoawq`/`auto-gptq`）の導入が必要です。

---

## 9) QLoRA学習: `scripts/qlora_train.py`
PEFT+bitsandbytesでQLoRAによるSFTを行います。設定はYAML（例 `configs/train.lora.qwen2.yaml`）で渡します。

- 実行例
  - `python scripts/qlora_train.py --config configs/train.lora.qwen2.yaml`


## 代表的なワークフロー（M1）
1. 対話CLIでモデルの感触確認
   - `python scripts/chat_cli.py --model-id Qwen/Qwen2.5-7B-Instruct --input "宿屋はどこ？" --save runs/chat.jsonl`
2. デコード探索で安定設定を特定
   - `python scripts/sweep_decode.py --model-id Qwen/Qwen2.5-7B-Instruct --grid configs/decode.sample.yaml --prompt "ギルドまで案内して" --out reports/sweep.csv`
3. モデル横比較
   - `python scripts/compare_models.py --models Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct --prompts data/scenarios/sample.jsonl --out reports/compare.csv`
4. JSON整形式の検証/集計
   - `python scripts/validate_actions.py --in runs/chat.jsonl --functions configs/functions.json --report reports/validate.txt`
   - `python scripts/log_summary.py --logs runs/*.jsonl`

---

## トラブルシューティング
- ImportError/ModuleNotFoundError: `pip install -r requirements.txt` を再実行。
- CUDA/メモリ不足: `--dtype float16` → `bfloat16/float32` へ、`--device cpu` や小さいモデルを選択。
- モデル読み込み失敗: モデルID・リビジョンの綴りと公開設定を確認。
- JSON検証失敗: スキーマ（`--json-schema`）と出力の整合、抽出ロジック（単純な最外括弧抽出）の限界を考慮。
