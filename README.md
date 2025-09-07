# llm-lab

7B級LLMのコスパ検証（レイテンシ/スループット/VRAM/安定性）と、NPC対話＋コード提案のプロト実装用リポジトリです。

## クイックスタート

前提: DevContainer もしくは venv を使用。CUDA 環境では `requirements.txt` をそのまま導入。

1) 依存導入

```
pip install -r requirements.txt
```

2) 単発推論（Transformers）

```
python scripts/run_infer.py --model-id Qwen/Qwen2-7B-Instruct --input "こんにちは"
```

3) ベンチ実行（スイープ）

```
python scripts/bench_infer.py --sweep configs/sweep.example.yaml --out results/runs.csv
```

4) API 起動（FastAPI）

```
uvicorn app.main:app --reload
```

5) 追加依存（任意）

- vLLM: `pip install "vllm>=0.5"`
- ExLlamaV2: 各環境の手順に従いインストール
- llama-cpp-python: `pip install llama-cpp-python`

導入されていない場合、ベンチでは該当ランタイムをスキップ（notes列に記録）します。

## Colab 手順（Pro+想定）

1) リポジトリ取得・依存導入

```
!git clone <this-repo>
%cd llm-lab
!pip install -r requirements.txt
```

2) 簡単な推論 → ベンチ

```
!python scripts/run_infer.py --model-id Qwen/Qwen2-7B-Instruct --input "こんにちは"
!python scripts/bench_infer.py --sweep configs/sweep.example.yaml --out results/runs.csv
```

GPU メモリ監視は別セルで `!nvidia-smi -l 3` を推奨。

## 結果の見方

`results/runs.csv` の列:

```
model,quant,runtime,attn,kv,kv_dtype,batch_size,load_ms,first_token_ms,tokens_per_s,peak_vram_alloc_mb,peak_vram_reserved_mb,avg_gpu_util,oom,notes
```

- first_token_ms: 初トークンまでの遅延
- tokens_per_s: 生成スループット
- peak_vram_*: PyTorch のピーク確保/予約（MB）
- avg_gpu_util: NVML による平均GPU使用率（未取得時は -1）

## トラブルシュート

- OOM: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を試す、max_new_tokens を下げる、量子化を使う
- 依存の不整合: `pip install -r requirements.txt` を再実行
- 追加ランタイム未導入: ベンチでは自動で skip。

