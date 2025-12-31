# HyperContinuousThoughtMachine 使用ガイド

## 概要

`HyperContinuousThoughtMachine`は、通常のCTMにハイパーネットワークを追加したバージョンです。
入力状態に応じてSynapseモジュールの重みを動的に調整することで、より柔軟な思考処理を実現します。

## 実装完了内容

### ✅ 実装済み

1. **HyperLoRALinear** (`models/modules.py`)
   - Low-rank動的重み調整
   - 学習可能なGate機構
   - 計算最適化済み

2. **HyperSynapseUNET** (`models/modules.py`)
   - 指定層のみハイパー化可能
   - 通常のSynapseUNETと同じインターフェース

3. **HyperContinuousThoughtMachine** (`models/ctm.py`)
   - CTMを継承した実装
   - 2つの追加パラメータのみ

## 基本的な使い方

```python
from models.ctm import HyperContinuousThoughtMachine

model = HyperContinuousThoughtMachine(
    # 通常のCTMパラメータ
    iterations=50,
    d_model=1024,
    d_input=256,
    heads=4,
    n_synch_out=128,
    n_synch_action=128,
    synapse_depth=8,
    memory_length=25,
    deep_nlms=True,
    memory_hidden_dims=32,
    do_layernorm_nlm=False,
    backbone_type='resnet18-1',
    positional_embedding_type='none',
    out_dims=250,
    neuron_select_type='random-pairing',
    dropout=0.0,
    
    # ハイパーネットワーク追加パラメータ
    hyper_layers='bottleneck',  # どの層をハイパー化するか
    hyper_rank=8,               # LoRAランク
)

# 通常のCTMと全く同じインターフェース
predictions, certainties, _ = model(x)
```

## ハイパーパラメータ

### `hyper_layers` (重要！)

どの層にハイパーネットワークを適用するかを制御：

- **`'bottleneck'`** (推奨): ボトルネック層のみ
  - 最も情報が圧縮される層
  - パラメータ増加: 最小
  - 効果: 中程度
  
- **`'down'`**: Down projection層すべて
  - エンコード過程を動的化
  - パラメータ増加: 中
  
- **`'up'`**: Up projection層すべて
  - デコード過程を動的化
  - パラメータ増加: 中
  
- **`'all'`**: すべての層
  - 最大の柔軟性
  - パラメータ増加: 大
  - 計算コスト: 高
  
- **`'none'`**: ハイパーネットなし
  - 通常のSynapseUNETと同じ

### `hyper_rank`

Low-rank分解のランク (デフォルト: 8)

- **低い (2-4)**: パラメータ効率的、表現力低め
- **中程度 (8-16)**: バランス良い
- **高い (32+)**: 表現力高い、パラメータ増加

## 訓練スクリプトへの統合例

### Option 1: 直接コードを修正

`tasks/mazes/train.py` の変更例：

```python
# Import追加
from models.ctm import ContinuousThoughtMachine, HyperContinuousThoughtMachine

# argparse追加 (line 73付近)
# CTM specific
parser.add_argument('--synapse_depth', type=int, default=8, ...)
# 以下を追加：
parser.add_argument('--use_hyper', action='store_true', 
                    help='Use HyperContinuousThoughtMachine instead of CTM')
parser.add_argument('--hyper_layers', type=str, default='bottleneck',
                    choices=['none', 'bottleneck', 'down', 'up', 'all'],
                    help='Which layers to apply hypernetwork to')
parser.add_argument('--hyper_rank', type=int, default=8,
                    help='Rank for LoRA decomposition in hypernetwork')

# モデル作成部分 (line 166付近)
if args.model == 'ctm':
    # 元のコード
    model_class = HyperContinuousThoughtMachine if args.use_hyper else ContinuousThoughtMachine
    
    model_kwargs = {
        'iterations': args.iterations,
        'd_model': args.d_model,
        'd_input': args.d_input,
        'heads': args.heads,
        'n_synch_out': args.n_synch_out,
        'n_synch_action': args.n_synch_action,
        'synapse_depth': args.synapse_depth,
        'memory_length': args.memory_length,
        'deep_nlms': args.deep_memory,
        'memory_hidden_dims': args.memory_hidden_dims,
        'do_layernorm_nlm': args.do_normalisation,
        'backbone_type': args.backbone_type,
        'positional_embedding_type': args.positional_embedding_type,
        'out_dims': args.out_dims,
        'prediction_reshaper': prediction_reshaper,
        'dropout': args.dropout,
        'dropout_nlm': args.dropout_nlm,
        'neuron_select_type': args.neuron_select_type,
        'n_random_pairing_self': args.n_random_pairing_self,
    }
    
    # ハイパーネットワーク使用時のみ追加
    if args.use_hyper:
        model_kwargs['hyper_layers'] = args.hyper_layers
        model_kwargs['hyper_rank'] = args.hyper_rank
    
    model = model_class(**model_kwargs).to(device)
```

### Option 2: Pythonスクリプトで直接使用

```python
# test_hyper.py
import torch
from models.ctm import HyperContinuousThoughtMachine

# モデル作成
model = HyperContinuousThoughtMachine(
    iterations=50,
    d_model=1024,
    d_input=256,
    heads=4,
    n_synch_out=128,
    n_synch_action=128,
    synapse_depth=8,
    memory_length=25,
    deep_nlms=True,
    memory_hidden_dims=32,
    do_layernorm_nlm=False,
    backbone_type='resnet18-1',
    positional_embedding_type='none',
    out_dims=250,
    neuron_select_type='random-pairing',
    hyper_layers='bottleneck',
    hyper_rank=8,
).cuda()

# ダミーデータでテスト
x = torch.randn(4, 3, 99, 99).cuda()
predictions, certainties, _ = model(x)

print(f"Predictions shape: {predictions.shape}")  # (4, 250, 50)
print(f"Certainties shape: {certainties.shape}")  # (4, 2, 50)
```

## コマンドライン例

```bash
# 通常のCTM
python -m tasks.mazes.train \
    --model ctm \
    --d_model 1024 \
    --d_input 256 \
    --synapse_depth 8 \
    --hyper_layers none \
    ...

# Bottleneckのみハイパー化（推奨）
python -m tasks.mazes.train \
    --model ctm \
    --use_hyper \
    --hyper_layers bottleneck \
    --hyper_rank 8 \
    --d_model 1024 \
    --d_input 256 \
    --synapse_depth 8 \
    ...

# すべての層をハイパー化
python -m tasks.mazes.train \
    --model ctm \
    --use_hyper \
    --hyper_layers all \
    --hyper_rank 8 \
    ...
```

## パラメータ数の比較

例: `d_model=1024`, `synapse_depth=8`, `rank=8`

| 設定 | 追加パラメータ数 (概算) |
|------|-------------------------|
| `hyper_layers='none'` | 0 (通常のCTM) |
| `hyper_layers='bottleneck'` | ~100K |
| `hyper_layers='down'` | ~500K |
| `hyper_layers='up'` | ~500K |
| `hyper_layers='all'` | ~1M |

## 推奨設定

### 初期実験
```python
hyper_layers='bottleneck'
hyper_rank=8
```
- バランスが良く、効果を確認しやすい

### 高性能
```python
hyper_layers='all'
hyper_rank=16
```
- 最大の柔軟性、計算コスト高

### 軽量
```python
hyper_layers='bottleneck'
hyper_rank=4
```
- パラメータ最小、効果は限定的

## トラブルシューティング

### Q: 学習が不安定になった
A: `hyper_rank`を下げるか、`hyper_layers='bottleneck'`から始めてください

### Q: メモリ不足
A: `hyper_layers='bottleneck'`を使うか、`batch_size`を減らしてください

### Q: 通常のCTMとの性能差がない
A: 以下を試してください:
- `hyper_rank`を増やす (8 → 16)
- `hyper_layers='all'`に変更
- タスクがハイパーネットワークの恩恵を受けにくい可能性

## 次のステップ

1. ✅ **実装完了**: HyperLoRALinear, HyperSynapseUNET, HyperContinuousThoughtMachine
2. 🔄 **統合作業**: 訓練スクリプトへの統合 (オプション)
3. 🧪 **実験**: Bottleneckでの性能検証
4. 📊 **分析**: 動的重みの可視化
5. 🚀 **拡張**: NLM (SuperLinear) へのハイパーネット適用

## 技術詳細

### アーキテクチャ

```
Input (x) → [Context Network] → Gate, U, V
                                   ↓
Base Layer (W_base) ← x       LoRA (U @ V^T)
         ↓                          ↓
    out_base              Gate * (U @ (V^T @ x))
         ↓                          ↓
         └────────── (+) ───────────┘
                      ↓
                   Output
```

### 計算量

- Base Layer: O(B * In * Out)
- LoRA Path: O(B * In * Rank + B * Out * Rank)
- Total: O(B * (In * Out + In * Rank + Out * Rank))

Rankが小さいため、通常の線形層とほぼ同じ計算量です。

