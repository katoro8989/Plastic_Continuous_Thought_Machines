# HyperSuperLinear（NLM用ハイパーネット）使用ガイド

## 概要

`HyperSuperLinear`は、NLM（Neuron-Level Model）に動的な重み調整機能を追加します。
各ニューロンの時間的な更新則を、現在の内部状態に応じて適応的に変更できます。

## 実装の特徴

### **Synapseとの違い**

| | HyperSynapseUNET | HyperSuperLinear |
|---|------------------|------------------|
| **対象** | ニューロン間の結合 | 各ニューロンの時間的更新 |
| **入力** | `pre_synapse_input`<br>(外部データ + 内部状態) | `state_trace`<br>(各ニューロンの履歴) |
| **context** | `pre_synapse_input`<br>(同じもの) | `activated_state`<br>(現在のニューロン状態) |
| **設計** | 層ごとに独立したハイパーネット | 全ニューロンで共有のハイパーネット |

### **全ニューロン共有の設計**

```python
# HyperSynapseUNET: 各層が独立したハイパーネット
layer1: context → U1, V1, Gate1
layer2: context → U2, V2, Gate2
...

# HyperSuperLinear: 全ニューロンで共有
context → U (共通), V (共通), Gate (ニューロンごと)
```

**メリット:**
- パラメータ効率的（N=1024ニューロンでも1つのハイパーネット）
- 全ニューロンが協調して学習
- 過学習しにくい

## 使用方法

### **基本例**

```bash
python -m tasks.mazes.train_hyper \
    --model ctm \
    --use_hyper \
    --hyper_layers bottleneck \
    --hyper_rank 8 \
    --use_hyper_nlm \
    --hyper_nlm_rank 4 \
    --dataset mazes-small \
    --d_model 1024 \
    --synapse_depth 8 \
    --log_dir logs/test-nlm-hyper \
    --device 0
```

### **組み合わせパターン**

#### **1. Synapseのみハイパー化**
```bash
--use_hyper --hyper_layers bottleneck --hyper_rank 8
# use_hyper_nlmなし
```

#### **2. NLMのみハイパー化**
```bash
--use_hyper --hyper_layers none --hyper_rank 8 \
--use_hyper_nlm --hyper_nlm_rank 4
```

#### **3. 両方ハイパー化（推奨）**
```bash
--use_hyper --hyper_layers bottleneck --hyper_rank 8 \
--use_hyper_nlm --hyper_nlm_rank 4
```

#### **4. 全力ハイパー化**
```bash
--use_hyper --hyper_layers all --hyper_rank 16 \
--use_hyper_nlm --hyper_nlm_rank 8
```

## パラメータ数の比較

例: `d_model=1024`, `memory_length=25`, `memory_hidden_dims=32`

### **Deep NLM (2層SuperLinear) の場合**

| 設定 | 追加パラメータ（概算） |
|------|---------------------|
| `use_hyper_nlm=False` | 0 |
| `use_hyper_nlm=True, rank=4` | ~200K |
| `use_hyper_nlm=True, rank=8` | ~400K |
| `use_hyper_nlm=True, rank=16` | ~800K |

**計算根拠:**
- Context net: `1024 → 128` = ~130K
- head_u: `128 → in_dims * rank` = ~4K-50K
- head_v: `128 → out_dims * rank` = ~1K-4K
- head_gate: `128 → 1024` = ~130K

### **Synapse + NLM 両方の場合**

| Synapse設定 | NLM rank | 合計追加パラメータ |
|------------|----------|-------------------|
| bottleneck, r=8 | 4 | ~300K |
| bottleneck, r=8 | 8 | ~500K |
| all, r=16 | 8 | ~1.5M |

## 推奨設定

### **初期実験（軽量）**
```bash
--use_hyper --hyper_layers bottleneck --hyper_rank 8 \
--use_hyper_nlm --hyper_nlm_rank 4
```
- パラメータ増加: 最小
- 学習安定性: 高い
- まず効果があるか確認

### **標準（バランス）**
```bash
--use_hyper --hyper_layers bottleneck --hyper_rank 8 \
--use_hyper_nlm --hyper_nlm_rank 8
```

### **高性能（重い）**
```bash
--use_hyper --hyper_layers all --hyper_rank 16 \
--use_hyper_nlm --hyper_nlm_rank 16
```

## Deep NLM vs Shallow NLM

```python
# Deep NLM（デフォルト）
SuperLinear(M → 2H) → GLU → HyperSuperLinear(H → 2) → GLU
                                ↑ ここだけハイパー化

# Shallow NLM
HyperSuperLinear(M → 2) → GLU
↑ 全体がハイパー化
```

**Deep NLMの利点:**
- 前段は固定重み → 安定
- 後段だけハイパー化 → 柔軟
- パラメータも少なめ

## トラブルシューティング

### Q: "NoneType object has no attribute 'view'"
```
エラー: u = self.head_u(ctx).view(...)
```
**原因**: `context=None`が渡されている  
**解決**: `--use_hyper_nlm`フラグを確認、または`context`渡し忘れ

### Q: 学習が不安定
**解決策:**
1. `hyper_nlm_rank`を下げる（8 → 4）
2. Deep NLMを使う（`--deep_memory`）
3. 学習率を下げる

### Q: メモリ不足
**解決策:**
1. `hyper_nlm_rank`を下げる
2. `batch_size`を減らす
3. Synapse側のハイパー化を減らす（`--hyper_layers bottleneck`）

## 期待される効果

### **NLMハイパー化の利点**

1. **適応的な時間的ダイナミクス**
   - 状態に応じてニューロンの反応特性を変更
   - 例: 確信が高いときは急速な収束、不確実なときは探索的な挙動

2. **ニューロンごとの役割分化**
   - Gateが各ニューロンで異なる値
   - あるニューロンは動的、別のニューロンは固定的

3. **Synapseとの相乗効果**
   - Synapse: ニューロン間の情報共有を調整
   - NLM: 各ニューロンの更新則を調整
   - 両方で柔軟な思考プロセス

## 次のステップ

1. **まずSynapseのみ**で効果確認
   ```bash
   --use_hyper --hyper_layers bottleneck --hyper_rank 8
   ```

2. **NLMのみ**で効果確認
   ```bash
   --use_hyper --hyper_layers none \
   --use_hyper_nlm --hyper_nlm_rank 4
   ```

3. **両方**で相乗効果を確認
   ```bash
   --use_hyper --hyper_layers bottleneck --hyper_rank 8 \
   --use_hyper_nlm --hyper_nlm_rank 4
   ```

4. 効果があれば、rankや層を増やして最適化

