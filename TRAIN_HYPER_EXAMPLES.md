# train_hyper.py 使用例

`train_hyper.py`は、通常のCTMとHyperContinuousThoughtMachineの両方をサポートする訓練スクリプトです。

## 修正内容

### 追加されたパラメータ

```bash
--use_hyper              # HyperCTMを使用（フラグ）
--hyper_layers           # どの層をハイパー化するか
                         # 選択肢: 'none', 'bottleneck', 'down', 'up', 'all'
                         # デフォルト: 'bottleneck'
--hyper_rank             # LoRAランク
                         # デフォルト: 8
```

## 使用例

### 1. 通常のCTM（ベースライン）

```bash
python -m tasks.mazes.train_hyper \
    --model ctm \
    --dataset mazes-small \
    --d_model 1024 \
    --d_input 256 \
    --synapse_depth 8 \
    --heads 4 \
    --n_synch_out 128 \
    --n_synch_action 128 \
    --iterations 50 \
    --memory_length 25 \
    --deep_memory \
    --memory_hidden_dims 32 \
    --backbone_type resnet18-1 \
    --neuron_select_type random-pairing \
    --batch_size 64 \
    --batch_size_test 32 \
    --lr 1e-4 \
    --training_iterations 10000 \
    --track_every 1000 \
    --save_every 2000 \
    --log_dir logs/mazes-small-baseline \
    --device 0
```

### 2. HyperCTM - Bottleneck（推奨初期設定）

```bash
python -m tasks.mazes.train_hyper \
    --model ctm \
    --use_hyper \
    --hyper_layers bottleneck \
    --hyper_rank 8 \
    --dataset mazes-small \
    --d_model 1024 \
    --d_input 256 \
    --synapse_depth 8 \
    --heads 4 \
    --n_synch_out 128 \
    --n_synch_action 128 \
    --iterations 50 \
    --memory_length 25 \
    --deep_memory \
    --memory_hidden_dims 32 \
    --backbone_type resnet18-1 \
    --neuron_select_type random-pairing \
    --batch_size 64 \
    --batch_size_test 32 \
    --lr 1e-4 \
    --training_iterations 10000 \
    --track_every 1000 \
    --save_every 2000 \
    --log_dir logs/mazes-small-hyper-bottleneck \
    --device 0
```

### 3. HyperCTM - すべての層（高性能）

```bash
python -m tasks.mazes.train_hyper \
    --model ctm \
    --use_hyper \
    --hyper_layers all \
    --hyper_rank 16 \
    --dataset mazes-small \
    --d_model 1024 \
    --d_input 256 \
    --synapse_depth 8 \
    --heads 4 \
    --n_synch_out 128 \
    --n_synch_action 128 \
    --iterations 50 \
    --memory_length 25 \
    --deep_memory \
    --memory_hidden_dims 32 \
    --backbone_type resnet18-1 \
    --neuron_select_type random-pairing \
    --batch_size 32 \
    --batch_size_test 32 \
    --lr 1e-4 \
    --training_iterations 10000 \
    --track_every 1000 \
    --save_every 2000 \
    --log_dir logs/mazes-small-hyper-all \
    --device 0
```

### 4. HyperCTM - Down層のみ

```bash
python -m tasks.mazes.train_hyper \
    --model ctm \
    --use_hyper \
    --hyper_layers down \
    --hyper_rank 8 \
    --dataset mazes-small \
    --d_model 1024 \
    --d_input 256 \
    --synapse_depth 8 \
    --log_dir logs/mazes-small-hyper-down \
    --device 0 \
    ... # その他のパラメータ
```

### 5. HyperCTM - Up層のみ

```bash
python -m tasks.mazes.train_hyper \
    --model ctm \
    --use_hyper \
    --hyper_layers up \
    --hyper_rank 8 \
    --dataset mazes-small \
    --d_model 1024 \
    --d_input 256 \
    --synapse_depth 8 \
    --log_dir logs/mazes-small-hyper-up \
    --device 0 \
    ... # その他のパラメータ
```

## 比較実験の設定

### 実験1: Bottleneck vs ベースライン

```bash
# ベースライン
python -m tasks.mazes.train_hyper --model ctm \
    --dataset mazes-small --d_model 1024 --d_input 256 \
    --synapse_depth 8 --iterations 50 \
    --log_dir logs/exp1-baseline --device 0

# HyperCTM (Bottleneck)
python -m tasks.mazes.train_hyper --model ctm --use_hyper \
    --hyper_layers bottleneck --hyper_rank 8 \
    --dataset mazes-small --d_model 1024 --d_input 256 \
    --synapse_depth 8 --iterations 50 \
    --log_dir logs/exp1-hyper-bottleneck --device 0
```

### 実験2: Rankの影響

```bash
# Rank 4
python -m tasks.mazes.train_hyper --model ctm --use_hyper \
    --hyper_layers bottleneck --hyper_rank 4 \
    --log_dir logs/exp2-rank4 --device 0 ...

# Rank 8
python -m tasks.mazes.train_hyper --model ctm --use_hyper \
    --hyper_layers bottleneck --hyper_rank 8 \
    --log_dir logs/exp2-rank8 --device 0 ...

# Rank 16
python -m tasks.mazes.train_hyper --model ctm --use_hyper \
    --hyper_layers bottleneck --hyper_rank 16 \
    --log_dir logs/exp2-rank16 --device 0 ...
```

### 実験3: 層の選択

```bash
# Bottleneck
python -m tasks.mazes.train_hyper --model ctm --use_hyper \
    --hyper_layers bottleneck --log_dir logs/exp3-bottleneck --device 0 ...

# Down
python -m tasks.mazes.train_hyper --model ctm --use_hyper \
    --hyper_layers down --log_dir logs/exp3-down --device 0 ...

# Up
python -m tasks.mazes.train_hyper --model ctm --use_hyper \
    --hyper_layers up --log_dir logs/exp3-up --device 0 ...

# All
python -m tasks.mazes.train_hyper --model ctm --use_hyper \
    --hyper_layers all --log_dir logs/exp3-all --device 0 ...
```

## パラメータ数の確認

訓練開始時に以下が表示されます：

```
Using HyperContinuousThoughtMachine with hyper_layers=bottleneck, hyper_rank=8
Using neuron select type: random-pairing
Synch representation size action: 128
Synch representation size out: 128
Total params: 12,345,678
```

## ログの確認

訓練中に生成されるファイル：

```
logs/your-experiment/
├── accuracies.png              # 精度のプロット
├── losses.png                  # 損失のプロット
├── prediction.gif              # 予測の可視化
├── neural_dynamics_other.pdf   # ニューロン動態
├── checkpoint.pt               # モデルチェックポイント
├── args.txt                    # 使用したパラメータ
└── repo_state.zip              # コードのスナップショット
```

## トラブルシューティング

### メモリ不足

```bash
# バッチサイズを減らす
--batch_size 32 \
--batch_size_test 16

# または hyper_layers を bottleneck に
--hyper_layers bottleneck
```

### 学習が不安定

```bash
# Rankを下げる
--hyper_rank 4

# または学習率を下げる
--lr 5e-5
```

### Compile エラー

```bash
# コンパイルを無効化
# --do_compile フラグを削除
```

## 推奨される実験順序

1. **ベースライン確立**
   ```bash
   python -m tasks.mazes.train_hyper --model ctm \
       --dataset mazes-small --training_iterations 10000 \
       --log_dir logs/baseline
   ```

2. **Bottleneck HyperCTM**
   ```bash
   python -m tasks.mazes.train_hyper --model ctm --use_hyper \
       --hyper_layers bottleneck --hyper_rank 8 \
       --dataset mazes-small --training_iterations 10000 \
       --log_dir logs/hyper-bottleneck
   ```

3. **性能が良ければ、より多くの層を試す**
   ```bash
   python -m tasks.mazes.train_hyper --model ctm --use_hyper \
       --hyper_layers all --hyper_rank 8 \
       --dataset mazes-small --training_iterations 10000 \
       --log_dir logs/hyper-all
   ```

4. **Rankの調整**
   - 性能が悪い → rank を上げる (8 → 16)
   - メモリ不足 → rank を下げる (8 → 4)

## 注意事項

1. **synapse_depth**: 
   - `synapse_depth=1`の場合、ハイパーネットワークは使用されません
   - 推奨: `synapse_depth >= 4`

2. **--use_hyper フラグ**:
   - このフラグがないと通常のCTMが使用されます
   - `--hyper_layers`と`--hyper_rank`は`--use_hyper`がある場合のみ有効

3. **デバイス**:
   - `--device 0`: GPU 0を使用
   - `--device -1`: CPUを使用（非推奨、非常に遅い）

4. **チェックポイントからの再開**:
   ```bash
   python -m tasks.mazes.train_hyper ... \
       --reload \
       --log_dir logs/existing-experiment
   ```

## まとめ

- ✅ **初心者**: `--use_hyper --hyper_layers bottleneck --hyper_rank 8`
- ✅ **標準**: `--use_hyper --hyper_layers bottleneck --hyper_rank 8`
- 🚀 **高性能**: `--use_hyper --hyper_layers all --hyper_rank 16`
- 💡 **軽量**: `--use_hyper --hyper_layers bottleneck --hyper_rank 4`
- 📊 **ベースライン**: `--use_hyper`なし（通常のCTM）

