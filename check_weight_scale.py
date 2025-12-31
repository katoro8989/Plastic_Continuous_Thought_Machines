import torch

checkpoint_path = '/workspace/continuous-thought-machines/logs/mazes-small-hyper_initi_bottle_z_large_init_remove_gate_100000/checkpoint.pt'

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
state_dict = checkpoint['model_state_dict']

print(f"\nIteration: {checkpoint['iteration']}")

# 1. 性能指標
print("\n" + "=" * 80)
print("【1】性能指標")
print("=" * 80)

# Full Maze Accuracy
if 'test_accuracies_most_certain_permaze' in checkpoint:
    test_full = checkpoint['test_accuracies_most_certain_permaze']
    train_full = checkpoint['train_accuracies_most_certain_permaze']
    
    print(f"\n■ Full Maze Accuracy:")
    print(f"  Test:  {test_full[-1]:.4f} ({test_full[-1]*100:.2f}%)")
    print(f"  Train: {train_full[-1]:.4f} ({train_full[-1]*100:.2f}%)")
    print(f"  Gap:   {(train_full[-1] - test_full[-1])*100:.2f}%")

# Step-wise Accuracy
if 'test_accuracies_most_certain' in checkpoint:
    test_step = checkpoint['test_accuracies_most_certain']
    train_step = checkpoint['train_accuracies_most_certain']
    
    print(f"\n■ Step-wise Accuracy:")
    print(f"  Test:  {test_step[-1]:.4f} ({test_step[-1]*100:.2f}%)")
    print(f"  Train: {train_step[-1]:.4f} ({train_step[-1]*100:.2f}%)")

# Loss
if 'test_losses' in checkpoint:
    test_loss = checkpoint['test_losses']
    train_loss = checkpoint['train_losses']
    
    print(f"\n■ Loss:")
    print(f"  Test:  {test_loss[-1]:.4f}")
    print(f"  Train: {train_loss[-1]:.4f}")

# 2. 全モデル比較
print("\n" + "=" * 80)
print("【2】全モデル比較")
print("=" * 80)

baseline_test_full = 0.3680
bottle_z_test_full = 0.3820  # Synapseのみ (Gate有)
bottle_z_nlm_small = 0.3800  # Synapse + NLM (small init, Gate有)
bottle_z_nlm_large = 0.0000  # Synapse + NLM (large init, Gate有) - 学習崩壊
bottle_z_nlm_no_gate = 0.2870  # Synapse + NLM (large init, Gate無)

# 3. Gate機構の有無確認
print("\n" + "=" * 80)
print("【3】Gate機構の確認")
print("=" * 80)

nlm_keys = [k for k in state_dict.keys() if 'trace_processor' in k]
nlm_hyper = [k for k in nlm_keys if any(x in k for x in ['head_u', 'head_v', 'head_gate'])]

nlm_gate_keys = [k for k in nlm_hyper if 'head_gate' in k]
synapse_keys = [k for k in state_dict.keys() if 'synapses' in k]
synapse_gate_keys = [k for k in synapse_keys if 'head_gate' in k]

print(f"\n■ NLM:")
if nlm_gate_keys:
    print(f"  Gate あり: {len(nlm_gate_keys)} 個のパラメータ")
    for key in nlm_gate_keys[:3]:
        print(f"    - {key}")
else:
    print(f"  Gate なし ✓")

print(f"\n■ Synapse:")
if synapse_gate_keys:
    print(f"  Gate あり: {len(synapse_gate_keys)} 個のパラメータ")
    for key in synapse_gate_keys[:3]:
        print(f"    - {key}")
else:
    print(f"  Gate なし ✓")

# 4. NLM Hypernetwork分析
print("\n" + "=" * 80)
print("【4】NLM Hypernetwork 分析")
print("=" * 80)

if nlm_hyper:
    print("\n✅ NLMがハイパー化されています")
    
    # U, V の重み
    u_keys = [k for k in nlm_hyper if 'head_u.weight' in k]
    v_keys = [k for k in nlm_hyper if 'head_v.weight' in k]
    
    if u_keys and v_keys:
        u_weight = state_dict[u_keys[0]]
        v_weight = state_dict[v_keys[0]]
        
        print(f"\n■ NLM U/V 重み:")
        print(f"  U shape: {u_weight.shape}")
        print(f"  V shape: {v_weight.shape}")
        print(f"  |U| mean: {u_weight.abs().mean().item():.6f}")
        print(f"  |U| std:  {u_weight.std().item():.6f}")
        print(f"  |V| mean: {v_weight.abs().mean().item():.6f}")
        print(f"  |V| std:  {v_weight.std().item():.6f}")
        
        # 動的成分の推定（Gate無し）
        u_scale = u_weight.abs().mean().item()
        v_scale = v_weight.abs().mean().item()
        dynamic = u_scale * v_scale  # Gate = 1.0 として計算
        
        print(f"\n■ NLM 動的成分（Gate無しと仮定）:")
        print(f"  推定スケール: {dynamic:.8f}")
        print(f"  寄与率: {dynamic*100:.4f}%")
        
        # ベース層との比較
        base_keys = [k for k in nlm_keys if 'w1_base' in k]
        if base_keys:
            base_weight = state_dict[base_keys[0]]
            base_scale = base_weight.abs().mean().item()
            print(f"\n■ ベース層との比較:")
            print(f"  |Base| mean: {base_scale:.6f}")
            print(f"  Dynamic/Base ratio: {dynamic/base_scale:.6f} ({dynamic/base_scale*100:.2f}%)")
else:
    print("\n❌ NLMはハイパー化されていません（Synapseのみのモデル）")

# 5. Synapse Hypernetwork分析
print("\n" + "=" * 80)
print("【5】Synapse Hypernetwork 分析")
print("=" * 80)

synapse_keys = [k for k in state_dict.keys() if 'synapses' in k]
hyper_synapse = [k for k in synapse_keys if any(x in k for x in ['head_u', 'head_v'])]

if hyper_synapse:
    print(f"\n■ Synapse Hypernetwork 寄与率分析 (Gateなし / Static Approximation):")
    
    # Base層、Head U/V のキーを特定
    u_keys = sorted([k for k in hyper_synapse if 'head_u.weight' in k])
    v_keys = sorted([k for k in hyper_synapse if 'head_v.weight' in k])
    
    for u_k, v_k in zip(u_keys, v_keys):
        # Base層のキーを推測 (実装に合わせて調整してください)
        # 例: 'synapses.down...linear.head_u.weight' -> '...linear.base_layer.weight'
        base_k = u_k.replace('head_u.weight', 'base_layer.weight')
        
        if base_k in state_dict:
            # 1. 重みテンソルの取得
            w_base = state_dict[base_k]      # [Out, In]
            w_head_u = state_dict[u_k]       # [Out*Rank, Hidden]
            w_head_v = state_dict[v_k]       # [In*Rank, Hidden]
            
            # 2. スケールの計算 (L2ノルムを使用)
            # Baseの大きさ
            base_norm = w_base.norm().item()
            
            # Dynamicの大きさ (概算)
            # Gateがないので単純に UとVの積
            # ※入力zが単位ベクトル程度の大きさだと仮定
            u_norm = w_head_u.norm().item()
            v_norm = w_head_v.norm().item()
            dynamic_norm = u_norm * v_norm
            
            # 3. 相対比率 (ここが本当の寄与率)
            ratio = dynamic_norm / (base_norm + 1e-9)
            
            layer_name = u_k.split('.linear')[0]
            print(f"  Layer: {layer_name}")
            print(f"    Base Norm : {base_norm:.4f}")
            print(f"    Dyn Norm  : {dynamic_norm:.4f}")
            print(f"    -> 相対寄与率: {ratio*100:.2f}%")
        else:
            print(f"  Warning: Base layer not found for {u_k}")

# 6. スケールファクター確認
print("\n" + "=" * 80)
print("【6】スケールファクター確認")
print("=" * 80)

scale_keys = [k for k in state_dict.keys() if 'scale' in k or 'alpha' in k]

if scale_keys:
    print(f"\n✅ スケールファクター発見:")
    for key in scale_keys:
        param = state_dict[key]
        print(f"  {key}: {param.item() if param.numel() == 1 else param.mean().item():.6f}")
else:
    print(f"\n❌ スケールファクターなし")

# Args確認
if 'args' in checkpoint:
    args = checkpoint['args']
    print("\n" + "=" * 80)
    print("【7】モデル設定")
    print("=" * 80)
    print(f"use_hyper: {getattr(args, 'use_hyper', False)}")
    print(f"hyper_layers: {getattr(args, 'hyper_layers', 'N/A')}")
    print(f"hyper_rank: {getattr(args, 'hyper_rank', 'N/A')}")
    print(f"use_hyper_nlm: {getattr(args, 'use_hyper_nlm', False)}")
    print(f"hyper_nlm_rank: {getattr(args, 'hyper_nlm_rank', 'N/A')}")

# 結論
print("\n" + "=" * 80)
print("【結論】")
print("=" * 80)

delta_baseline = test_full[-1] - baseline_test_full
delta_synapse_gate = test_full[-1] - bottle_z_test_full
delta_nlm_no_gate = test_full[-1] - bottle_z_nlm_no_gate

print(f"\n■ 性能評価:")
if abs(delta_baseline) < 0.005:
    print(f"  vs Baseline: ほぼ同等 (差: {delta_baseline*100:+.2f}%)")
elif delta_baseline > 0:
    print(f"  vs Baseline: ✅ 改善 (+{delta_baseline*100:.2f}%)")
else:
    print(f"  vs Baseline: ❌ 低下 ({delta_baseline*100:.2f}%)")

print(f"  vs Synapse only (Gate有): {delta_synapse_gate*100:+.2f}%")
print(f"  vs Synapse+NLM (no gate): {delta_nlm_no_gate*100:+.2f}%")

# ハイパーネット機能状態
if 'dynamic_syn' in locals():
    print(f"\n■ ハイパーネット機能状態:")
    
    # Synapse
    if dynamic_syn < 0.0001:
        syn_status = "❌ 無効（動的成分 < 0.01%）"
    elif dynamic_syn < 0.001:
        syn_status = "⚠️ 弱い（動的成分 < 0.1%）"
    else:
        syn_status = "✅ 機能（動的成分 >= 0.1%）"
    print(f"  Synapse: {dynamic_syn*100:.4f}% {syn_status}")

    # Gate削除の効果（Synapseのみ）
    print(f"\n■ Gate削除の効果（Synapseのみ）:")
    print(f"  Synapse (Gate有):        Acc 38.20%")
    print(f"  Synapse (no gate):        Acc {test_full[-1]*100:.2f}%")
    print(f"  動的成分: ~{dynamic_syn*100:.4f}%")

del checkpoint