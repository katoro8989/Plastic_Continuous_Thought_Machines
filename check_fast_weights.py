import torch
import numpy as np

checkpoint_path = '/workspace/continuous-thought-machines/logs/cifar10-versus-humans/ctm_bottleneck_fast_weights/d=256--i=64--heads=16--sd=5--synch=256-512-0-h=64-random-pairing--iters=50x15--backbone=18-1--seed=1/checkpoint.pt'

print("=" * 80)
print("Fast Weight 確認スクリプト")
print("=" * 80)

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
state_dict = checkpoint['model_state_dict']

print(f"\nIteration: {checkpoint.get('iteration', 'N/A')}")

# FastWeightSynapseのパラメータを探す
fast_weight_keys = []
for key in state_dict.keys():
    if 'to_u' in key or 'to_v' in key:
        fast_weight_keys.append(key)

if not fast_weight_keys:
    print("\n❌ FastWeightSynapseのパラメータが見つかりませんでした")
    print("   確認したキー:")
    synapse_keys = [k for k in state_dict.keys() if 'synapses' in k]
    if synapse_keys:
        for k in synapse_keys[:10]:
            print(f"     - {k}")
else:
    print(f"\n✅ FastWeightSynapseのパラメータを {len(fast_weight_keys)} 個発見")
    
    # to_u と to_v を分類
    to_u_keys = sorted([k for k in fast_weight_keys if 'to_u' in k])
    to_v_keys = sorted([k for k in fast_weight_keys if 'to_v' in k])
    
    print(f"\n  to_u (Write Value): {len(to_u_keys)} 個")
    print(f"  to_v (Write Key):   {len(to_v_keys)} 個")
    
    # 各FastWeightSynapse層の重みを分析
    print("\n" + "=" * 80)
    print("【Fast Weight 重みの大きさ分析】")
    print("=" * 80)
    
    # base_layerの重みも取得して比較
    base_keys = sorted([k for k in state_dict.keys() if 'base_layer.weight' in k and 'synapses' in k])
    
    # 層ごとに分析
    layer_groups = {}
    for key in to_u_keys + to_v_keys + base_keys:
        # 層名を抽出 (例: 'synapses.down_projections.0.linear.to_u.weight')
        parts = key.split('.')
        if 'down_projections' in parts:
            idx = parts.index('down_projections')
            layer_name = '.'.join(parts[:idx+2])  # 'synapses.down_projections.0'
        elif 'up_projections' in parts:
            idx = parts.index('up_projections')
            layer_name = '.'.join(parts[:idx+2])  # 'synapses.up_projections.0'
        else:
            layer_name = 'unknown'
        
        if layer_name not in layer_groups:
            layer_groups[layer_name] = {}
        layer_groups[layer_name][key] = state_dict[key]
    
    # 各層の統計を表示
    all_to_u_norms = []
    all_to_v_norms = []
    all_base_norms = []
    
    for layer_name, params in sorted(layer_groups.items()):
        print(f"\n■ {layer_name}")
        
        # to_u
        to_u_key = [k for k in params.keys() if 'to_u.weight' in k]
        if to_u_key:
            to_u_weight = params[to_u_key[0]]
            to_u_norm = to_u_weight.norm().item()
            to_u_mean = to_u_weight.abs().mean().item()
            to_u_std = to_u_weight.std().item()
            to_u_max = to_u_weight.abs().max().item()
            all_to_u_norms.append(to_u_norm)
            print(f"  to_u.weight:")
            print(f"    Shape:     {list(to_u_weight.shape)}")
            print(f"    L2 Norm:   {to_u_norm:.6f}")
            print(f"    |Mean|:    {to_u_mean:.6f}")
            print(f"    Std:       {to_u_std:.6f}")
            print(f"    |Max|:     {to_u_max:.6f}")
        
        # to_v
        to_v_key = [k for k in params.keys() if 'to_v.weight' in k]
        if to_v_key:
            to_v_weight = params[to_v_key[0]]
            to_v_norm = to_v_weight.norm().item()
            to_v_mean = to_v_weight.abs().mean().item()
            to_v_std = to_v_weight.std().item()
            to_v_max = to_v_weight.abs().max().item()
            all_to_v_norms.append(to_v_norm)
            print(f"  to_v.weight:")
            print(f"    Shape:     {list(to_v_weight.shape)}")
            print(f"    L2 Norm:   {to_v_norm:.6f}")
            print(f"    |Mean|:    {to_v_mean:.6f}")
            print(f"    Std:       {to_v_std:.6f}")
            print(f"    |Max|:     {to_v_max:.6f}")
        
        # base_layer (比較用)
        base_key = [k for k in params.keys() if 'base_layer.weight' in k]
        if base_key:
            base_weight = params[base_key[0]]
            base_norm = base_weight.norm().item()
            base_mean = base_weight.abs().mean().item()
            all_base_norms.append(base_norm)
            print(f"  base_layer.weight:")
            print(f"    Shape:     {list(base_weight.shape)}")
            print(f"    L2 Norm:   {base_norm:.6f}")
            print(f"    |Mean|:    {base_mean:.6f}")
            
            # Fast Weightの相対的な大きさ
            if to_u_key and to_v_key:
                # Fast weightの寄与の推定: to_uとto_vのノルムの積
                fast_weight_scale = to_u_norm * to_v_norm
                ratio = fast_weight_scale / (base_norm + 1e-9)
                print(f"  Fast Weight スケール推定:")
                print(f"    to_u_norm × to_v_norm: {fast_weight_scale:.6f}")
                print(f"    Base層との比率:        {ratio:.4f} ({ratio*100:.2f}%)")
    
    # 全体統計
    print("\n" + "=" * 80)
    print("【全体統計】")
    print("=" * 80)
    
    if all_to_u_norms:
        print(f"\n■ to_u (全層):")
        print(f"  平均 L2 Norm: {np.mean(all_to_u_norms):.6f}")
        print(f"  最小 L2 Norm: {np.min(all_to_u_norms):.6f}")
        print(f"  最大 L2 Norm: {np.max(all_to_u_norms):.6f}")
    
    if all_to_v_norms:
        print(f"\n■ to_v (全層):")
        print(f"  平均 L2 Norm: {np.mean(all_to_v_norms):.6f}")
        print(f"  最小 L2 Norm: {np.min(all_to_v_norms):.6f}")
        print(f"  最大 L2 Norm: {np.max(all_to_v_norms):.6f}")
    
    if all_base_norms:
        print(f"\n■ base_layer (全層):")
        print(f"  平均 L2 Norm: {np.mean(all_base_norms):.6f}")
        print(f"  最小 L2 Norm: {np.min(all_base_norms):.6f}")
        print(f"  最大 L2 Norm: {np.max(all_base_norms):.6f}")
    
    # alpha_scale パラメータの確認
    print("\n" + "=" * 80)
    print("【Fast Weight ハイパーパラメータ】")
    print("=" * 80)
    
    # FastWeightSynapseのパラメータを探す
    alpha_keys = [k for k in state_dict.keys() if 'alpha' in k and 'synapses' in k]
    lambda_keys = [k for k in state_dict.keys() if 'lambda' in k and 'synapses' in k]
    eta_keys = [k for k in state_dict.keys() if 'eta' in k and 'synapses' in k]
    
    if alpha_keys:
        print(f"\n✅ alpha_scale パラメータ発見:")
        for key in alpha_keys:
            param = state_dict[key]
            if param.numel() == 1:
                print(f"  {key}: {param.item():.6f}")
            else:
                print(f"  {key}: shape={list(param.shape)}, mean={param.mean().item():.6f}")
    else:
        print(f"\n⚠️  alpha_scale パラメータが見つかりません（デフォルト値を使用している可能性）")
        print(f"    コードでは alpha_scale=1 が設定されています")
    
    if lambda_keys:
        print(f"\n✅ lambda_decay パラメータ発見:")
        for key in lambda_keys:
            param = state_dict[key]
            if param.numel() == 1:
                print(f"  {key}: {param.item():.6f}")
    
    if eta_keys:
        print(f"\n✅ eta_learning_rate パラメータ発見:")
        for key in eta_keys:
            param = state_dict[key]
            if param.numel() == 1:
                print(f"  {key}: {param.item():.6f}")
    
    # 結論
    print("\n" + "=" * 80)
    print("【結論】")
    print("=" * 80)
    
    if all_to_u_norms and all_to_v_norms:
        avg_to_u = np.mean(all_to_u_norms)
        avg_to_v = np.mean(all_to_v_norms)
        avg_fast_scale = avg_to_u * avg_to_v
        
        print(f"\n■ Fast Weight の状態:")
        print(f"  平均 to_u L2 Norm: {avg_to_u:.6f}")
        print(f"  平均 to_v L2 Norm: {avg_to_v:.6f}")
        print(f"  推定 Fast Weight スケール: {avg_fast_scale:.6f}")
        
        # 重みが十分に大きいか判定
        # 一般的に、L2ノルムが0.01以上あれば機能していると判断
        threshold = 0.01
        
        if avg_to_u > threshold and avg_to_v > threshold:
            print(f"\n✅ Fast Weight は十分に大きい値を持っています")
            print(f"   (to_u: {avg_to_u:.6f} > {threshold}, to_v: {avg_to_v:.6f} > {threshold})")
        elif avg_to_u > threshold * 0.1 or avg_to_v > threshold * 0.1:
            print(f"\n⚠️  Fast Weight は小さいですが、機能している可能性があります")
            print(f"   (to_u: {avg_to_u:.6f}, to_v: {avg_to_v:.6f})")
        else:
            print(f"\n❌ Fast Weight が非常に小さいです。機能していない可能性があります")
            print(f"   (to_u: {avg_to_u:.6f}, to_v: {avg_to_v:.6f})")
        
        if all_base_norms:
            avg_base = np.mean(all_base_norms)
            relative_scale = avg_fast_scale / (avg_base + 1e-9)
            print(f"\n■ Base層との比較:")
            print(f"  平均 Base L2 Norm: {avg_base:.6f}")
            print(f"  Fast Weight / Base 比率: {relative_scale:.4f} ({relative_scale*100:.2f}%)")

del checkpoint




