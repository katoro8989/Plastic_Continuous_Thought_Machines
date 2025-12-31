"""
HyperContinuousThoughtMachine の動作確認テスト

このスクリプトは、HyperContinuousThoughtMachineが正しく動作するかを確認します。
"""

import torch
from models.ctm import ContinuousThoughtMachine, HyperContinuousThoughtMachine


def count_parameters(model):
    """モデルのパラメータ数を計算"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_hyper_ctm():
    """HyperContinuousThoughtMachineのテスト"""
    
    print("=" * 60)
    print("HyperContinuousThoughtMachine テスト")
    print("=" * 60)
    
    # 共通パラメータ
    common_params = {
        'iterations': 10,
        'd_model': 128,
        'd_input': 64,
        'heads': 2,
        'n_synch_out': 32,
        'n_synch_action': 32,
        'synapse_depth': 4,  # depth >= 2 でUNET使用
        'memory_length': 10,
        'deep_nlms': True,
        'memory_hidden_dims': 16,
        'do_layernorm_nlm': False,
        'backbone_type': 'none',
        'positional_embedding_type': 'none',
        'out_dims': 50,
        'neuron_select_type': 'random-pairing',
        'dropout': 0.0,
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # テスト1: 通常のCTM
    print("\n" + "-" * 60)
    print("テスト1: 通常のCTM (baseline)")
    print("-" * 60)
    
    model_baseline = ContinuousThoughtMachine(**common_params).to(device)
    baseline_params = count_parameters(model_baseline)
    print(f"パラメータ数: {baseline_params:,}")
    
    # ダミー入力
    batch_size = 2
    x = torch.randn(batch_size, 3, 32, 32).to(device)
    
    with torch.no_grad():
        pred_base, cert_base, sync_base = model_baseline(x)
    
    print(f"出力形状:")
    print(f"  - predictions: {pred_base.shape}")
    print(f"  - certainties: {cert_base.shape}")
    print(f"  - synchronisation: {sync_base.shape}")
    
    # テスト2: Bottleneck ハイパー化
    print("\n" + "-" * 60)
    print("テスト2: HyperCTM (hyper_layers='bottleneck', rank=4)")
    print("-" * 60)
    
    model_hyper_bottleneck = HyperContinuousThoughtMachine(
        **common_params,
        hyper_layers='bottleneck',
        hyper_rank=4
    ).to(device)
    
    hyper_bottleneck_params = count_parameters(model_hyper_bottleneck)
    print(f"パラメータ数: {hyper_bottleneck_params:,}")
    print(f"増加: {hyper_bottleneck_params - baseline_params:,} " 
          f"({(hyper_bottleneck_params / baseline_params - 1) * 100:.2f}%)")
    
    with torch.no_grad():
        pred_hyper, cert_hyper, sync_hyper = model_hyper_bottleneck(x)
    
    print(f"出力形状:")
    print(f"  - predictions: {pred_hyper.shape}")
    print(f"  - certainties: {cert_hyper.shape}")
    print(f"  - synchronisation: {sync_hyper.shape}")
    
    # テスト3: すべての層をハイパー化
    print("\n" + "-" * 60)
    print("テスト3: HyperCTM (hyper_layers='all', rank=8)")
    print("-" * 60)
    
    model_hyper_all = HyperContinuousThoughtMachine(
        **common_params,
        hyper_layers='all',
        hyper_rank=8
    ).to(device)
    
    hyper_all_params = count_parameters(model_hyper_all)
    print(f"パラメータ数: {hyper_all_params:,}")
    print(f"増加: {hyper_all_params - baseline_params:,} "
          f"({(hyper_all_params / baseline_params - 1) * 100:.2f}%)")
    
    with torch.no_grad():
        pred_all, cert_all, sync_all = model_hyper_all(x)
    
    print(f"出力形状:")
    print(f"  - predictions: {pred_all.shape}")
    print(f"  - certainties: {cert_all.shape}")
    print(f"  - synchronisation: {sync_all.shape}")
    
    # テスト4: hyper_layers='none' (通常のSynapseUNETと同じ)
    print("\n" + "-" * 60)
    print("テスト4: HyperCTM (hyper_layers='none') - 通常と同じはず")
    print("-" * 60)
    
    model_hyper_none = HyperContinuousThoughtMachine(
        **common_params,
        hyper_layers='none',
        hyper_rank=8
    ).to(device)
    
    hyper_none_params = count_parameters(model_hyper_none)
    print(f"パラメータ数: {hyper_none_params:,}")
    print(f"差分: {hyper_none_params - baseline_params:,}")
    
    if hyper_none_params == baseline_params:
        print("✅ 'none'設定時は通常のCTMと同じパラメータ数です")
    else:
        print(f"⚠️  パラメータ数が異なります (差: {hyper_none_params - baseline_params:,})")
    
    # 勾配計算テスト
    print("\n" + "-" * 60)
    print("テスト5: 勾配計算 (backward pass)")
    print("-" * 60)
    
    model_hyper_bottleneck.train()
    x_train = torch.randn(batch_size, 3, 32, 32, requires_grad=True).to(device)
    
    pred, cert, _ = model_hyper_bottleneck(x_train)
    loss = pred.mean()
    loss.backward()
    
    # ハイパーネットワークのパラメータに勾配がついているか確認
    has_hyper_grads = False
    for name, param in model_hyper_bottleneck.named_parameters():
        if 'hyper' in name.lower() or 'head_' in name or 'context_net' in name:
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_hyper_grads = True
                print(f"✅ ハイパーネットワークのパラメータに勾配: {name[:50]}...")
                break
    
    if has_hyper_grads:
        print("✅ 勾配計算: 正常")
    else:
        print("⚠️  ハイパーネットワークのパラメータに勾配が確認できませんでした")
    
    # まとめ
    print("\n" + "=" * 60)
    print("テスト結果まとめ")
    print("=" * 60)
    print(f"通常のCTM:                {baseline_params:>12,} パラメータ")
    print(f"HyperCTM (bottleneck):    {hyper_bottleneck_params:>12,} パラメータ " 
          f"(+{hyper_bottleneck_params - baseline_params:,})")
    print(f"HyperCTM (all):           {hyper_all_params:>12,} パラメータ "
          f"(+{hyper_all_params - baseline_params:,})")
    print(f"HyperCTM (none):          {hyper_none_params:>12,} パラメータ "
          f"(+{hyper_none_params - baseline_params:,})")
    print("\n✅ すべてのテストが完了しました！")
    print("=" * 60)


if __name__ == '__main__':
    test_hyper_ctm()

