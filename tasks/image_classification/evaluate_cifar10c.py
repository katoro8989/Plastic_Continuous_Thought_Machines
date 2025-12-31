"""
CIFAR-10Cでチェックポイントを評価するスクリプト
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
import pickle

# プロジェクトのルートディレクトリをパスに追加
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.ctm import ContinuousThoughtMachine, HyperContinuousThoughtMachine
from models.lstm import LSTMBaseline
from models.ff import FFBaseline
from utils.losses import image_classification_loss
from utils.housekeeping import set_seed


class CIFAR10C(Dataset):
    """
    CIFAR-10Cデータセットを読み込むクラス
    CIFAR-10Cは、CIFAR-10のテストセットに様々なコルプションを加えたデータセット
    """
    def __init__(self, root, corruption_type, severity, transform=None):
        """
        Args:
            root: CIFAR-10Cデータセットのルートディレクトリ
            corruption_type: コルプションタイプ (例: 'gaussian_noise', 'fog', 'brightness'など)
            severity: 強度レベル (1-5)
            transform: 画像変換
        """
        self.root = root
        self.corruption_type = corruption_type
        self.severity = severity
        self.transform = transform
        
        # CIFAR-10Cデータファイルのパス
        data_path = os.path.join(root, 'CIFAR-10-C', f'{corruption_type}.npy')
        labels_path = os.path.join(root, 'CIFAR-10-C', 'labels.npy')
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"CIFAR-10Cデータが見つかりません: {data_path}\n"
                f"CIFAR-10Cデータセットをダウンロードして、{root}/CIFAR-10-C/ に配置してください。\n"
                f"ダウンロード先: https://zenodo.org/record/2535967"
            )
        
        # データを読み込む
        self.data = np.load(data_path)
        self.labels = np.load(labels_path)
        
        # 強度レベルに応じてデータをフィルタリング
        # CIFAR-10Cは各強度レベルごとに10000枚の画像がある
        # データは [50000, 32, 32, 3] の形状で、最初の10000がseverity=1, 次の10000がseverity=2, ...
        start_idx = (severity - 1) * 10000
        end_idx = severity * 10000
        self.data = self.data[start_idx:end_idx]
        self.labels = self.labels[start_idx:end_idx]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        
        # numpy配列をPIL Imageに変換
        img = Image.fromarray(img.astype(np.uint8))
        
        if self.transform:
            img = self.transform(img)
        
        # ラベルをLong型（int64）に変換（CrossEntropyLossはLong型を期待）
        label = int(label)
        
        return img, label


def get_cifar10c_dataset(root, corruption_type, severity):
    """CIFAR-10Cデータセットを取得"""
    dataset_mean = [0.49139968, 0.48215827, 0.44653124]
    dataset_std = [0.24703233, 0.24348505, 0.26158768]
    normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    dataset = CIFAR10C(root, corruption_type, severity, transform=test_transform)
    class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    return dataset, class_labels, dataset_mean, dataset_std


def parse_args_from_file(args_path):
    """args.txtファイルからNamespaceオブジェクトを構築"""
    with open(args_path, 'r') as f:
        content = f.read().strip()
    
    # Namespace(attr1=value1, attr2=value2, ...)の形式をパース
    import re
    args = argparse.Namespace()
    
    # Namespace(の部分を除去
    content = re.sub(r'^Namespace\(', '', content)
    content = re.sub(r'\)$', '', content)
    
    # カンマで分割（ただし、文字列内のカンマは無視）
    # 簡易的なパース: key='value'またはkey=valueの形式
    pattern = r"(\w+)=([^,]+?)(?=,\s*\w+=|$)"
    matches = re.findall(pattern, content)
    
    for key, value in matches:
        key = key.strip()
        value = value.strip()
        
        # 文字列のクォートを除去
        if value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        elif value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        
        # 型を推測
        if value == 'True':
            setattr(args, key, True)
        elif value == 'False':
            setattr(args, key, False)
        elif value == 'None':
            setattr(args, key, None)
        elif value.startswith('[') and value.endswith(']'):
            # リストの場合（簡易パース）
            try:
                # 安全にeval（数値リストのみ）
                inner = value[1:-1].strip()
                if inner:
                    items = [item.strip() for item in inner.split(',')]
                    parsed_items = []
                    for item in items:
                        if item.isdigit():
                            parsed_items.append(int(item))
                        elif item.replace('.', '').replace('-', '').isdigit():
                            parsed_items.append(float(item))
                        else:
                            parsed_items.append(item.strip("'\" "))
                    setattr(args, key, parsed_items)
                else:
                    setattr(args, key, [])
            except:
                setattr(args, key, value)
        elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            setattr(args, key, int(value))
        elif '.' in value and value.replace('.', '').replace('-', '').isdigit():
            setattr(args, key, float(value))
        else:
            setattr(args, key, value)
    
    return args


def load_model_from_checkpoint(checkpoint_path, device):
    """チェックポイントからモデルを読み込む"""
    print(f'チェックポイントを読み込み中: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # fast_weightsチェックポイントかどうかを判定（最初に判定）
    checkpoint_path_lower = checkpoint_path.lower()
    use_fast_weights = 'fast_weight' in checkpoint_path_lower
    
    # argsを取得（チェックポイントに保存されている場合）
    if 'args' in checkpoint:
        args = checkpoint['args']
        if hasattr(args, 'log_dir'):
            use_fast_weights = use_fast_weights or 'fast_weight' in args.log_dir.lower()
    else:
        # args.txtから読み込む
        args_path = os.path.join(os.path.dirname(checkpoint_path), 'args.txt')
        if os.path.exists(args_path):
            args = parse_args_from_file(args_path)
            if hasattr(args, 'log_dir'):
                use_fast_weights = use_fast_weights or 'fast_weight' in args.log_dir.lower()
        else:
            raise ValueError(f"args.txtが見つかりません: {args_path}")
    
    # モデルを構築
    prediction_reshaper = [-1]
    args.out_dims = 10  # CIFAR-10
    
    if args.model == 'ctm':
        ctm_kwargs = {
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
            'dropout_nlm': args.dropout_nlm if hasattr(args, 'dropout_nlm') else None,
            'neuron_select_type': args.neuron_select_type,
            'n_random_pairing_self': args.n_random_pairing_self,
        }
        
        if hasattr(args, 'use_hyper') and args.use_hyper:
            ctm_kwargs['hyper_layers'] = args.hyper_layers if hasattr(args, 'hyper_layers') else 'bottleneck'
            ctm_kwargs['hyper_rank'] = args.hyper_rank if hasattr(args, 'hyper_rank') else 8
            ctm_kwargs['use_hyper_nlm'] = args.use_hyper_nlm if hasattr(args, 'use_hyper_nlm') else False
            ctm_kwargs['hyper_nlm_rank'] = args.hyper_nlm_rank if hasattr(args, 'hyper_nlm_rank') else 4
            
            # fast_weightsを使用する場合は、HyperContinuousThoughtMachineにuse_fast_weightsを設定
            # ただし、現在の実装ではHyperFastWeightSynapseUNETを直接使う必要がある
            # 一時的な解決策として、モデル構築後にsynapsesを置き換える
            model = HyperContinuousThoughtMachine(**ctm_kwargs).to(device)
            
            # fast_weightsを使用する場合、synapsesをHyperFastWeightSynapseUNETに置き換え
            if use_fast_weights:
                from models.modules import HyperFastWeightSynapseUNET, HyperSynapseUNET
                print("Fast weightsを使用するため、synapsesを置き換えます。")
                
                # チェックポイントからfirst_projectionの入力サイズを確認
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                first_proj_key = 'synapses.first_projection.0.weight'
                if first_proj_key in checkpoint['model_state_dict']:
                    first_proj_weight = checkpoint['model_state_dict'][first_proj_key]
                    input_dim = first_proj_weight.shape[1]  # [out_dim, in_dim]
                    print(f"チェックポイントからfirst_projectionの入力サイズを検出: {input_dim}")
                else:
                    # デフォルトはd_model
                    input_dim = args.d_model
                    print(f"チェックポイントにfirst_projectionが見つからないため、d_model={input_dim}を使用します。")
                
                # チェックポイントの構造を検出（head_u/head_vとcontext_netがあるか）
                checkpoint_keys = list(checkpoint['model_state_dict'].keys())
                has_head_u = any('head_u' in k and 'down_projections.3.linear' in k for k in checkpoint_keys)
                has_context_net = any('context_net' in k and 'down_projections.3.linear' in k for k in checkpoint_keys)
                has_to_u = any('to_u' in k and 'down_projections.3.linear' in k for k in checkpoint_keys)
                
                if has_head_u and has_context_net:
                    # HyperLoRALinear構造（seed=2,3）
                    print("チェックポイントの構造を検出: HyperLoRALinear構造（head_u/head_v + context_net）")
                    print("HyperSynapseUNETを使用します（HyperLoRALinear構造に対応）。")
                    new_synapses = HyperSynapseUNET(
                        out_dims=args.d_model,
                        depth=args.synapse_depth,
                        hyper_layers=args.hyper_layers if hasattr(args, 'hyper_layers') else 'bottleneck',
                        hyper_rank=args.hyper_rank if hasattr(args, 'hyper_rank') else 8,
                        minimum_width=16,
                        dropout=args.dropout
                    ).to(device)
                elif has_to_u:
                    # FastWeightSynapse構造（seed=1）
                    print("チェックポイントの構造を検出: FastWeightSynapse構造（to_u/to_v）")
                    new_synapses = HyperFastWeightSynapseUNET(
                        out_dims=args.d_model,
                        depth=args.synapse_depth,
                        hyper_layers=args.hyper_layers if hasattr(args, 'hyper_layers') else 'bottleneck',
                        hyper_rank=args.hyper_rank if hasattr(args, 'hyper_rank') else 8,
                        minimum_width=16,
                        dropout=args.dropout
                    ).to(device)
                else:
                    # デフォルト
                    print("チェックポイントの構造を検出できませんでした。デフォルトのHyperFastWeightSynapseUNETを使用します。")
                    new_synapses = HyperFastWeightSynapseUNET(
                        out_dims=args.d_model,
                        depth=args.synapse_depth,
                        hyper_layers=args.hyper_layers if hasattr(args, 'hyper_layers') else 'bottleneck',
                        hyper_rank=args.hyper_rank if hasattr(args, 'hyper_rank') else 8,
                        minimum_width=16,
                        dropout=args.dropout
                    ).to(device)
                
                # LazyLinearを正しい入力サイズで初期化
                dummy_input = torch.randn(1, input_dim).to(device)
                with torch.no_grad():
                    _ = new_synapses(dummy_input)
                model.synapses = new_synapses
        else:
            model = ContinuousThoughtMachine(**ctm_kwargs).to(device)
            
    elif args.model == 'lstm':
        model = LSTMBaseline(
            num_layers=args.num_layers,
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            backbone_type=args.backbone_type,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=args.dropout,
        ).to(device)
    elif args.model == 'ff':
        model = FFBaseline(
            d_model=args.d_model,
            backbone_type=args.backbone_type,
            out_dims=args.out_dims,
            dropout=args.dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    # 未初期化パラメータを初期化（LazyLinearなど）
    try:
        # ダミー入力でforwardを実行して初期化
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        with torch.no_grad():
            _ = model(dummy_input)
    except Exception as e:
        print(f"警告: ダミー入力での初期化に失敗しました: {e}")
        print("一部のパラメータが未初期化の可能性があります。")
    
    # モデルの重みを読み込む
    # fast_weightsなどの異なるモデル構造に対応するため、strict=Falseで試す
    use_strict = True
    
    # fast_weightsチェックポイントの場合、サイズが一致しないパラメータをスキップ
    checkpoint_state_dict = checkpoint['model_state_dict'].copy()
    if use_fast_weights:
        # モデルのstate_dictを取得してサイズを確認
        model_state_dict = model.state_dict()
        
        # サイズが一致しないパラメータを削除
        keys_to_remove = []
        for key in list(checkpoint_state_dict.keys()):
            if key in model_state_dict:
                try:
                    checkpoint_shape = checkpoint_state_dict[key].shape
                    model_shape = model_state_dict[key].shape
                    if checkpoint_shape != model_shape:
                        # 特にfast weight関連のパラメータでサイズ不一致の場合
                        if 'head_u' in key or 'head_v' in key or 'to_u' in key or 'to_v' in key:
                            keys_to_remove.append(key)
                            print(f"⚠️  サイズ不一致のためスキップ: {key} (チェックポイント: {checkpoint_shape}, モデル: {model_shape})")
                except RuntimeError:
                    # 未初期化パラメータの場合はスキップ
                    if 'head_u' in key or 'head_v' in key or 'to_u' in key or 'to_v' in key:
                        keys_to_remove.append(key)
                        print(f"⚠️  未初期化パラメータのためスキップ: {key}")
        
        for key in keys_to_remove:
            checkpoint_state_dict.pop(key, None)
        
        # head_u/head_vをto_u/to_vにマッピング（サイズが一致する場合のみ）
        key_mapping = {}
        for key in list(checkpoint_state_dict.keys()):
            if 'head_u' in key:
                new_key = key.replace('head_u', 'to_u')
                # サイズが一致する場合のみマッピング
                if new_key in model_state_dict:
                    try:
                        checkpoint_shape = checkpoint_state_dict[key].shape
                        model_shape = model_state_dict[new_key].shape
                        if checkpoint_shape == model_shape:
                            key_mapping[key] = new_key
                    except RuntimeError:
                        pass  # 未初期化パラメータの場合はスキップ
            elif 'head_v' in key:
                new_key = key.replace('head_v', 'to_v')
                # サイズが一致する場合のみマッピング
                if new_key in model_state_dict:
                    try:
                        checkpoint_shape = checkpoint_state_dict[key].shape
                        model_shape = model_state_dict[new_key].shape
                        if checkpoint_shape == model_shape:
                            key_mapping[key] = new_key
                    except RuntimeError:
                        pass  # 未初期化パラメータの場合はスキップ
        
        # マッピングを適用
        for old_key, new_key in key_mapping.items():
            if old_key in checkpoint_state_dict:
                checkpoint_state_dict[new_key] = checkpoint_state_dict.pop(old_key)
                print(f"キー名をマッピング: {old_key} -> {new_key}")
    
    try:
        load_result = model.load_state_dict(checkpoint_state_dict, strict=True)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(f"警告: 一部のキーが一致しませんでした。")
            if load_result.missing_keys:
                print(f"  不足しているキー: {load_result.missing_keys[:5]}... (合計{len(load_result.missing_keys)}個)")
            if load_result.unexpected_keys:
                print(f"  予期しないキー: {load_result.unexpected_keys[:5]}... (合計{len(load_result.unexpected_keys)}個)")
    except RuntimeError as e:
        # strict=Trueで失敗した場合、strict=Falseで再試行
        use_strict = False
        print(f"警告: strict=Trueで読み込みに失敗しました。strict=Falseで再試行します。")
        print(f"エラー詳細: {str(e)[:200]}...")
        load_result = model.load_state_dict(checkpoint_state_dict, strict=False)
        if load_result.missing_keys:
            print(f"警告: 以下のキーが読み込まれませんでした: {load_result.missing_keys[:10]}... (合計{len(load_result.missing_keys)}個)")
        if load_result.unexpected_keys:
            print(f"警告: 以下のキーが無視されました: {load_result.unexpected_keys[:10]}... (合計{len(load_result.unexpected_keys)}個)")
    
    # fast_weightsが正しく読み込まれたか確認
    checkpoint_path_lower = checkpoint_path.lower()
    log_dir_lower = args.log_dir.lower() if hasattr(args, 'log_dir') else ''
    use_fast_weights_check = 'fast_weight' in checkpoint_path_lower or 'fast_weight' in log_dir_lower
    
    if use_fast_weights_check:
        # チェックポイント内のfast weightパラメータを確認
        checkpoint_fast_weight_keys = [k for k in checkpoint['model_state_dict'].keys() 
                                       if 'head_u' in k or 'head_v' in k or 'to_u' in k or 'to_v' in k]
        
        # モデル内のfast weightパラメータを確認
        model_fast_weight_keys = []
        for name, param in model.named_parameters():
            # HyperFastWeightSynapseUNETはhead_uとhead_vを使用
            if 'head_u' in name or 'head_v' in name or 'to_u' in name or 'to_v' in name:
                model_fast_weight_keys.append(name)
        
        if checkpoint_fast_weight_keys:
            print(f"チェックポイント内のfast weightパラメータ: {len(checkpoint_fast_weight_keys)}個")
            print(f"   例: {checkpoint_fast_weight_keys[0] if checkpoint_fast_weight_keys else 'N/A'}")
        
        if model_fast_weight_keys:
            print(f"✅ モデル内のfast weightパラメータ: {len(model_fast_weight_keys)}個")
            print(f"   例: {model_fast_weight_keys[0] if model_fast_weight_keys else 'N/A'}")
            
            # パラメータがゼロでないことを確認
            non_zero_count = 0
            for name in model_fast_weight_keys[:3]:  # 最初の3つだけ確認
                param = dict(model.named_parameters())[name]
                if param.abs().sum() > 1e-6:
                    non_zero_count += 1
            if non_zero_count > 0:
                print(f"✅ Fast weightsパラメータは非ゼロ値を持っています（{non_zero_count}/3確認済み）")
            else:
                print(f"⚠️  Fast weightsパラメータがゼロの可能性があります")
            
            # チェックポイントから読み込まれたか確認
            if not use_strict and load_result.missing_keys:
                missing_fast_weights = [k for k in load_result.missing_keys 
                                       if 'head_u' in k or 'head_v' in k or 'to_u' in k or 'to_v' in k]
                if missing_fast_weights:
                    print(f"⚠️  以下のfast weightパラメータが読み込まれませんでした: {missing_fast_weights[:5]}... (合計{len(missing_fast_weights)}個)")
                else:
                    print(f"✅ すべてのfast weightパラメータが読み込まれました")
            
            # チェックポイントとモデルのパラメータ値を比較して、実際に読み込まれているか確認
            if checkpoint_fast_weight_keys and model_fast_weight_keys:
                # チェックポイントのキー名をモデルのキー名に変換
                checkpoint_to_model_key = {}
                for ckpt_key in checkpoint_fast_weight_keys:
                    if 'head_u' in ckpt_key:
                        model_key = ckpt_key.replace('head_u', 'head_u')  # そのまま
                    elif 'head_v' in ckpt_key:
                        model_key = ckpt_key.replace('head_v', 'head_v')  # そのまま
                    elif 'to_u' in ckpt_key:
                        model_key = ckpt_key.replace('to_u', 'to_u')  # そのまま
                    elif 'to_v' in ckpt_key:
                        model_key = ckpt_key.replace('to_v', 'to_v')  # そのまま
                    else:
                        continue
                    checkpoint_to_model_key[ckpt_key] = model_key
                
                # 一致するキーでパラメータ値を比較
                matched_count = 0
                for ckpt_key, model_key in checkpoint_to_model_key.items():
                    if model_key in model_fast_weight_keys:
                        try:
                            ckpt_param = checkpoint['model_state_dict'][ckpt_key]
                            model_param = dict(model.named_parameters())[model_key]
                            # サイズが一致する場合のみ比較
                            if ckpt_param.shape == model_param.shape:
                                # 値が一致しているか確認（浮動小数点誤差を考慮）
                                if torch.allclose(ckpt_param.cpu(), model_param.cpu(), atol=1e-5):
                                    matched_count += 1
                        except (KeyError, RuntimeError):
                            pass
                
                if matched_count > 0:
                    print(f"✅ チェックポイントから{matched_count}個のfast weightパラメータが正しく読み込まれました")
                elif len(checkpoint_to_model_key) > 0:
                    print(f"⚠️  fast weightパラメータの値が一致していません（サイズ不一致の可能性）")
        else:
            print(f"⚠️  モデル内にfast weightパラメータが見つかりませんでした。")
    
    # 未初期化パラメータがある場合、ダミー入力でforwardを実行して初期化
    # これはLazyModuleやstrict=Falseで一部パラメータが読み込まれなかった場合に必要
    try:
        # パラメータ数を計算して、未初期化パラメータがないか確認
        _ = sum(p.numel() for p in model.parameters())
    except (ValueError, RuntimeError):
        # 未初期化パラメータがある場合、ダミー入力で初期化
        print("未初期化パラメータを検出しました。ダミー入力で初期化します...")
        # CIFAR-10の画像サイズに合わせたダミー入力を作成
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        with torch.no_grad():
            try:
                _ = model(dummy_input)
            except Exception as e:
                print(f"警告: ダミー入力での初期化に失敗しました: {e}")
                print("評価を続行しますが、一部のパラメータが未初期化の可能性があります。")
    
    model.eval()
    
    return model, args


def evaluate_model(model, dataloader, device, model_type):
    """モデルを評価"""
    all_targets = []
    all_predictions = []
    all_predictions_most_certain = []
    all_losses = []
    
    with torch.inference_mode():
        for inputs, targets in tqdm(dataloader, desc='評価中'):
            inputs = inputs.to(device)
            # ラベルをLong型に変換（CrossEntropyLossはLong型を期待）
            targets = targets.long().to(device)
            all_targets.append(targets.detach().cpu().numpy())
            
            if model_type in ['ctm', 'lstm']:
                predictions, certainties, _ = model(inputs)
                loss, where_most_certain = image_classification_loss(
                    predictions, certainties, targets, use_most_certain=True
                )
                all_predictions.append(predictions.argmax(1).detach().cpu().numpy())
                all_predictions_most_certain.append(
                    predictions.argmax(1)[
                        torch.arange(predictions.size(0), device=predictions.device),
                        where_most_certain
                    ].detach().cpu().numpy()
                )
            else:  # ff
                predictions = model(inputs)
                loss = nn.CrossEntropyLoss()(predictions, targets)
                all_predictions.append(predictions.argmax(1).detach().cpu().numpy())
            
            all_losses.append(loss.item())
    
    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)
    
    if model_type in ['ctm', 'lstm']:
        # 各tickでの精度を計算
        accuracies_per_tick = np.mean(all_predictions == all_targets[..., np.newaxis], axis=0)
        all_predictions_most_certain = np.concatenate(all_predictions_most_certain)
        accuracy_most_certain = (all_targets == all_predictions_most_certain).mean()
        return {
            'loss': np.mean(all_losses),
            'accuracy_most_certain': accuracy_most_certain,
            'accuracies_per_tick': accuracies_per_tick,
        }
    else:
        accuracy = (all_targets == all_predictions).mean()
        return {
            'loss': np.mean(all_losses),
            'accuracy': accuracy,
        }


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10Cでチェックポイントを評価')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='チェックポイントファイルのパス')
    parser.add_argument('--data_root', type=str, default='data/',
                        help='データセットのルートディレクトリ')
    parser.add_argument('--corruption_type', type=str, default=None,
                        help='コルプションタイプ (指定しない場合は全タイプを評価)')
    parser.add_argument('--severity', type=int, default=None,
                        help='強度レベル (1-5, 指定しない場合は全レベルを評価)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='バッチサイズ (デフォルト: GPU使用時512, CPU使用時32)')
    parser.add_argument('--device', type=int, default=None,
                        help='使用するGPUデバイス (デフォルト: 利用可能な最初のGPU、なければCPU)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoaderのワーカー数 (デフォルト: 0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='ランダムシード')
    
    args = parser.parse_args()
    
    set_seed(args.seed, False)
    
    # デバイス設定
    if args.device is not None:
        if args.device == -1:
            device = 'cpu'
        else:
            device = f'cuda:{args.device}'
    elif torch.cuda.is_available():
        device = 'cuda:0'
        print(f'GPUが利用可能です。デバイス0を使用します。')
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
        print(f'警告: CPUモードで実行します。GPUを使用する場合は --device 0 を指定してください。')
    print(f'デバイス: {device}')
    
    # バッチサイズの設定
    if args.batch_size is None:
        if 'cuda' in device or device == 'mps':
            args.batch_size = 512
        else:
            args.batch_size = 32
        print(f'バッチサイズ: {args.batch_size} (自動設定)')
    else:
        print(f'バッチサイズ: {args.batch_size}')
    
    # モデルを読み込む
    model, model_args = load_model_from_checkpoint(args.checkpoint, device)
    print(f'モデルタイプ: {model_args.model}')
    print(f'総パラメータ数: {sum(p.numel() for p in model.parameters())}')
    
    # CIFAR-10Cのコルプションタイプ
    corruption_types = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    
    severities = [5]
    
    # 評価するコルプションタイプと強度を決定
    if args.corruption_type:
        corruption_types = [args.corruption_type]
    if args.severity:
        severities = [args.severity]
    
    results = {}
    
    # データセットの存在確認（最初のコルプションタイプで確認）
    first_data_path = os.path.join(args.data_root, 'CIFAR-10-C', f'{corruption_types[0]}.npy')
    if not os.path.exists(first_data_path):
        print(f'\nエラー: CIFAR-10Cデータセットが見つかりません: {first_data_path}')
        print(f'データセットをダウンロードして、{args.data_root}/CIFAR-10-C/ に配置してください。')
        print(f'ダウンロード先: https://zenodo.org/record/2535967')
        print(f'\nまたは、以下のコマンドで展開してください:')
        print(f'  cd {args.data_root} && tar -xf CIFAR-10-C.tar')
        return
    
    # 各コルプションタイプと強度で評価
    for corruption_type in corruption_types:
        results[corruption_type] = {}
        for severity in severities:
            try:
                print(f'\n評価中: {corruption_type}, severity={severity}')
                dataset, class_labels, dataset_mean, dataset_std = get_cifar10c_dataset(
                    args.data_root, corruption_type, severity
                )
                dataloader = DataLoader(
                    dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
                )
                
                metrics = evaluate_model(model, dataloader, device, model_args.model)
                results[corruption_type][severity] = metrics
                
                if model_args.model in ['ctm', 'lstm']:
                    print(f'  損失: {metrics["loss"]:.4f}')
                    print(f'  精度 (most certain): {metrics["accuracy_most_certain"]:.4f}')
                    print(f'  精度 (final tick): {metrics["accuracies_per_tick"][-1]:.4f}')
                else:
                    print(f'  損失: {metrics["loss"]:.4f}')
                    print(f'  精度: {metrics["accuracy"]:.4f}')
                    
            except FileNotFoundError as e:
                print(f'  エラー: {e}')
                results[corruption_type][severity] = None
                # データセットが見つからない場合は、そのコルプションタイプをスキップ
                break
            except Exception as e:
                print(f'  エラー: {e}')
                import traceback
                traceback.print_exc()
                results[corruption_type][severity] = None
    
    # 結果をまとめて表示
    print('\n' + '='*80)
    print('評価結果サマリー')
    print('='*80)
    
    if model_args.model in ['ctm', 'lstm']:
        print(f'{"コルプションタイプ":<25} {"Severity":<10} {"Loss":<10} {"Accuracy (most certain)":<25} {"Accuracy (final tick)":<25}')
        print('-'*80)
        for corruption_type in corruption_types:
            for severity in severities:
                if results[corruption_type].get(severity) is not None:
                    metrics = results[corruption_type][severity]
                    print(f'{corruption_type:<25} {severity:<10} {metrics["loss"]:<10.4f} {metrics["accuracy_most_certain"]:<25.4f} {metrics["accuracies_per_tick"][-1]:<25.4f}')
    else:
        print(f'{"コルプションタイプ":<25} {"Severity":<10} {"Loss":<10} {"Accuracy":<15}')
        print('-'*60)
        for corruption_type in corruption_types:
            for severity in severities:
                if results[corruption_type].get(severity) is not None:
                    metrics = results[corruption_type][severity]
                    print(f'{corruption_type:<25} {severity:<10} {metrics["loss"]:<10.4f} {metrics["accuracy"]:<15.4f}')
    
    # 平均精度を計算
    if model_args.model in ['ctm', 'lstm']:
        all_accuracies_most_certain = []
        all_accuracies_final = []
        for corruption_type in corruption_types:
            for severity in severities:
                if results[corruption_type].get(severity) is not None:
                    all_accuracies_most_certain.append(results[corruption_type][severity]['accuracy_most_certain'])
                    all_accuracies_final.append(results[corruption_type][severity]['accuracies_per_tick'][-1])
        if all_accuracies_most_certain:
            print(f'\n平均精度 (most certain): {np.mean(all_accuracies_most_certain):.4f}')
            print(f'平均精度 (final tick): {np.mean(all_accuracies_final):.4f}')
    else:
        all_accuracies = []
        for corruption_type in corruption_types:
            for severity in severities:
                if results[corruption_type].get(severity) is not None:
                    all_accuracies.append(results[corruption_type][severity]['accuracy'])
        if all_accuracies:
            print(f'\n平均精度: {np.mean(all_accuracies):.4f}')


if __name__ == '__main__':
    main()

