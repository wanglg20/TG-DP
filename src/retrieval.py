from shlex import join
import warnings

from sympy import diag
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import os
import json
import dataloader as dataloader
import dataloader_sync
from dataloader_sync import train_collate_fn
import torch
import numpy as np
from torch.cuda.amp import autocast
from torch import nn
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
from tabulate import tabulate
import argparse
import torch.nn.functional as F
from models.cavmae_dp import CAVMAE_DP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

@torch.inference_mode()
def diag_mean_cosine(A, B, device='cuda', dtype=torch.float32, eps=1e-12):
    """
    A, B: torch.Tensor [N, T, D]
    return: torch.Tensor [N, N] on CPU
    """
    print("using diag_mean_cosine")
    assert A.ndim == 3 and B.ndim == 3 and A.shape[1:] == B.shape[1:]
    N, T, D = A.shape

    A = F.normalize(A.to(device, dtype=dtype), p=2, dim=-1, eps=eps)
    B = F.normalize(B.to(device, dtype=dtype), p=2, dim=-1, eps=eps)

    A_flat = A.reshape(N, T * D)
    B_flat = B.reshape(N, T * D)

    torch.set_float32_matmul_precision('highest')

    S = (A_flat @ B_flat.T) / T
    return S.float().cpu()
import torch.nn.functional as F



def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]


def get_similarity(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim

# get mean
def get_sim_mat(a, b):
    B = a.shape[0]
    sim_mat = np.empty([B, B])
    for i in range(B):
        for j in range(B):
            sim_mat[i, j] = get_similarity(a[i, :], b[j, :])
    return sim_mat

def get_agg_sim_mat(a, b, strategy='max'):
    # a and b are tensors of shape (batch_size, num_frames, feature_dim)
    # e.g., torch.Size([100, 10, 768])
    B = a.shape[0]
    num_frames = a.shape[1]  # All videos have the same number of frames
    sim_mat = np.empty([B, B])
    
    for i in range(B):
        for j in range(B):
            # Compute similarity for all frame pairs
            if strategy == 'max' or strategy == 'mean':
                frame_similarities = np.array([[get_similarity(a[i, k], b[j, l]) for l in range(num_frames)] for k in range(num_frames)])
            # Aggregate similarities based on the chosen strategy
            if strategy == 'max':
                sim_mat[i, j] = np.max(frame_similarities)
            elif strategy == 'mean':
                sim_mat[i, j] = np.mean(frame_similarities)
            elif strategy == 'diagonal_mean':
                # Compare elements from the diagonal of the video submatrix
                diagonal_similarities = [get_similarity(a[i, k], b[j, k]) for k in range(num_frames)]
                sim_mat[i, j] = np.mean(diagonal_similarities)
            elif strategy == 'diagonal_max':
                # Compare elements from the diagonal of the video submatrix
                diagonal_similarities = [get_similarity(a[i, k], b[j, k]) for k in range(num_frames)]
                sim_mat[i, j] = np.max(diagonal_similarities)
            else:
                raise ValueError(f"Unknown aggregation strategy: {strategy}")
    return sim_mat

def get_agg_sim_mat_audio_video(a, b, strategy='max', direction='audio'):
    print(f"A shape: {a.shape}")
    print(f"B shape: {b.shape}")
    num_frames = 16    
    
    if direction == 'audio':  # A is audio
        batch_size_a = a.shape[0]
        batch_size_b = b.shape[0] // num_frames
        sim_mat = np.empty([batch_size_a, batch_size_b])
        
        for i in range(batch_size_a):
            for j in range(batch_size_b):
                frame_similarities = np.array([get_similarity(a[i], b[j*num_frames + k]) for k in range(num_frames)])
                sim_mat[i, j] = np.max(frame_similarities) if strategy == 'max' else np.mean(frame_similarities)
    
    elif direction == 'video':  # B is audio
        batch_size_a = a.shape[0] // num_frames
        batch_size_b = b.shape[0] 
        sim_mat = np.empty([batch_size_a, batch_size_b])
        
        for i in range(batch_size_a):
            for j in range(batch_size_b):
                frame_similarities = np.array([get_similarity(a[i*num_frames + k], b[j]) for k in range(num_frames)])
                sim_mat[i, j] = np.max(frame_similarities) if strategy == 'max' else np.mean(frame_similarities)
    
    else:
        raise ValueError("Invalid input shapes. One input should be (batch_size, 768) and the other (batch_size, num_frames, 768)")
    
    return sim_mat


@torch.inference_mode()
def get_retrieval_sim_and_ids(audio_model, val_loader, direction='audio', model_type='pretrain', strategy='max', cls_token=False, local_matching=False):
    """
    与 get_retrieval_result 几乎相同，但返回：
      - sim_mat: (N, N) 的相似度矩阵（numpy）
      - all_ids: 长度为 N 的样本 video_id 列表（字符串或 int）
    """
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    A_a_feat, A_v_feat = [], []
    all_ids = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc="Processing batches"):
            # 注意：这里保持与原脚本一致
            if 'sync' in model_type or 'enhanced' in model_type:
                a_input, v_input, labels, video_id, frame_indices = batch
            else:
                (a_input, v_input, labels) = batch
                # 如果不是 sync/enhanced，就自己造一个简单 id
                video_id = list(range(len(a_input)))

            if i == 0:
                print("A_shape", a_input.shape)
                print("V_shape", v_input.shape)

            if 'sync' in model_type or 'enhanced' in model_type:
                # flatten batch so we process all frames at the same time
                a_input = a_input.reshape(a_input.shape[0] * a_input.shape[1], a_input.shape[2], a_input.shape[3])
                v_input = v_input.reshape(v_input.shape[0] * v_input.shape[1], v_input.shape[2], v_input.shape[3], v_input.shape[4])

            audio_input, video_input = a_input.to(device), v_input.to(device)
            with autocast():
                if cls_token:
                    tokens_audio_output, tokens_video_output, cls_audio_output, cls_video_output = audio_model.module.forward_feat(audio_input, video_input)
                    if local_matching:
                        audio_output = torch.mean(tokens_audio_output, dim=1)
                        video_output = torch.mean(tokens_video_output, dim=1)
                    else:
                        audio_output = cls_audio_output
                        video_output = cls_video_output
                else:
                    # mean pool all patches
                    audio_output, video_output = audio_model.module.forward_feat(audio_input, video_input)
                    audio_output = torch.mean(audio_output, dim=1)
                    if 'enhanced' in model_type:
                        audio_output = audio_output[::16]
                    video_output = torch.mean(video_output, dim=1)

                # normalization
                audio_output = torch.nn.functional.normalize(audio_output, dim=-1)
                video_output = torch.nn.functional.normalize(video_output, dim=-1)

            audio_output = audio_output.to('cpu').detach()
            video_output = video_output.to('cpu').detach()

            if 'sync' in model_type:
                # Group features from the same video together
                num_frames = audio_output.shape[0] // len(video_id)
                audio_output = audio_output.view(len(video_id), num_frames, -1)
                video_output = video_output.view(len(video_id), num_frames, -1)

            A_a_feat.append(audio_output)
            A_v_feat.append(video_output)
            # 保持与特征拼接顺序一致
            all_ids.extend(list(video_id))

    A_a_feat = torch.cat(A_a_feat)
    A_v_feat = torch.cat(A_v_feat)
    print(f"A_a_feat shape: {A_a_feat.shape}, A_v_feat shape: {A_v_feat.shape}")
    print(f"Num ids: {len(all_ids)}")

    if direction == 'audio':
        # audio->visual retrieval
        if 'sync' in model_type:
            sim_mat = diag_mean_cosine(A_a_feat, A_v_feat, device=device).numpy()
        elif 'enhanced' in model_type:
            sim_mat = get_agg_sim_mat_audio_video(A_a_feat.numpy(), A_v_feat.numpy(), strategy='mean', direction=direction)
        else:
            sim_mat = get_sim_mat(A_a_feat.numpy(), A_v_feat.numpy())
    elif direction == 'video':
        # visual->audio retrieval
        if 'sync' in model_type:
            sim_mat = diag_mean_cosine(A_v_feat, A_a_feat, device=device).numpy()
        elif 'enhanced' in model_type:
            sim_mat = get_agg_sim_mat_audio_video(A_v_feat.numpy(), A_a_feat.numpy(), strategy='mean', direction=direction)
        else:
            sim_mat = get_sim_mat(A_v_feat.numpy(), A_a_feat.numpy())
    else:
        raise ValueError(f"Unknown direction: {direction}")

    # 可选：打印一下整体指标做 sanity check
    metrics = compute_metrics(sim_mat)
    print_computed_metrics(metrics)

    return sim_mat, all_ids

def compute_metrics(x):
    # Sort the similarity matrix in descending order for each row
    sx = np.sort(-x, axis=1)
    
    # Get the diagonal elements (self-similarity scores)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    
    print(sx.shape, d.shape)
    # Calculate the difference between sorted similarities and self-similarities
    ind = sx - d
    
    # Find the indices where the difference is zero (correct matches)
    ind = np.where(ind == 0)
    ind = ind[1]  # We only need the column indices
    
    # Initialize a dictionary to store the computed metrics
    metrics = {}
    
    # Compute Recall@1: percentage of correct matches at the top position
    metrics['R1'] = float(np.sum(ind == 0)) / len(ind)
    
    # Compute Recall@5: percentage of correct matches in the top 5 positions
    metrics['R5'] = float(np.sum(ind < 5)) / len(ind)
    
    # Compute Recall@10: percentage of correct matches in the top 10 positions
    metrics['R10'] = float(np.sum(ind < 10)) / len(ind)
    
    # Compute Median Rank (MR): the median position of the correct matches
    metrics['MR'] = np.median(ind) + 1  # Add 1 because indices start at 0
    
    return metrics

def print_computed_metrics(metrics):
    r1 = metrics['R1']
    r5 = metrics['R5']
    r10 = metrics['R10']
    mr = metrics['MR']
    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}'.format(r1, r5, r10, mr))

# direction: 'audio' means audio->visual retrieval, 'video' means visual->audio retrieval
def get_retrieval_result(audio_model, val_loader, direction='audio', model_type='pretrain', strategy='max', cls_token=False, local_matching=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    A_a_feat, A_v_feat = [], []
    with torch.no_grad():
        # Add tqdm progress bar
        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc="Processing batches"):
            if 'sync' or 'enhanced' in model_type:
                a_input, v_input, labels, video_id, frame_indices = batch
            else:
                (a_input, v_input, labels) = batch
            if i == 0:
                print("A_shape", a_input.shape)
                print("V_shape", v_input.shape)
            if 'sync' or 'enhanced' in model_type:
                # flatten batch so we process all frames at the same time
                a_input = a_input.reshape(a_input.shape[0] * a_input.shape[1], a_input.shape[2], a_input.shape[3])
                v_input = v_input.reshape(v_input.shape[0] * v_input.shape[1], v_input.shape[2], v_input.shape[3], v_input.shape[4])

            audio_input, video_input = a_input.to(device), v_input.to(device)
            with autocast():
                if cls_token:
                    tokens_audio_output, tokens_video_output, cls_audio_output, cls_video_output = audio_model.module.forward_feat(audio_input, video_input)
                    if local_matching:
                        audio_output = torch.mean(tokens_audio_output, dim=1)
                        video_output = torch.mean(tokens_video_output, dim=1)
                    else:
                        audio_output = cls_audio_output
                        video_output = cls_video_output
                else:
                    # mean pool all patches
                    audio_output, video_output = audio_model.module.forward_feat(audio_input, video_input)
                    audio_output = torch.mean(audio_output, dim=1)
                    if 'enhanced' in model_type:
                        audio_output = audio_output[::16]
                    video_output = torch.mean(video_output, dim=1)
                # normalization
                audio_output = torch.nn.functional.normalize(audio_output, dim=-1)
                video_output = torch.nn.functional.normalize(video_output, dim=-1)
            audio_output = audio_output.to('cpu').detach()
            video_output = video_output.to('cpu').detach()
            
            if 'sync' in model_type:
                # Group features from the same video together
                num_frames = audio_output.shape[0] // len(video_id)
                audio_output = audio_output.view(len(video_id), num_frames, -1)
                video_output = video_output.view(len(video_id), num_frames, -1)
            A_a_feat.append(audio_output)
            A_v_feat.append(video_output)

    A_a_feat = torch.cat(A_a_feat)
    A_v_feat = torch.cat(A_v_feat)
    print(f"A_a_feat shape: {A_a_feat.shape}, A_v_feat shape: {A_v_feat.shape}")
    if direction == 'audio':
        # audio->visual retrieval
        if 'sync' in model_type:
            sim_mat = diag_mean_cosine(A_a_feat, A_v_feat, device=device)
            #sim_mat = get_agg_sim_mat(A_a_feat, A_v_feat, strategy=strategy)
        elif 'enhanced' in model_type:
            sim_mat = get_agg_sim_mat_audio_video(A_a_feat, A_v_feat, strategy='mean', direction=direction)
        else:
            sim_mat = get_sim_mat(A_a_feat, A_v_feat)
    elif direction == 'video':
        # visual->audio retrieval
        if 'sync' in model_type:
            #sim_mat = get_agg_sim_mat(A_v_feat, A_a_feat, strategy=strategy)
            sim_mat = diag_mean_cosine(A_v_feat, A_a_feat, device=device)
        elif 'enhanced' in model_type:
            sim_mat = get_agg_sim_mat_audio_video(A_v_feat, A_a_feat, strategy='mean', direction=direction)
        else:
            sim_mat = get_sim_mat(A_v_feat, A_a_feat)
    result = compute_metrics(sim_mat)
    print_computed_metrics(result)
    return result['R1'], result['R5'], result['R10'], result['MR']

def eval_retrieval(model, data, audio_conf, label_csv, direction, num_class, model_type='pretrain', batch_size=196, strategy='max', num_register_tokens=8, cls_token=False, local_matching=False):
    print("num register tokens: ", num_register_tokens)
    print(model)
    print(data)
    frame_use = 5
    # eval setting
    val_audio_conf = audio_conf
    val_audio_conf['frame_use'] = frame_use
    if 'sync' or 'enhanced' in model_type:
        val_loader = torch.utils.data.DataLoader(dataloader_sync.AudiosetDataset(data, label_csv=label_csv, audio_conf=val_audio_conf), batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True, collate_fn=train_collate_fn)
    else:
        val_loader = torch.utils.data.DataLoader(dataloader.AudiosetDataset(data, label_csv=label_csv, audio_conf=val_audio_conf), batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True)
    # cav-mae only been ssl pretrained

    if 'ch' in model_type:
        model_type = model_type.replace('_ch', '')
        contrastive_heads = True
    else:
        contrastive_heads = False


    if model_type == 'sync_pretrain_registers':
        audio_model = CAVMAE_DP(audio_length=val_audio_conf['target_length'], modality_specific_depth=11, num_register_tokens=num_register_tokens, total_frame=audio_conf['total_frame'], contrastive_heads=contrastive_heads)
    elif model_type == 'sync_pretrain_registers_cls':
        audio_model = CAVMAE_DP(audio_length=val_audio_conf['target_length'], modality_specific_depth=11, num_register_tokens=num_register_tokens, cls_token=True, total_frame=audio_conf['total_frame'], contrastive_heads=contrastive_heads)
    elif model_type == 'sync_pretrain_registers_cls_global_local':
        audio_model = CAVMAE_DP(audio_length=val_audio_conf['target_length'], modality_specific_depth=11, num_register_tokens=num_register_tokens, cls_token=True, global_local_losses=True, total_frame=audio_conf['total_frame'], contrastive_heads=contrastive_heads)
    elif model_type == 'sync_pretrain':
        audio_model = CAVMAE_DP(audio_length=val_audio_conf['target_length'], modality_specific_depth=11, num_register_tokens=0, total_frame=audio_conf['total_frame'], contrastive_heads=contrastive_heads)
    elif model_type == 'pretrain' or model_type == 'pretrain_enhanced':
        audio_model = CAVMAE_DP(modality_specific_depth=11)
    # cav-mae only been ssl pretrained + supervisedly finetuned
    elif model_type == 'finetune':
        audio_model = CAVMAE_DP(label_dim=num_class, modality_specific_depth=11)


    print(f"Loading model from {model}")
    if not os.path.isfile(model):
        raise FileNotFoundError(f"Model file not found: {model}")
    size = os.path.getsize(model)
    print(f"Model size: {size / 1024 / 1024} MB")
    assert isinstance(model, str), f"Expected model path (str) but got {type(model)}"
    sdA = torch.load(model, map_location=device)
    if isinstance(audio_model, torch.nn.DataParallel) == False:
        audio_model = torch.nn.DataParallel(audio_model)
    msg = audio_model.load_state_dict(sdA, strict=False)
    print(msg)
    audio_model.eval()
    r1, r5, r10, mr = get_retrieval_result(audio_model, val_loader, direction, model_type, strategy, cls_token, local_matching)
    return r1 * 100, r5 * 100, r10 * 100, mr


def build_model_names_from_args(args, parser):
    """
    Build model mapping from CLI args.
    Supports two formats:
    1) --model_dict_json '{"name": ["/path/to/model.pth", "model_type"]}'
    2) --model_names ... --model_paths ... --model_types ... (same length)
    """
    if args.model_dict_json:
        try:
            parsed = json.loads(args.model_dict_json)
        except json.JSONDecodeError as exc:
            parser.error(f"Invalid --model_dict_json JSON: {exc}")

        if not isinstance(parsed, dict):
            parser.error("--model_dict_json must be a JSON object like {'name': ['/path', 'type']}")

        model_names = {}
        for name, value in parsed.items():
            if not isinstance(name, str):
                parser.error("All model names in --model_dict_json must be strings")
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                parser.error("Each model config in --model_dict_json must be [model_path, model_type]")

            model_path, model_type = value
            if not isinstance(model_path, str) or not isinstance(model_type, str):
                parser.error("model_path and model_type in --model_dict_json must be strings")
            model_names[name] = (model_path, model_type)

        if len(model_names) == 0:
            parser.error("--model_dict_json cannot be empty")
        return model_names

    if args.model_names is None or args.model_paths is None or args.model_types is None:
        parser.error(
            "Please provide either --model_dict_json or all of --model_names --model_paths --model_types"
        )

    if not (len(args.model_names) == len(args.model_paths) == len(args.model_types)):
        parser.error(
            "--model_names, --model_paths and --model_types must have the same length"
        )

    model_names = {}
    for model_name, model_path, model_type in zip(args.model_names, args.model_paths, args.model_types):
        model_names[model_name] = (model_path, model_type)
    return model_names

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Toy example for argument parsing')
    
    parser.add_argument('--dataset', type=str, choices=['audioset', 'vggsound'], 
                        help='Dataset to use for retrieval')
    parser.add_argument('--strategy', type=str, 
                        help='Strategy for aggregation')
    parser.add_argument('--directions', type=str, nargs='+', 
                        help='Directions for evaluation')
    parser.add_argument('--nums_samples', type=int, nargs='+', 
                        help='Number of samples to test')
    parser.add_argument('--model_names', type=str, nargs='+',
                        help='Model names to test (used with --model_paths and --model_types)')
    parser.add_argument('--model_paths', type=str, nargs='+',
                        help='Model checkpoint paths (one per model name)')
    parser.add_argument('--model_types', type=str, nargs='+',
                        help='Model types (one per model name)')
    parser.add_argument('--model_dict_json', type=str,
                        help='JSON dict for model configs, e.g. {"name": ["/path/model.pth", "sync_pretrain_registers_cls_4s"]}')
    parser.add_argument("--as_data_path", type=str, default='/data/wanglinge/project/weighted-cav-mae/src/data_info/as2M/audioset_eval_5_per_class_for_retrieval_cleaned_valid.json', help="Path to Audioset data info")
    parser.add_argument("--as_label_csv", type=str, default='/data/wanglinge/project/weighted-cav-mae/src/data_info/as2M/class_labels_indices.csv', help="Path to Audioset label csv")
    parser.add_argument("--vgg_data_path", type=str, default='/data/home/zdhs0059/wanglinge/project/weighted-cav-mae/src/data_info/VGG/vgg_partition_test_5_per_class.json', help="Path to VGGSound data info")
    parser.add_argument("--vgg_label_csv", type=str, default='/data/home/zdhs0059/wanglinge/project/weighted-cav-mae/src/data_info/VGG/class_labels_indices_vgg.csv', help="Path to VGGSound label csv")    
    args = parser.parse_args()

    # Print out the parsed arguments
    print(f"Dataset: {args.dataset}")
    print(f"Strategy: {args.strategy}")
    print(f"Directions: {args.directions}")
    print(f"Number of Samples: {args.nums_samples}")

    dataset = args.dataset
    strategy = args.strategy
    directions = args.directions
    nums_samples = args.nums_samples

    model_names = build_model_names_from_args(args, parser)
    
    print("Running retrieval for the following models: ", model_names.keys())

    res = []
    for dataset in ['vggsound']:
        if dataset == "audioset":
            data = args.as_data_path
            label_csv = args.as_label_csv
            num_class = 527
        elif dataset == "vggsound":
            data = args.vgg_data_path
            label_csv = args.vgg_label_csv
            num_class = 309
        else:
            print(f"Unsupported dataset: {dataset}")
            exit(1)

        for num_samples in tqdm(nums_samples, desc="Testing sample sizes"):
            for model_name, (model_path, model_type) in tqdm(model_names.items(), desc=f"Processing models for {dataset}", leave=False):
                if 'sync' in model_type:
                    if '2s' in model_type:
                        target_length = 192
                    elif '3s' in model_type:
                        target_length = 304
                    elif '4s' in model_type:
                        target_length = 416
                    elif '5s' in model_type:
                        target_length = 512
                    elif '6s' in model_type:
                        target_length = 624
                    elif '7s' in model_type:
                        target_length = 720
                    elif '10s' in model_type:
                        target_length = 1024
                    else:
                        target_length = 96
                else:
                    target_length = 1024
                print("Using target_length: ", target_length)
                model_type = model_type.replace('_2s', '').replace('_3s', '').replace('_4s', '').replace('_5s', '').replace('_6s', '').replace('_7s', '').replace('_10s', '')
                if 'cls' in model_type:
                    cls_token = True
                else:
                    cls_token = False
                print("Using model_type: ", model_type)
                for direction in tqdm(directions, desc="Evaluating directions", leave=False):
                    audio_conf = {'num_mel_bins': 128, 'target_length': target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': dataset,
                                'mode': 'retrieval', 'mean': -5.081, 'std': 4.4849, 'noise': False, 'im_res': 224, 'frame_use': 5, 'num_samples': num_samples, 'total_frame': 16}
                    

                    if 'local' in model_name:
                        r1, r5, r10, mr = eval_retrieval(model_path, data, audio_conf=audio_conf, label_csv=label_csv, num_class=num_class, direction=direction, model_type=model_type, batch_size=50, strategy=strategy, num_register_tokens=8, cls_token=cls_token, local_matching=True, )
                    else:
                        r1, r5, r10, mr = eval_retrieval(model_path, data, audio_conf=audio_conf, label_csv=label_csv, num_class=num_class, direction=direction, model_type=model_type, batch_size=50, strategy=strategy, num_register_tokens=8, cls_token=cls_token, local_matching=False, )
                    print('Results for model: {}, dataset: {}, direction: {}, num_samples: {} - R@1: {:.4f}, R@5: {:.4f}, R@10: {:.4f}, MR: {}'.format(model_name, dataset, direction, num_samples, r1, r5, r10, mr))
                    res.append([model_name, dataset, direction, num_samples, r1, r5, r10, mr])
                    res_sorted = sorted(res, key=lambda x: x[-1])  # Sort by MR
                    print("\nCurrent Results Table:")
                    print(tabulate(res_sorted, headers=["Model", "Dataset", "Direction", "Num Samples", "R@1", "R@5", "R@10", "MR"]))




