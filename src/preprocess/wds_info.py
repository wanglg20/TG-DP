#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, math, time, statistics
from collections import defaultdict, deque, Counter
from typing import Iterable, Optional, Dict, Any, Tuple, List, Set

import numpy as np
import torch
from torch.utils.data import DataLoader
from wds_loader import *
from dataloader_sync import AudiosetDataset  # 替换为保存你贴的AudiosetDataset类的文件路径

def _labels_from_tensor(label_indices: torch.Tensor) -> List[str]:
    """
    将 label_indices（可能是 one-hot/multi-hot）转成可hash的类别标识列表（字符串）。
    - 对 one-hot：取 argmax
    - 对 multi-hot：取非零索引组成的'|'连接字符串
    - 对稀疏或float：按 >0.5 视为正类；如需要可改阈值
    """
    label_indices = label_indices.detach().cpu()
    if label_indices.ndim == 1:
        # 单样本
        nonzero = (label_indices > 0.5).nonzero(as_tuple=True)[0].tolist()
        if len(nonzero) == 0:
            return ["__UNK__"]
        elif len(nonzero) == 1:
            return [str(nonzero[0])]
        else:
            return ["|".join(map(str, sorted(nonzero)))]
    else:
        out = []
        nz = (label_indices > 0.5).nonzero(as_tuple=False)  # [k,2]: (i, j)
        # 逐样本收集
        by_row = defaultdict(list)
        for i, j in nz.tolist():
            by_row[int(i)].append(int(j))
        B = label_indices.shape[0]
        for i in range(B):
            idxs = by_row.get(i, [])
            if len(idxs) == 0:
                out.append("__UNK__")
            elif len(idxs) == 1:
                out.append(str(idxs[0]))
            else:
                out.append("|".join(map(str, sorted(idxs))))
        return out

def _entropy_from_counts(counts: Iterable[int], eps: float = 1e-12) -> float:
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        p = c / total
        if p > 0:
            ent -= p * math.log(p + eps)
    return ent  # 自然对数

def _jaccard(a: Set[Any], b: Set[Any]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

@torch.no_grad()
def probe_loader(
    loader: DataLoader,
    dataset_size: Optional[int] = None,
    epochs: int = 1,
    max_steps_per_epoch: Optional[int] = None,
    call_set_epoch: bool = True,
    print_every: int = 200,
    window_for_diversity: int = 4096,
) -> Dict[str, Any]:
    """
    评测 DataLoader 的打乱质量。
    参数：
      - loader: 产出 (fbank, image_tensor, label_indices, vid, frame_idx)
      - dataset_size: 若提供，将计算覆盖率（唯一vid/总样本数）
      - epochs: 评测的 epoch 数
      - max_steps_per_epoch: 每个 epoch 最多迭代多少个 batch（None 表示跑完整个 loader）
      - call_set_epoch: 若 dataset 提供 set_epoch(e)，是否在每个 epoch 调用（用于对比“不调用”的影响）
      - print_every: 间隔多少批打印一次明细指标
      - window_for_diversity: 滑动窗口大小（单位：样本）用于估算“有效打散窗口多样性”

    返回：
      - 一个字典，包含每个 epoch 的汇总与跨 epoch 对比指标
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = getattr(loader, "dataset", None)
    has_set_epoch = hasattr(ds, "set_epoch")

    results = {
        "epochs": [],
        "cross_epoch": {}
    }

    # 用于跨 epoch 的对比
    epoch_vid_sequences: List[List[str]] = []  # 每个 epoch 见过的 vid 顺序
    epoch_unique_vids: List[Set[str]] = []

    for e in range(epochs):
        if call_set_epoch and has_set_epoch:
            try:
                ds.set_epoch(e)
            except Exception:
                pass

        uniq_vid_per_batch = []
        uniq_lbl_per_batch = []
        entropy_per_batch = []
        jaccard_between_batches = []

        # 滑动窗口：最近 W 个样本的 vid
        window = deque(maxlen=window_for_diversity)
        window_unique_ratios = []

        # 重复间隔：记录上次出现的 step
        last_seen: Dict[str, int] = {}
        repeat_distances: List[int] = []

        # 汇总
        seen_vids: Set[str] = set()
        seen_labels: Counter = Counter()

        prev_batch_vids: Set[str] = set()

        all_vids_this_epoch: List[str] = []

        steps = 0
        start_time = time.time()

        for batch in loader:
            # 允许 DataLoader 返回 dict 或 tuple；按约定解析
            if isinstance(batch, (list, tuple)) and len(batch) >= 5:
                fbank, image_tensor, label_indices, vid, frame_idx = batch[:5]
            elif isinstance(batch, dict):
                fbank = batch["fbank"]; image_tensor = batch["image_tensor"]
                label_indices = batch["label_indices"]; vid = batch["vid"]
                frame_idx = batch["frame_idx"]
            else:
                raise RuntimeError("Unknown batch structure. Expect (fbank, image_tensor, label_indices, vid, frame_idx).")

            # 取 batch 的 vid 列表（转 str）
            if isinstance(vid, (list, tuple)):
                vids = [str(v) for v in vid]
            elif isinstance(vid, torch.Tensor):
                vids = [str(v) for v in vid]
            else:
                # 单个
                vids = [str(vid)]

            # 取 batch 的 label（转成可hash的字符串）
            labels = _labels_from_tensor(label_indices)

            # --- 批内统计 ---
            vid_set = set(vids)
            lbl_set = set(labels)
            uniq_vid_per_batch.append(len(vid_set))
            uniq_lbl_per_batch.append(len(lbl_set))
            # 类别熵（批内 label 分布）
            _, counts = np.unique(np.array(labels), return_counts=True)
            entropy_per_batch.append(_entropy_from_counts(counts))

            # --- 批间相似度（Jaccard） ---
            if steps > 0:
                jaccard_between_batches.append(_jaccard(prev_batch_vids, vid_set))
            prev_batch_vids = vid_set

            # --- 滑动窗口多样性 ---
            for v in vids:
                window.append(v)
            window_unique_ratios.append(len(set(window)) / len(window))

            # --- 重复间隔 ---
            for v in vids:
                if v in last_seen:
                    repeat_distances.append(steps - last_seen[v])
                last_seen[v] = steps

            # --- 覆盖 ---
            seen_vids.update(vids)
            seen_labels.update(labels)
            all_vids_this_epoch.extend(vids)

            steps += 1
            if print_every and (steps % print_every == 0):
                print(f"[epoch {e} | batch {steps}] "
                      f"uniq_vid={uniq_vid_per_batch[-1]:3d}  "
                      f"uniq_lbl={uniq_lbl_per_batch[-1]:3d}  "
                      f"entropy={entropy_per_batch[-1]:.4f}  "
                      f"jaccard_prev={jaccard_between_batches[-1] if jaccard_between_batches else float('nan'):.3f}  "
                      f"win_unique={window_unique_ratios[-1]:.3f}")

            if (max_steps_per_epoch is not None) and (steps >= max_steps_per_epoch):
                break

        elapsed = time.time() - start_time

        # 汇总统计
        def _summary(arr: List[float]) -> Dict[str, float]:
            if not arr:
                return {"count": 0}
            return {
                "count": len(arr),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "p90": float(np.percentile(arr, 90)),
                "max": float(np.max(arr)),
            }

        uniq_vid_summary  = _summary(uniq_vid_per_batch)
        uniq_lbl_summary  = _summary(uniq_lbl_per_batch)
        entropy_summary   = _summary(entropy_per_batch)
        jaccard_summary   = _summary(jaccard_between_batches)
        window_diversity  = _summary(window_unique_ratios)
        repeat_summary    = _summary(repeat_distances)

        cover_ratio = None
        if dataset_size is not None and dataset_size > 0:
            cover_ratio = len(seen_vids) / float(dataset_size)

        epoch_result = {
            "epoch": e,
            "elapsed_sec": elapsed,
            "batches": steps,
            "unique_videos": len(seen_vids),
            "unique_labels": len(seen_labels),
            "coverage_ratio": cover_ratio,
            "uniq_vid_per_batch": uniq_vid_summary,
            "uniq_lbl_per_batch": uniq_lbl_summary,
            "entropy_per_batch": entropy_summary,
            "jaccard_between_batches": jaccard_summary,
            "window_unique_ratio": window_diversity,
            "repeat_distance_steps": repeat_summary,
        }
        results["epochs"].append(epoch_result)

        epoch_vid_sequences.append(all_vids_this_epoch)
        epoch_unique_vids.append(set(all_vids_this_epoch))

        # 控制台总结
        print(f"\n[Epoch {e}] batches={steps}, unique_videos={len(seen_vids)}, "
              f"coverage={cover_ratio if cover_ratio is not None else 'NA'}, "
              f"uniq_vid/batch(mean)={uniq_vid_summary.get('mean', float('nan')):.1f}, "
              f"entropy(mean)={entropy_summary.get('mean', float('nan')):.3f}, "
              f"jaccard_prev(mean)={jaccard_summary.get('mean', float('nan')):.3f}, "
              f"win_unique(mean)={window_diversity.get('mean', float('nan')):.3f}, "
              f"repeat_dist(mean_steps)={repeat_summary.get('mean', float('nan')):.1f}\n")

    # 跨 epoch 相似度（如果 >=2 轮）
    if epochs >= 2:
        # 1) 每两轮的“唯一vid集合”Jaccard
        pair_jaccard = []
        for i in range(epochs - 1):
            a, b = epoch_unique_vids[i], epoch_unique_vids[i + 1]
            pair_jaccard.append(_jaccard(a, b))
        results["cross_epoch"]["unique_vid_set_jaccard_mean"] = float(np.mean(pair_jaccard))

        # 2) 最前 N 样本序列的 Jaccard（看“前面几千样本是否重复”）
        N = 10000
        seq_jaccard = []
        for i in range(epochs - 1):
            a = set(epoch_vid_sequences[i][:N])
            b = set(epoch_vid_sequences[i + 1][:N])
            seq_jaccard.append(_jaccard(a, b))
        results["cross_epoch"]["first_N_seq_set_jaccard_mean"] = float(np.mean(seq_jaccard))
        results["cross_epoch"]["first_N"] = N

    return results




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps-per-epoch", type=int, default=None)
    ap.add_argument("--no-set-epoch", action="store_true", help="不调用 dataset.set_epoch(e)")
    ap.add_argument("--window", type=int, default=4096)
    args = ap.parse_args()

    # TODO: 替换为你的 DataLoader
    # loader, dataset_size = create_your_loader()
    audio_conf = {'num_mel_bins': 128, 'target_length': 416, 'freqm': 0, 'timem': 0, 'mixup': 0.0, 'dataset': 'as', 'mode':'train', 'mean':-5.081, 'std':4.4849,
                  'noise':False, 'label_smooth': 0, 'im_res': 224, 'shuffle': True}
    label_csv = '/data/wanglinge/project/weighted-cav-mae/src/data_info/vgg/class_labels_indices_vgg.csv'
    shards = "/data/wanglinge/dataset/VGGSound/wds/train/shard-{000000..000036}.tar"
    webdataset = AudioSetWebDataset(shards, audio_conf, label_csv=label_csv)
    loader = DataLoader(
        webdataset,
        batch_size=512,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    #dataset_size = len(loader)
    res = probe_loader(
        loader,
        # dataset_size=dataset_size,
        epochs=1,                  # 跑两轮看跨epoch相似度
        max_steps_per_epoch=None, 
        call_set_epoch=True,       # True=每轮调用 dataset.set_epoch(e)（若存在）
        print_every=200,
        window_for_diversity=4096
    )
    print(res)
    # ori_set = AudiosetDataset('/data/wanglinge/project/weighted-cav-mae/src/data_info/vgg/vggsound_train_partition.json', audio_conf, label_csv='/data/wanglinge/project/weighted-cav-mae/src/data_info/vgg/class_labels_indices_vgg.csv')
    # ori_loader = DataLoader(
    #     ori_set,
    #     batch_size=512,
    #     num_workers=8,
    #     pin_memory=True,
    #     persistent_workers=True,
    #     shuffle=True
    # )
    # ori_res = probe_loader(
    #     ori_loader,
    #     dataset_size=len(ori_set),
    #     epochs=1,                  # 跑两轮看跨epoch相似度
    #     max_steps_per_epoch=None,  # None = 跑完整个 loader
    #     call_set_epoch=False,       # True=每轮调用 dataset.set_epoch(e)（若存在）
    #     print_every=50,
    #     window_for_diversity=4096
    # )
    # print(ori_res)
    # return

if __name__ == "__main__":
    main()

