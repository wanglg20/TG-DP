#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, json, tarfile, random, argparse, hashlib
from multiprocessing import Pool
from functools import partial
from typing import Iterable, Iterator, List, Tuple, Dict, Union
from tqdm import tqdm

# ---------------------------
# Utilities
# ---------------------------

def file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return -1

def gather_frames(video_id: str, video_path: str, num_frames: int = 16) -> Union[List[Tuple[str, str, int, bool]], None]:
    """收集帧（以路径形式返回）。
    返回列表元素格式: (arcname_in_tar, src, size, is_bytes)
      - 对于大文件：src = 文件路径，is_bytes=False
      - 对于小对象（如meta）：src = bytes/BytesIO，is_bytes=True
    """
    items = []
    for i in range(num_frames):
        d = os.path.join(video_path, f"frame_{i}")
        fpath = os.path.join(d, f"{video_id}.jpg")
        if not os.path.exists(fpath):
            return None
        sz = file_size(fpath)
        if sz <= 0:
            return None
        items.append((f"frame_{i:02d}.jpg", fpath, sz, False))
    return items

def pack_one_sample(sample: Dict, num_frames: int = 16) -> Tuple[bool, str, Union[str, List[Tuple[str, Union[str, bytes, io.BytesIO], int, bool]]]]:
    """
    返回:
      - (True, key, files) 其中 files 是 [(arcname, src, size, is_bytes), ...]
      - (False, key, err_msg)
    """
    try:
        video_id = sample["video_id"]
        video_path = sample["video_path"]
        wav_path   = sample["wav"]
        raw_video_path = sample.get("raw_video_path", "")
        labels = sample.get("labels", "")

        # 1) 帧
        frames = gather_frames(video_id, video_path, num_frames=num_frames)
        if frames is None:
            return (False, video_id, f"missing frame(s) for {video_id}")

        # 2) 音频
        if not os.path.exists(wav_path):
            return (False, video_id, f"missing wav for {video_id}")
        wav_sz = file_size(wav_path)
        if wav_sz <= 0:
            return (False, video_id, f"bad wav for {video_id}")

        # 3) 元信息
        if isinstance(labels, str):
            label_list = [x for x in labels.split(",") if x]
        elif isinstance(labels, list):
            label_list = labels
        else:
            label_list = []

        meta = {
            "video_id": video_id,
            "labels": label_list,
            "raw_video_path": raw_video_path,
            "video_path": video_path,
            "wav_path": wav_path,
            "num_frames": num_frames,
        }
        meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")

        files: List[Tuple[str, Union[str, bytes, io.BytesIO], int, bool]] = []
        files.extend(frames)  # 帧（路径，流式读）
        files.append(("audio.wav", wav_path, wav_sz, False))  # 音频（路径，流式读）
        files.append(("meta.json", meta_bytes, len(meta_bytes), True))  # 小对象（内存）

        return (True, video_id, files)
    except Exception as e:
        return (False, sample.get("video_id", "unknown"), f"exception: {e}")

def worker_pack(args):
    sample, num_frames = args
    return pack_one_sample(sample, num_frames=num_frames)

# ---------------------------
# Streaming tar writer
# ---------------------------

class ShardWriter:
    """管理分片打开/关闭，支持按样本数和累计字节切分。"""
    def __init__(self, out_dir: str, shard_prefix: str, zstd_level: int = 0,
                 max_count: int = 5000, max_bytes: int = 0):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.shard_prefix = shard_prefix
        self.zstd_level = zstd_level
        self.max_count = max_count
        self.max_bytes = max_bytes  # 0 表示不按字节限制
        self.shard_idx = 0
        self.count_in_shard = 0
        self.bytes_in_shard = 0
        self.tf: tarfile.TarFile = self._open_new_shard(self.shard_idx)

    def _open_new_shard(self, idx: int, offset=297) -> tarfile.TarFile:
        idx = idx + offset  # 避免和已有文件名冲突
        fname = os.path.join(self.out_dir, f"{self.shard_prefix}-{idx:06d}.tar")
        mode = "w"
        if self.zstd_level > 0:
            # Python 3.12+ 支持 ":zst"，兜底到不压缩
            try:
                return tarfile.open(fname, mode=mode + ":zst", preset=self.zstd_level)
            except Exception:
                print("[warn] tarfile zstd not supported, fallback to uncompressed tar:", fname)
        return tarfile.open(fname, mode=mode)

    def _rollover_if_needed(self):
        need_roll = False
        if self.max_count > 0 and self.count_in_shard >= self.max_count:
            need_roll = True
        if self.max_bytes > 0 and self.bytes_in_shard >= self.max_bytes:
            need_roll = True
        if need_roll:
            self.tf.close()
            self.shard_idx += 1
            self.count_in_shard = 0
            self.bytes_in_shard = 0
            self.tf = self._open_new_shard(self.shard_idx)

    def add_sample(self, key: str, files: List[Tuple[str, Union[str, bytes, io.BytesIO], int, bool]]):
        """把一个样本写入当前分片（key/arcname）。"""
        # 先预估这条样本的字节（不含tar header开销，够用了）
        sample_bytes = sum(sz for _, _, sz, _ in files if sz > 0)

        # 如果开启按字节切分，且单个样本就超过上限，直接在空分片里写它（避免“永远无法写入”）
        if self.max_bytes > 0 and sample_bytes > self.max_bytes and self.count_in_shard == 0:
            # 允许该分片只包含一个超大样本
            pass
        else:
            # 正常滚动
            self._rollover_if_needed()

        for arcname, src, sz, is_bytes in files:
            ti = tarfile.TarInfo(name=f"{key}/{arcname}")
            ti.size = sz if sz >= 0 else 0

            if is_bytes:
                if isinstance(src, bytes):
                    fileobj = io.BytesIO(src)
                elif isinstance(src, io.BytesIO):
                    fileobj = src
                else:
                    # 不应发生，兜底成空
                    fileobj = io.BytesIO(b"")
                self.tf.addfile(ti, fileobj)
            else:
                # 路径 -> 流式读
                with open(src, "rb") as f:
                    self.tf.addfile(ti, f)

        self.count_in_shard += 1
        self.bytes_in_shard += sample_bytes

    def close(self):
        if self.tf is not None:
            self.tf.close()
            self.tf = None

def write_shards_stream(samples_iter: Iterator[Tuple[bool, str, Union[str, List[Tuple[str, Union[str, bytes, io.BytesIO], int, bool]]]]],
                        out_dir: str, shard_prefix: str = "shard",
                        zstd_level: int = 0, max_count: int = 5000, max_bytes: int = 0) -> Iterator[Tuple[bool, str, Union[None, str]]]:
    """
    流式写分片：
      - 逐个消费 pack 结果；成功则立刻写入当前分片；
      - 到阈值（样本数/字节数）自动滚动到下一个分片；
      - 每处理完一个样本就 yield (ok, key, info)
    """
    sw = ShardWriter(out_dir, shard_prefix, zstd_level, max_count, max_bytes)
    try:
        for ok, key, files_or_err in samples_iter:
            if not ok:
                yield (False, key, files_or_err)  # 错误透传
                continue
            files = files_or_err  # type: ignore
            sw.add_sample(key, files)
            yield (True, key, None)
    finally:
        sw.close()

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default='/data/wanglinge/project/weighted-cav-mae/src/data_info/as2M/pending/pending3_valid.json', help="标注 JSON 路径（含 data 列表）") 
    parser.add_argument("--out", default='/data/wanglinge/dataset/tmp', help="输出目录，用于存放 shard-*.tar")
    parser.add_argument("--num-frames", type=int, default=16, help="每个样本的帧数，默认 16")
    parser.add_argument("--shard-max-count", type=int, default=5000, help="每分片最多样本数；0 表示不按样本数限制")
    parser.add_argument("--shard-max-bytes", type=str, default="0",
                        help="每分片最大字节数，支持单位：K/M/G（如 '4G'）；0 表示不按字节限制")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--zstd", type=int, default=0, help="zstd 压缩等级，0 不压缩（推荐不压或 3~6）")
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 个样本（调试用）")
    parser.add_argument("--shard-prefix", type=str, default="shard", help="分片前缀名")
    args = parser.parse_args()

    # 解析 shard-max-bytes
    def parse_bytes(s: str) -> int:
        s = s.strip().upper()
        if s.endswith("G"):
            return int(float(s[:-1]) * (1024 ** 3))
        if s.endswith("M"):
            return int(float(s[:-1]) * (1024 ** 2))
        if s.endswith("K"):
            return int(float(s[:-1]) * 1024)
        return int(s)

    max_bytes = parse_bytes(args.shard_max_bytes)

    with open(args.json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    items = meta["data"]
    items = items[87395:]  # 上次处理到这里为止
    if args.limit > 0:
        items = items[:args.limit]

    # 确定性 shuffle
    rng = random.Random(args.seed)
    rng.shuffle(items)

    pack_args = [(s, args.num_frames) for s in items]
    total = len(pack_args)

    errors = []
    ok_cnt = 0

    if args.workers > 1:
        with Pool(processes=args.workers) as pool:
            results_iter = pool.imap_unordered(worker_pack, pack_args, chunksize=64)
            stream = write_shards_stream(results_iter, args.out,
                                         shard_prefix=args.shard_prefix,
                                         zstd_level=args.zstd,
                                         max_count=args.shard_max_count,
                                         max_bytes=max_bytes)
            for ok, key, info in tqdm(stream, total=total, desc="pack+write"):
                if not ok:
                    errors.append((key, info))
                else:
                    ok_cnt += 1
    else:
        def gen() -> Iterator[Tuple[bool, str, Union[str, List[Tuple[str, Union[str, bytes, io.BytesIO], int, bool]]]]]:
            for pa in tqdm(pack_args, desc="pack"):
                yield worker_pack(pa)

        stream = write_shards_stream(gen(), args.out,
                                     shard_prefix=args.shard_prefix,
                                     zstd_level=args.zstd,
                                     max_count=args.shard_max_count,
                                     max_bytes=max_bytes)
        for ok, key, info in tqdm(stream, total=total, desc="write"):
            if not ok:
                errors.append((key, info))
            else:
                ok_cnt += 1

    print(f"[done] OK samples: {ok_cnt} / {total}")
    if errors:
        print(f"[warn] {len(errors)} samples failed. Show first 20:")
        for k, e in errors[:20]:
            print("  -", k, "=>", e)

if __name__ == "__main__":
    main()
