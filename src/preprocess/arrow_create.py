#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, json, argparse, random
from typing import List, Dict, Any, Tuple, Optional
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.ipc as ipc

# ---------------------------
# Utils
# ---------------------------

def file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return -1

def parse_bytes(s: str) -> int:
    s = s.strip().upper()
    if s.endswith("G"): return int(float(s[:-1]) * (1024 ** 3))
    if s.endswith("M"): return int(float(s[:-1]) * (1024 ** 2))
    if s.endswith("K"): return int(float(s[:-1]) * 1024)
    return int(s)

def gather_frames_bytes(video_id: str, video_path: str, num_frames: int) -> Optional[List[bytes]]:
    frames = []
    for i in range(num_frames):
        d = os.path.join(video_path, f"frame_{i}")
        fpath = os.path.join(d, f"{video_id}.jpg")
        if not os.path.exists(fpath): return None
        sz = file_size(fpath)
        if sz <= 0: return None
        with open(fpath, "rb") as f:
            frames.append(f.read())
    return frames

def gather_frames_paths(video_id: str, video_path: str, num_frames: int) -> Optional[List[str]]:
    frames = []
    for i in range(num_frames):
        d = os.path.join(video_path, f"frame_{i}")
        fpath = os.path.join(d, f"{video_id}.jpg")
        if not os.path.exists(fpath): return None
        if file_size(fpath) <= 0: return None
        frames.append(fpath)
    return frames

def load_json_list(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta["data"]

# ---------------------------
# Worker (pack one)
# ---------------------------

def _pack_one(sample: Dict[str, Any], num_frames: int, store: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Return:
      ok, record (dict of scalar/list for Arrow), err_msg
    """
    try:
        video_id        = sample["video_id"]
        video_path      = sample["video_path"]
        wav_path        = sample["wav"]
        raw_video_path  = sample.get("raw_video_path", "")
        labels          = sample.get("labels", "")

        # labels -> list[str]
        if isinstance(labels, str):
            label_list = [x for x in labels.split(",") if x]
        elif isinstance(labels, list):
            label_list = labels
        else:
            label_list = []

        if store == "bytes":
            frames = gather_frames_bytes(video_id, video_path, num_frames)
        else:
            frames = gather_frames_paths(video_id, video_path, num_frames)
        if frames is None:
            return False, None, f"missing/bad frames for {video_id}"

        if not os.path.exists(wav_path) or file_size(wav_path) <= 0:
            return False, None, f"missing/bad wav for {video_id}"

        if store == "bytes":
            with open(wav_path, "rb") as f:
                audio_blob = f.read()
            record = {
                "key":            video_id,
                "labels":         label_list,
                "num_frames":     num_frames,
                "frames":         frames,        # list<binary>
                "audio":          audio_blob,    # binary
                "raw_video_path": raw_video_path,
                "video_path":     video_path,
                "wav_path":       wav_path,
                "store":          "bytes",
            }
        else:  # paths
            record = {
                "key":            video_id,
                "labels":         label_list,
                "num_frames":     num_frames,
                "frame_paths":    frames,        # list<string>
                "audio_path":     wav_path,      # string
                "raw_video_path": raw_video_path,
                "video_path":     video_path,
                "wav_path":       wav_path,
                "store":          "paths",
            }
        return True, record, ""
    except Exception as e:
        return False, None, f"exception: {e}"

# ---------------------------
# Arrow shard writer
# ---------------------------

class ArrowShardWriter:
    """
    将记录累积到 RecordBatch 后写到 .arrow 分片。
    支持按记录数和估算字节数切分。
    """
    def __init__(self, out_dir: str, shard_prefix: str, max_count: int, max_bytes: int, store: str):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.shard_prefix = shard_prefix
        self.max_count = max_count
        self.max_bytes = max_bytes
        self.store = store

        self.shard_idx = 0
        self.rows: List[Dict[str, Any]] = []
        self.bytes_in_shard = 0

        self.manifest_rows = []  # [(global_index, shard_path, row_in_shard)]

        # 构建 schema
        if store == "bytes":
            self.schema = pa.schema([
                pa.field("key", pa.string()),
                pa.field("labels", pa.list_(pa.string())),
                pa.field("num_frames", pa.int32()),
                pa.field("frames", pa.list_(pa.large_binary())),
                pa.field("audio", pa.large_binary()),
                pa.field("raw_video_path", pa.string()),
                pa.field("video_path", pa.string()),
                pa.field("wav_path", pa.string()),
                pa.field("store", pa.string()),
            ])
        else:
            self.schema = pa.schema([
                pa.field("key", pa.string()),
                pa.field("labels", pa.list_(pa.string())),
                pa.field("num_frames", pa.int32()),
                pa.field("frame_paths", pa.list_(pa.string())),
                pa.field("audio_path", pa.string()),
                pa.field("raw_video_path", pa.string()),
                pa.field("video_path", pa.string()),
                pa.field("wav_path", pa.string()),
                pa.field("store", pa.string()),
            ])

        self.global_row_idx = 0  # for manifest

    def _estimate_record_size(self, rec: Dict[str, Any]) -> int:
        # 粗略估算计入 bytes，用于滚动（不必严丝合缝）
        size = 128  # base
        size += sum(len(x) for x in rec.get("frames", [])) if "frames" in rec else 0
        size += len(rec.get("audio", b"")) if "audio" in rec else 0
        # paths 模式，估算极小
        if "frame_paths" in rec:
            size += 4 * len(rec["frame_paths"])
        if "audio_path" in rec:
            size += 32
        return size

    def _flush_shard(self):
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        shard_path = os.path.join(self.out_dir, f"{self.shard_prefix}-{self.shard_idx:06d}.arrow")
        with ipc.new_file(shard_path, table.schema) as writer:
            # 一次写入（也可按 batch 写，简化起见统一写）
            writer.write_table(table)

        # 写 manifest 条目
        for i in range(len(self.rows)):
            self.manifest_rows.append({
                "global_index": self.global_row_idx - len(self.rows) + i,
                "shard_path": shard_path,
                "row_in_shard": i,
            })

        # 准备下一个分片
        self.shard_idx += 1
        self.rows.clear()
        self.bytes_in_shard = 0

    def add(self, record: Dict[str, Any]):
        self.rows.append(record)
        self.global_row_idx += 1
        self.bytes_in_shard += self._estimate_record_size(record)
        need_roll = False
        if self.max_count > 0 and len(self.rows) >= self.max_count:
            need_roll = True
        if self.max_bytes > 0 and self.bytes_in_shard >= self.max_bytes:
            need_roll = True
        if need_roll:
            self._flush_shard()

    def close(self, manifest_path: str):
        self._flush_shard()
        # 写 manifest.parquet
        if self.manifest_rows:
            mtable = pa.Table.from_pylist(self.manifest_rows, schema=pa.schema([
                pa.field("global_index", pa.int64()),
                pa.field("shard_path", pa.string()),
                pa.field("row_in_shard", pa.int32()),
            ]))
            pq.write_table(mtable, manifest_path)

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="/data/wanglinge/project/weighted-cav-mae/src/data_info/as2M/unbalanced_145w.json", help="标注 JSON 路径（含 data 列表）")
    parser.add_argument("--out", default="/data/wanglinge/dataset/AudioSet2M/arrow/train", help="输出目录，用于存放 shard-*.arrow 与 manifest.parquet")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--shard-max-count", type=int, default=1000)
    parser.add_argument("--shard-max-bytes", type=str, default="0")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-prefix", type=str, default="shard")
    parser.add_argument("--store", choices=["bytes", "paths"], default="bytes",
                        help="bytes: 将帧与音频以二进制写入 Arrow；paths: 仅写路径，训练端懒加载")
    args = parser.parse_args()

    max_bytes = parse_bytes(args.shard_max_bytes)
    items = load_json_list(args.json)
    if args.limit > 0:
        items = items[:args.limit]

    # rng = random.Random(args.seed)
    # rng.shuffle(items)

    os.makedirs(args.out, exist_ok=True)
    writer = ArrowShardWriter(args.out, args.shard_prefix, args.shard_max_count, max_bytes, store=args.store)

    pack_fn = partial(_pack_one, num_frames=args.num_frames, store=args.store)

    ok_cnt, total = 0, len(items)
    errors = []

    if args.workers > 1:
        with Pool(processes=args.workers) as pool:
            for ok, rec, err in tqdm(pool.imap_unordered(pack_fn, items, chunksize=64), total=total, desc="pack"):
                if not ok:
                    errors.append(err)
                    continue
                writer.add(rec)
                ok_cnt += 1
    else:
        for s in tqdm(items, total=total, desc="pack"):
            ok, rec, err = pack_fn(s)
            if not ok:
                errors.append(err); continue
            writer.add(rec)
            ok_cnt += 1

    manifest_path = os.path.join(args.out, "manifest.parquet")
    writer.close(manifest_path)

    print(f"[done] OK samples: {ok_cnt} / {total}")
    if errors:
        print(f"[warn] {len(errors)} samples failed. Show first 20:")
        for e in errors[:20]:
            print("  -", e)

if __name__ == "__main__":
    main()
