#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import argparse
from collections import OrderedDict

def gen_video_id_candidates(basename_noext: str):
    """
    从原始文件名生成一组候选 video_id，按顺序尝试：
    1) 原始 basename（含 id_ 前缀）
    2) 去掉前缀 'id_'（如果有）
    3) 1) + '_000000'（如果原始不以 _000000 结尾）
    4) 2) + '_000000'（如果第二种不以 _000000 结尾）
    """
    cands = []
    b0 = basename_noext
    b1 = basename_noext[3:] if basename_noext.startswith("id_") else basename_noext

    def add(x):
        if x not in cands:
            cands.append(x)

    add(b0)
    add(b1)
    if not b0.endswith("_000000"):
        add(b0 + "_000000")
    if not b1.endswith("_000000"):
        add(b1 + "_000000")
    return cands

def frames_exist(frames_root: str, vid: str, num_frames: int = 16):
    """
    检查 frame_0 ... frame_{num_frames-1}/vid.jpg 是否全部存在
    """
    for i in range(num_frames):
        p = os.path.join(frames_root, f"frame_{i}", f"{vid}.jpg")
        if not os.path.isfile(p):
            return False
    return True

def audio_exist(audio_root: str, vid: str):
    return os.path.isfile(os.path.join(audio_root, f"{vid}.wav"))

def resolve_video_id(frames_root: str, audio_root: str, basename_noext: str, num_frames: int = 16):
    """
    依次尝试候选 video_id，找到同时满足帧与音频存在的那个；找不到则返回 None
    """
    for vid in gen_video_id_candidates(basename_noext):
        if frames_exist(frames_root, vid, num_frames) and audio_exist(audio_root, vid):
            return vid
    return None

def parse_csv_line(row):
    """
    你的 CSV 例子为： video_path, ,labels
    有时 labels 被引号包裹且内部含逗号。
    这里用 csv 模块的解析结果：取第一列为 path，最后一列为 labels。
    """
    # 移除空白列
    fields = [x.strip() for x in row if x is not None]
    if not fields:
        return None, None
    path = fields[0]
    labels = fields[-1] if len(fields) >= 2 else ""
    return path, labels

def main():
    parser = argparse.ArgumentParser(description="Clean AudioSet CSV to filtered JSON with existence checks.")
    parser.add_argument("--csv", default="/data/wanglinge/project/weighted-cav-mae/src/data_info/as2M/pending/merged_pending3_fixed.csv", help="Input CSV file path")
    parser.add_argument("--out", default="/data/wanglinge/project/weighted-cav-mae/src/data_info/as2M/pending/pending3_valid.json", help="Output JSON file path")
    parser.add_argument("--frames_root", default="/data/wanglinge/dataset/AudioSet2M/pending3/frames",
                        help="Root dir for frames (contains frame_0 ... frame_15)")
    parser.add_argument("--audio_root", default="/data/wanglinge/dataset/AudioSet2M/pending3/audio",
                        help="Root dir for audios (.wav)")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to check (default: 16)")
    parser.add_argument("--dialect", default="excel", help="CSV dialect passed to csv.reader (default: excel)")
    parser.add_argument("--encoding", default="utf-8", help="CSV file encoding (default: utf-8)")
    args = parser.parse_args()

    kept, skipped, total = 0, 0, 0
    data = []

    with open(args.csv, "r", encoding=args.encoding, newline="") as f:
        reader = csv.reader(f, dialect=args.dialect)
        # 如果第一行是表头且不规则，可以尝试检测并跳过
        header_peek = next(reader, None)
        if header_peek is None:
            print("Empty CSV.")
            return
        # 判断是否是表头：包含 'video' 和 'label' 等关键字
        header_is_header = any(k in (header_peek[0] or "").lower() for k in ["video", "path"]) or \
                           any("label" in (x or "").lower() for x in header_peek)
        if not header_is_header:
            # 第一行就是数据，放回处理
            row = header_peek
            path, labels = parse_csv_line(row)
            if path:
                reader = (r for r in ([row] + list(reader)))
        # 否则，header 已经被消耗，继续读剩余行

        for row in reader:
            total += 1
            path, labels = parse_csv_line(row)
            if not path:
                skipped += 1
                continue

            # 从原始路径中获得 basename（不含扩展名）
            base = os.path.splitext(os.path.basename(path))[0]  # 例如: id_--U7joUcTCo
            vid = resolve_video_id(args.frames_root, args.audio_root, base, args.num_frames)
            if vid is None:
                skipped += 1
                continue
            
            item = OrderedDict()
            item["video_id"] = vid
            item["raw_video_path"] = path
            item["video_path"] = args.frames_root
            item["wav"] = os.path.join(args.audio_root, f"{vid}.wav")
            # 直接保留原 CSV 的 labels 字符串（如果你需要映射，可在这里替换）
            item["labels"] = labels

            data.append(item)
            kept += 1

    out_obj = {"data": data}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as wf:
        json.dump(out_obj, wf, ensure_ascii=False, indent=4)

    print(f"Done. total={total}, kept={kept}, skipped={skipped}")
    print(f"Output -> {args.out}")

if __name__ == "__main__":
    main()
