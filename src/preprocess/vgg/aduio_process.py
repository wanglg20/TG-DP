#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

def check_sox():
    try:
        subprocess.run(["sox", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except Exception:
        return False

def build_pairs(root: Path):
    """在root下递归查找 *_intermediate.wav，返回 (src, dst) 列表。"""
    pairs = []
    for p in root.rglob("*_intermediate.wav"):
        dst = p.with_name(p.name.replace("_intermediate.wav", ".wav"))
        pairs.append((p, dst))
    return pairs

def process_one(src: Path, dst: Path, overwrite: bool=False, remove_intermediate_on_success: bool=False) -> tuple[Path, bool, str]:
    """调用 sox 提取第1声道到 dst。返回 (src, ok, msg)。"""
    try:
        if dst.exists() and not overwrite:
            return (src, True, "exists, skipped")
        dst.parent.mkdir(parents=True, exist_ok=True)

        # sox "src" "dst" remix 1
        cmd = ["sox", str(src), str(dst), "remix", "1"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0 or (not dst.exists()):
            return (src, False, proc.stderr.strip() or "unknown error")

        if remove_intermediate_on_success:
            try:
                src.unlink(missing_ok=True)
            except Exception as e:
                # 不影响整体成功，只记录
                return (src, True, f"ok, but rm intermediate failed: {e}")

        return (src, True, "ok")
    except Exception as e:
        return (src, False, str(e))

def main():
    parser = argparse.ArgumentParser(description="Batch convert *_intermediate.wav -> *.wav using sox remix 1 (multithread + tqdm).")
    parser.add_argument("--root", default="/data/wanglinge/dataset/AudioSet2M/audio",
                        help="根目录，递归处理该目录下所有 *_intermediate.wav")
    parser.add_argument("--workers", type=int, default=16,
                        help="并发线程数（默认=CPU核数）")
    parser.add_argument("--overwrite", action="store_true",
                        help="若目标 *.wav 已存在则覆盖")
    
    parser.add_argument("--log", default=None,
                        help="失败日志保存路径（默认不保存）")
    args = parser.parse_args()

    if not check_sox():
        print("ERROR: 未检测到 sox。请先安装：sudo apt-get install sox", file=sys.stderr)
        sys.exit(1)
    args.remove_intermediate = True
    root = Path(args.root)
    pairs = build_pairs(root)
    if not pairs:
        print(f"在 {root} 下未找到 *_intermediate.wav 文件。")
        return

    print(f"发现 {len(pairs)} 个待处理文件，开始转换...")

    failures = []
    exists_skipped = 0
    ok_count = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(process_one, src, dst, args.overwrite)
                   for src, dst in pairs]
        for fut in tqdm(as_completed(futures), total=len(futures), ncols=100):
            src, ok, msg = fut.result()
            if ok:
                ok_count += 1
                if msg.startswith("exists"):
                    exists_skipped += 1
            else:
                failures.append((str(src), msg))

    print(f"\n完成：总计 {len(pairs)} | 成功 {ok_count} | 已存在跳过 {exists_skipped} | 失败 {len(failures)}")
    if failures:
        print("示例失败项（前10条）：")
        for p, m in failures[:10]:
            print(f" - {p}: {m}")

    if args.log and failures:
        logp = Path(args.log)
        logp.parent.mkdir(parents=True, exist_ok=True)
        with open(logp, "w", encoding="utf-8") as f:
            for p, m in failures:
                f.write(f"{p}\t{m}\n")
        print(f"失败日志已写入：{logp}")

if __name__ == "__main__":
    main()
