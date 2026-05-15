# 由于pending3系列存在大量异常长度视频，这里将他们删去：
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import subprocess
from tqdm import tqdm

# ===== 参数设置 =====
INPUT_JSON = "/data/wanglinge/project/weighted-cav-mae/src/data_info/as2M/pending/pending3_valid.json"
OUT_VALID = "/data/wanglinge/project/weighted-cav-mae/src/data_info/as2M/pending/pending3_filter_valid.json.json"
OUT_INVALID = "/data/wanglinge/project/weighted-cav-mae/src/data_info/as2M/pending/pending3_error.json.json"
THRESHOLD = 15.0  # 秒

# ===== 辅助函数 =====
def get_video_duration(path: str) -> float:
    """使用 ffprobe 获取视频时长（秒）。失败则返回 -1。"""
    if not os.path.exists(path):
        return -1.0
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return -1.0

# ===== 主逻辑 =====
def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        meta = json.load(f)

    items = meta["data"]
    valid, invalid = [], []

    for sample in tqdm(items, desc="checking durations"):
        raw_path = sample.get("raw_video_path", "")
        dur = get_video_duration(raw_path)
        if dur <= 0:
            # 文件不存在或无法读取，视为异常
            invalid.append(sample | {"duration": dur})
        elif dur > THRESHOLD:
            invalid.append(sample | {"duration": dur})
        else:
            valid.append(sample | {"duration": dur})

    print(f"[done] valid: {len(valid)}, invalid: {len(invalid)}")

    with open(OUT_VALID, "w", encoding="utf-8") as f:
        json.dump({"data": valid}, f, ensure_ascii=False, indent=2)
    with open(OUT_INVALID, "w", encoding="utf-8") as f:
        json.dump({"data": invalid}, f, ensure_ascii=False, indent=2)

    print(f"→ Saved: {OUT_VALID} and {OUT_INVALID}")
    
if __name__ == "__main__":
    main()
