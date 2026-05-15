import os
import json
import shutil
from tqdm import tqdm

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def reorganize(json_in, json_out, files_per_folder=10000):
    """
    在原地将文件分级存储，每 10k 一个子目录，并更新 JSON 文件
    """

    with open(json_in, "r") as f:
        data = json.load(f)

    new_entries = []
    frames_dir = ["frames_{}".format(i) for i in range(16)]  
    for idx, entry in enumerate(tqdm(data["data"], desc="Processing entries")):
        vid = entry["video_id"]

        # ==== 音频处理 ====
        old_wav = entry["wav"]

        wav_idx = idx // files_per_folder
        new_wav_dir = os.path.join(os.path.dirname(old_wav), f"{wav_idx:04d}")
        ensure_dir(new_wav_dir)
        new_wav = os.path.join(new_wav_dir, os.path.basename(old_wav))
        if not os.path.exists(new_wav):
            if not os.path.exists(old_wav):
                print(f"⚠️ 音频缺失: {old_wav}")
            else:
                if old_wav != new_wav:
                    os.rename(old_wav, new_wav)
        entry["wav"] = new_wav

        # ==== 帧图像处理 ====
        old_frame_dir = entry["video_path"]
        if not os.path.exists(old_frame_dir):
            print(f"⚠️ 帧目录缺失: {old_frame_dir}")
        else:
            frame_idx = idx // files_per_folder
            new_frame_dir = os.path.join(old_frame_dir, f"{frame_idx:04d}")
            jpg_name = f"{vid}.jpg"
            for i in range(16):
                old_jpg = os.path.join(old_frame_dir, f"frame_{i}", jpg_name)
                new_jpg_dir = os.path.join(new_frame_dir, f"frame_{i}")
                ensure_dir(new_jpg_dir)
                new_jpg = os.path.join(new_jpg_dir, jpg_name)
                exist_jpg = os.path.exists(new_jpg)
                if not os.path.exists(old_jpg) and not exist_jpg:
                    print(f"⚠️ 帧图像缺失: {old_jpg}")
                    continue
                if old_jpg != new_jpg and not exist_jpg:
                    os.rename(old_jpg, new_jpg)
            entry["video_path"] = new_frame_dir

        new_entries.append(entry)

    new_json = {"data": new_entries}
    with open(json_out, "w") as f:
        json.dump(new_json, f, indent=4)

    print(f"✅ Done. New json saved to {json_out}")


if __name__ == "__main__":
    json_in = "/data/wanglinge/project/weighted-cav-mae/src/data_info/as2M/unbalanced_partial_.json"
    json_out = "/data/wanglinge/project/weighted-cav-mae/src/data_info/as2M/unbalanced_partial_partition.json"

    reorganize(json_in, json_out, files_per_folder=5000)
