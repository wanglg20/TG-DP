import csv
import pandas as pd
from pathlib import Path
import json
import os 

ori_csv = '/data/wanglinge/project/weighted-cav-mae/src/data_info/as2M/pending/merged_pending2_fixed.csv'
img_root = '/data/wanglinge/dataset/AudioSet2M/pending/frames'
audio_root = '/data/wanglinge/dataset/AudioSet2M/pending/audio'
out_csv = '/data/wanglinge/project/weighted-cav-mae/src/data_info/as2M/pending/pending_valid.csv'
out_json = '/data/wanglinge/project/weighted-cav-mae/src/data_info/as2M/pending/pending_valid.json'
BASE = Path("/data/wanglinge/dataset/Fast-Audioset-Download")
PENDING_DIR = BASE / "cleaned_csvs/pending2/need_preprocess"
ORIG_DIR = BASE / "cleaned_csvs/split_chunks"

def load_original_labels_map():
    """
    从 split_chunks 下 45 个原始 csv 构建 {YTID: labels_str} 映射。
    labels_str 形如：/m/01xqw,/m/04rlf  （不含外层引号；写出时 csv 会自动加引号）
    """
    ytid2labels = {}
    for i in range(1, 46):
        f = ORIG_DIR / f"unbalanced_train_segments_split_{i:02d}.csv"
        if not f.exists():
            print(f"[WARN] Missing original CSV: {f}")
            continue
        with f.open("r", newline="", encoding="utf-8") as fp:
            reader = csv.reader(fp)
            for row in reader:
                if not row:
                    continue
                # 原始前几行是以#开头的注释，跳过
                first = row[0].strip()
                if first.startswith("#"):
                    continue
                # 预期四列：YTID, start_seconds, end_seconds, positive_labels
                # 使用 csv 解析后的第4列会是去掉双引号的字符串，例如 /m/01xqw,/m/04rlf
                if len(row) < 4:
                    continue
                ytid = row[0].strip()
                num_labels = len(row) - 3
                labels = ",".join(r.strip() for r in row[3:3+num_labels])
                ytid2labels[ytid] = labels
    return ytid2labels

def check_img(img_root, video_id):
    img_name = video_id + '.jpg'
    num_frmaes = 16
    for i in range(num_frmaes):
        img_path = os.path.join(img_root, 'frame_{}'.format(i), img_name)
        if not os.path.exists(img_path):
            return False
    return True

if __name__ == '__main__':
    ytid2labels = load_original_labels_map()
    ori_data = pd.read_csv(ori_csv, header=None)
    input_filelist = ori_data.iloc[:, 0].tolist()
    data_list = []
    valid_count = 0
    missing_img = 0
    missing_wav = 0
    for video_path in input_filelist:
        video_id = video_path.split('/')[-1].split('.')[0]
        wav_path = str(Path(audio_root) / f"{video_id}.wav")
        img_path = img_root
        if video_id[3:] not in ytid2labels:
            continue
        labels = ytid2labels[video_id[3:]]
        if not os.path.exists(wav_path):
            #print(f"[WARN] Missing audio file: {wav_path}")
            missing_wav += 1
            continue
        if not check_img(img_root, video_id):
            #print(f"[WARN] Missing image files for video ID: {video_id}")
            missing_img += 1
            continue

        valid_count += 1
        data_list.append({
            "video_id": video_id,
            "raw_video_path":video_path,
            "video_path": img_path,
            "wav": wav_path,
            "labels": labels
        })

    print(f"Total valid entries: {valid_count}")
    print(f"Total missing audio files: {missing_wav}")
    print(f"Total missing image files: {missing_img}")

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({"data": data_list}, f, indent=2)
