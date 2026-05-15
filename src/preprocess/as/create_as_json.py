import csv
import json
import os

target_audio_dir = "/data/wanglinge/dataset/audioset/raw_data/unbal_train/audio/unbal_train" 

# 输入CSV和输出JSON文件路径
input_csv = "/data/wanglinge/project/cav-mae/src/data/info/as/data/unbalanced_train_segments.csv"
output_json = "/data/wanglinge/project/cav-mae/src/data/info/as/data/unbalanced_train_segments.json"

data_list = []

with open(input_csv, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        # 跳过注释行或空行
        if not row or row[0].startswith('#'):
            continue
        
        video_id = row[0].strip()
        start_sec = row[1].strip()
        end_sec = row[2].strip()
        labels = ','.join(row[3:]).strip().strip('"')  # 合并剩下所有部分为labels

        wav_path = os.path.join(target_audio_dir, f"{video_id}.flac")

        data_list.append({
            "video_id": video_id,
            "wav": wav_path,
            "labels": labels
        })

# 保存JSON文件
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump({"data": data_list}, f, indent=2)

print(f"成功保存至 {output_json}")
