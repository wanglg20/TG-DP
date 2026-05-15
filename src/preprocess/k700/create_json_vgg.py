import csv
import json
import os

input_csv = "/data/wanglinge/dataset/VGGSound/vggsound.csv"
output_json = "/data/wanglinge/project/weighted-cav-mae/src/data_info/vgg/vggsound_train.json"

video_root = "/data/VGGSound/video"
frame_root = "/data/VGGSound/frames_16/test"
audio_root = "/data/VGGSound/audio/test"

results = {"data": []}

with open(input_csv, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        vid = row["video_id"]
        label = row["labels"]

        entry = {
            "video_id": vid,
            "raw_video_path": os.path.join(video_root, f"{vid}.mp4"),
            "video_path": frame_root,
            "wav": os.path.join(audio_root, f"{vid}.wav"),
            "labels": label
        }
        results["data"].append(entry)

with open(output_json, "w") as f:
    json.dump(results, f, indent=4)
